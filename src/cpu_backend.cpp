// CPU reference/fast path: fp32 activations, FP8/BF16 weights decoded on the fly (AVX-512 when available).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "backend.h"
#include "cpu_kernels.h"
#include "threadpool.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define QW_AVX512 1
#endif

namespace {

using namespace ck;

// Section timers, printed at exit when QW_PROFILE is set.
struct Prof {
    const char* names[16] = {};
    double secs[16] = {};
    ~Prof() {
        if (!std::getenv("QW_PROFILE")) return;
        for (int i = 0; i < 16; i++)
            if (names[i]) std::fprintf(stderr, "  %-14s %8.3f s\n", names[i], secs[i]);
    }
} g_prof;
struct ProfScope {
    int id;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ProfScope(int i, const char* n) : id(i) { g_prof.names[i] = n; }
    ~ProfScope() { g_prof.secs[id] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

// ---------------------------------------------------------------------------------------------
// CPU backend
// ---------------------------------------------------------------------------------------------
class CpuBackend final : public Backend {
public:
    CpuBackend(const Config& c, const ModelWeights& w, int max_ctx, int threads)
        : c_(c), w_(w), max_ctx_(max_ctx), pool_(threads) {
        for (int l = 0; l < c.n_layers; l++) {
            if (c.is_full_attn[l]) {
                slot_[l] = n_full_++;
            } else {
                slot_[l] = n_lin_++;
            }
        }
        const size_t kv = (size_t)max_ctx * c.n_kv_heads * c.head_dim;
        kcache_.assign(n_full_ * kv, 0.f);
        vcache_.assign(n_full_ * kv, 0.f);
        conv_state_.assign((size_t)n_lin_ * (c.conv_kernel - 1) * c.conv_dim(), 0.f);
        rec_state_.assign((size_t)n_lin_ * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, 0.f);
        // rope inverse frequencies
        for (int i = 0; i < c.rope_dim / 2; i++)
            inv_freq_.push_back((float)(1.0 / std::pow((double)c.rope_theta, (2.0 * i) / c.rope_dim)));
    }

    void reset() override {
        pos_ = 0;
        std::fill(conv_state_.begin(), conv_state_.end(), 0.f);
        std::fill(rec_state_.begin(), rec_state_.end(), 0.f);
    }
    int pos() const override { return pos_; }
    int max_ctx() const override { return max_ctx_; }
    const char* name() const override { return "cpu"; }
    void save_state() override {
        saved_conv_ = conv_state_;
        saved_rec_ = rec_state_;
        saved_pos_ = pos_;
    }
    void restore_state() override {
        conv_state_ = saved_conv_;
        rec_state_ = saved_rec_;
        pos_ = saved_pos_;
    }

    void forward(const int* tokens, int n, float* logits) override {
        if (pos_ + n > max_ctx_) throw std::runtime_error("context overflow");
        const int H = c_.hidden;
        x_.resize((size_t)n * H);
        for (int t = 0; t < n; t++) {
            const uint16_t* e = w_.embed.w + (size_t)tokens[t] * H;
            for (int i = 0; i < H; i++) x_[(size_t)t * H + i] = bf16_to_f32(e[i]);
        }
        for (int l = 0; l < c_.n_layers; l++) layer(l, n);
        // final norm + lm_head for the last token only
        std::vector<float> xn(H);
        rmsnorm(&x_[(size_t)(n - 1) * H], w_.final_norm, xn.data(), H, c_.rms_eps, true);
        ProfScope ps(7, "lm_head");
        run_jobs(pool_, {MatJob{nullptr, &w_.lm_head, xn.data(), logits, 1}});
        pos_ += n;
    }

private:
    const Config& c_;
    const ModelWeights& w_;
    int max_ctx_;
    int pos_ = 0;
    ThreadPool pool_;
    int slot_[256] = {};
    int n_full_ = 0, n_lin_ = 0;
    std::vector<float> kcache_, vcache_, conv_state_, rec_state_, inv_freq_;
    std::vector<float> saved_conv_, saved_rec_;
    int saved_pos_ = 0;
    std::vector<float> x_, xn_, a_, b_, c1_, c2_, c3_, out_;

    void layer(int l, int n) {
        const LayerWeights& L = w_.layers[l];
        const int H = c_.hidden;
        xn_.resize((size_t)n * H);
        out_.resize((size_t)n * H);
        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) rmsnorm(&x_[t * H], L.input_norm, &xn_[t * H], H, c_.rms_eps, true);
        });
        if (L.full_attn) attention(l, n);
        else deltanet(l, n);
        for (size_t i = 0; i < (size_t)n * H; i++) x_[i] += out_[i];

        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) rmsnorm(&x_[t * H], L.post_norm, &xn_[t * H], H, c_.rms_eps, true);
        });
        moe(l, n);
        for (size_t i = 0; i < (size_t)n * H; i++) x_[i] += out_[i];
    }

    // ---- Gated DeltaNet --------------------------------------------------------------------
    void deltanet(int l, int n) {
        const LayerWeights& L = w_.layers[l];
        const int CD = c_.conv_dim(), VD = c_.lin_value_dim(), NV = c_.lin_v_heads, KD = c_.lin_k_dim,
                  DV = c_.lin_v_dim, K = c_.conv_kernel;
        std::vector<float>& qkv = a_;  // [n, CD] -> conv output in place
        std::vector<float>& z = b_;    // [n, VD]
        qkv.resize((size_t)n * CD);
        z.resize((size_t)n * VD);
        std::vector<float> av((size_t)n * NV), bv((size_t)n * NV);
        { ProfScope ps(0, "dn.in_proj");
        run_jobs(pool_, {MatJob{&L.in_qkv, nullptr, xn_.data(), qkv.data(), n},
                         MatJob{&L.in_z, nullptr, xn_.data(), z.data(), n},
                         MatJob{nullptr, &L.in_a, xn_.data(), av.data(), n},
                         MatJob{nullptr, &L.in_b, xn_.data(), bv.data(), n}}); }
        ProfScope ps1(1, "dn.conv+rec");

        // causal depthwise conv1d + silu, sequential over tokens per channel
        float* cs = &conv_state_[(size_t)slot_[l] * (K - 1) * CD];  // [K-1, CD], oldest first
        pool_.parallel_for(CD, [&](int64_t b, int64_t e) {
            for (int64_t ch = b; ch < e; ch++) {
                float wk[8], hist[8];
                for (int k = 0; k < K; k++) wk[k] = L.conv1d.at(ch * K + k);
                for (int k = 0; k < K - 1; k++) hist[k] = cs[(size_t)k * CD + ch];
                for (int t = 0; t < n; t++) {
                    float in = qkv[(size_t)t * CD + ch];
                    float acc = wk[K - 1] * in;
                    for (int k = 0; k < K - 1; k++) acc += wk[k] * hist[k];
                    for (int k = 0; k < K - 2; k++) hist[k] = hist[k + 1];
                    hist[K - 2] = in;
                    qkv[(size_t)t * CD + ch] = silu(acc);
                }
                for (int k = 0; k < K - 1; k++) cs[(size_t)k * CD + ch] = hist[k];
            }
        }, 256);

        // gated delta rule recurrence, one v-head per task
        std::vector<float>& core = c1_;
        core.resize((size_t)n * VD);
        float* S0 = &rec_state_[(size_t)slot_[l] * NV * KD * DV];
        const int ratio = NV / c_.lin_k_heads, KEYD = c_.lin_key_dim();
        pool_.parallel_for(NV, [&](int64_t hb, int64_t he) {
            std::vector<float> q(KD), k(KD);
            for (int64_t h = hb; h < he; h++) {
                float* S = S0 + (size_t)h * KD * DV;  // [KD][DV]
                const int kh = (int)h / ratio;
                const float A = -std::exp(L.a_log.at(h));
                const float dtb = L.dt_bias.at(h);
                for (int t = 0; t < n; t++) {
                    const float* row = &qkv[(size_t)t * CD];
                    const float* qs = row + kh * KD;
                    const float* ks = row + KEYD + kh * KD;
                    const float* vs = row + 2 * KEYD + h * DV;
                    float qn = 0, kn = 0;
                    for (int i = 0; i < KD; i++) { qn += qs[i] * qs[i]; kn += ks[i] * ks[i]; }
                    qn = 1.0f / std::sqrt(qn + 1e-6f) / std::sqrt((float)KD);
                    kn = 1.0f / std::sqrt(kn + 1e-6f);
                    for (int i = 0; i < KD; i++) { q[i] = qs[i] * qn; k[i] = ks[i] * kn; }
                    const float beta = sigmoid(bv[(size_t)t * NV + h]);
                    const float decay = std::exp(A * softplus(av[(size_t)t * NV + h] + dtb));
                    float* o = &core[(size_t)t * VD + h * DV];
                    delta_step(S, k.data(), q.data(), vs, beta, decay, o, KD, DV);
                    // gated rmsnorm (weight not offset by 1) with silu(z) gate
                    float ss = 0;
                    for (int j = 0; j < DV; j++) ss += o[j] * o[j];
                    const float r = 1.0f / std::sqrt(ss / DV + c_.rms_eps);
                    const float* zs = &z[(size_t)t * VD + h * DV];
                    for (int j = 0; j < DV; j++) o[j] = o[j] * r * L.lin_norm.at(j) * silu(zs[j]);
                }
            }
        });
        ProfScope ps2(2, "dn.out_proj");
        run_jobs(pool_, {MatJob{&L.out_proj, nullptr, core.data(), out_.data(), n}});
    }

    // ---- Gated full attention --------------------------------------------------------------
    void rope(float* v, int p) {
        const int half = c_.rope_dim / 2;
        for (int i = 0; i < half; i++) {
            float ang = (float)p * inv_freq_[i], cs = std::cos(ang), sn = std::sin(ang);
            float a = v[i], b = v[i + half];
            v[i] = a * cs - b * sn;
            v[i + half] = b * cs + a * sn;
        }
    }

    void attention(int l, int n) {
        const LayerWeights& L = w_.layers[l];
        const int NH = c_.n_heads, NKV = c_.n_kv_heads, D = c_.head_dim;
        std::vector<float>& q = a_;  // [n, NH, 2*D]  (query | gate) per head
        std::vector<float>& k = b_;  // [n, NKV*D]
        std::vector<float>& v = c1_;
        q.resize((size_t)n * NH * 2 * D);
        k.resize((size_t)n * NKV * D);
        v.resize((size_t)n * NKV * D);
        ProfScope ps(3, "attention");
        run_jobs(pool_, {MatJob{&L.q_proj, nullptr, xn_.data(), q.data(), n},
                         MatJob{&L.k_proj, nullptr, xn_.data(), k.data(), n},
                         MatJob{&L.v_proj, nullptr, xn_.data(), v.data(), n}});
        const size_t kvs = (size_t)max_ctx_ * NKV * D;
        float* KC = &kcache_[slot_[l] * kvs];
        float* VC = &vcache_[slot_[l] * kvs];
        for (int t = 0; t < n; t++) {
            const int p = pos_ + t;
            for (int h = 0; h < NH; h++) {
                float* qh = &q[((size_t)t * NH + h) * 2 * D];
                rmsnorm(qh, L.q_norm, qh, D, c_.rms_eps, true);
                rope(qh, p);
            }
            for (int h = 0; h < NKV; h++) {
                float* kh = &k[((size_t)t * NKV + h) * D];
                rmsnorm(kh, L.k_norm, kh, D, c_.rms_eps, true);
                rope(kh, p);
                std::memcpy(&KC[((size_t)p * NKV + h) * D], kh, D * sizeof(float));
                std::memcpy(&VC[((size_t)p * NKV + h) * D], &v[((size_t)t * NKV + h) * D], D * sizeof(float));
            }
        }
        std::vector<float>& att = c2_;  // [n, NH*D]
        att.resize((size_t)n * NH * D);
        const float scale = 1.0f / std::sqrt((float)D);
        const int group = NH / NKV;
        pool_.parallel_for((int64_t)n * NH, [&](int64_t b, int64_t e) {
            std::vector<float> sc;
            for (int64_t i = b; i < e; i++) {
                const int t = (int)(i / NH), h = (int)(i % NH), kvh = h / group, len = pos_ + t + 1;
                const float* qh = &q[((size_t)t * NH + h) * 2 * D];
                const float* gate = qh + D;
                sc.resize(len);
                float mx = -1e30f;
                for (int j = 0; j < len; j++) {
                    sc[j] = dot_f32(qh, &KC[((size_t)j * NKV + kvh) * D], D) * scale;
                    mx = std::max(mx, sc[j]);
                }
                float sum = 0;
                for (int j = 0; j < len; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                float* o = &att[((size_t)t * NH + h) * D];
                std::fill(o, o + D, 0.f);
                for (int j = 0; j < len; j++) axpy(o, sc[j] / sum, &VC[((size_t)j * NKV + kvh) * D], D);
                for (int d = 0; d < D; d++) o[d] *= sigmoid(gate[d]);
            }
        });
        run_jobs(pool_, {MatJob{&L.o_proj, nullptr, att.data(), out_.data(), n}});
    }

    // ---- Mixture of experts ----------------------------------------------------------------
    void moe(int l, int n) {
        const LayerWeights& L = w_.layers[l];
        const int H = c_.hidden, E = c_.n_experts, TK = c_.top_k, I = c_.moe_inter, SI = c_.shared_inter;
        ProfScope ps(4, "moe.total");
        std::vector<float> logits((size_t)n * E), sgate(n), sg((size_t)n * SI), su((size_t)n * SI);
        run_jobs(pool_, {MatJob{nullptr, &L.router, xn_.data(), logits.data(), n},
                         MatJob{nullptr, &L.shared_gate, xn_.data(), sgate.data(), n},
                         MatJob{&L.shared.gate, nullptr, xn_.data(), sg.data(), n},
                         MatJob{&L.shared.up, nullptr, xn_.data(), su.data(), n}});

        // routing (parallel over tokens), then group tokens per expert
        std::vector<int> top_ids((size_t)n * TK);
        std::vector<float> top_w((size_t)n * TK);
        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) route_topk(&logits[(size_t)t * E], E, TK, &top_ids[t * TK], &top_w[t * TK]);
        });
        std::vector<std::vector<std::pair<int, float>>> per_expert(E);  // (token, weight)
        for (int t = 0; t < n; t++)
            for (int k = 0; k < TK; k++) per_expert[top_ids[(size_t)t * TK + k]].push_back({t, top_w[(size_t)t * TK + k]});

        // gather inputs per active expert
        struct Act { int e; int n; size_t off; };
        std::vector<Act> acts;
        size_t total = 0;
        for (int e = 0; e < E; e++)
            if (!per_expert[e].empty()) { acts.push_back({e, (int)per_expert[e].size(), total}); total += per_expert[e].size(); }
        std::vector<float> xin(total * H), g(total * I), u(total * I), y(total * H);
        pool_.parallel_for((int64_t)acts.size(), [&](int64_t b, int64_t e) {
            for (int64_t k = b; k < e; k++)
                for (int i = 0; i < acts[k].n; i++)
                    std::memcpy(&xin[(acts[k].off + i) * H], &xn_[(size_t)per_expert[acts[k].e][i].first * H], H * sizeof(float));
        });

        std::vector<MatJob> jobs;
        for (auto& a : acts) {
            const Expert& ex = L.experts[a.e];
            jobs.push_back({&ex.gate, nullptr, &xin[a.off * H], &g[a.off * I], a.n});
            jobs.push_back({&ex.up, nullptr, &xin[a.off * H], &u[a.off * I], a.n});
        }
        { ProfScope pe(5, "moe.gate_up"); run_jobs(pool_, jobs); }
        pool_.parallel_for((int64_t)g.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; i++) g[i] = silu(g[i]) * u[i];
        }, 4096);
        for (size_t i = 0; i < sg.size(); i++) sg[i] = silu(sg[i]) * su[i];

        std::vector<float> sy((size_t)n * H);
        jobs.clear();
        for (auto& a : acts) jobs.push_back({&L.experts[a.e].down, nullptr, &g[a.off * I], &y[a.off * H], a.n});
        jobs.push_back({&L.shared.down, nullptr, sg.data(), sy.data(), n});
        { ProfScope pe(6, "moe.down"); run_jobs(pool_, jobs); }

        // per token: (row in y, weight) of its experts
        std::vector<std::vector<std::pair<size_t, float>>> per_tok(n);
        for (auto& a : acts)
            for (int i = 0; i < a.n; i++) per_tok[per_expert[a.e][i].first].push_back({a.off + i, per_expert[a.e][i].second});
        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) {
                const float gs = sigmoid(sgate[t]);
                float* o = &out_[(size_t)t * H];
                for (int i = 0; i < H; i++) o[i] = gs * sy[(size_t)t * H + i];
                for (auto [r, wgt] : per_tok[t]) axpy(o, wgt, &y[r * H], H);
            }
        });
    }
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const Config& cfg, const ModelWeights& w, int max_ctx, int threads) {
    return std::make_unique<CpuBackend>(cfg, w, max_ctx, threads);
}
