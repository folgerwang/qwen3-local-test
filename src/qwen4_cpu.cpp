// CPU backend for the qwen4exp architecture: fp32 activations, GGUF-quantized weights decoded on the fly.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "cpu_kernels.h"
#include "qwen4.h"
#include "qwen4_qsa.h"
#include "threadpool.h"

namespace {

using namespace ck;
using gq::GMat;
using gq::Job;

// out = x * rsqrt(mean(x^2) + eps) * w   (the GGUF gammas already include the reference's "1 +")
inline void rms_mul(const float* x, const float* w, float* out, int n, float eps) {
    const float r = 1.0f / std::sqrt(dot_f32(x, x, n) / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

class Qwen4Cpu final : public Backend {
public:
    Qwen4Cpu(const Q4Config& c, const Q4Weights& w, int max_ctx, int threads)
        : c_(c), w_(w), max_ctx_(max_ctx), pool_(threads), qsa_(c, w, max_ctx, pool_) {
        for (int l = 0; l < c.n_layers; l++) slot_[l] = c.is_full_attn[l] ? n_full_++ : n_lin_++;
        const size_t kv = (size_t)max_ctx * c.n_kv_heads * c.head_dim;
        kcache_.assign(n_full_ * kv, 0.f);
        vcache_.assign(n_full_ * kv, 0.f);
        conv_state_.assign((size_t)n_lin_ * (c.conv_kernel - 1) * c.conv_dim(), 0.f);
        rec_state_.assign((size_t)n_lin_ * c.lin_v_heads * c.lin_k_dim * c.lin_v_dim, 0.f);
        ple_hist_len_ = c.ple_ngram > 0 ? (c.ple_kernel - 1) * c.ple_ngram : 0;
        ple_hist_.assign((size_t)ple_hist_len_ * c.hc_dim(), 0.f);
        toks_.assign(max_ctx, 0);
        for (int i = 0; i < c.rope_dim / 2; i++)
            inv_freq_.push_back((float)(1.0 / std::pow((double)c.rope_theta, (2.0 * i) / c.rope_dim)));
    }

    void reset() override {
        pos_ = 0;
        std::fill(conv_state_.begin(), conv_state_.end(), 0.f);
        std::fill(rec_state_.begin(), rec_state_.end(), 0.f);
        std::fill(ple_hist_.begin(), ple_hist_.end(), 0.f);
    }
    int pos() const override { return pos_; }
    int max_ctx() const override { return max_ctx_; }
    const char* name() const override { return "cpu"; }
    void save_state() override {
        saved_conv_ = conv_state_;
        saved_rec_ = rec_state_;
        saved_ple_ = ple_hist_;
        saved_pos_ = pos_;
    }
    void restore_state() override {
        conv_state_ = saved_conv_;
        rec_state_ = saved_rec_;
        ple_hist_ = saved_ple_;
        pos_ = saved_pos_;
    }

    void forward(const int* tokens, int n, float* logits) override {
        if (pos_ + n > max_ctx_) throw std::runtime_error("context overflow");
        // Chunks bound the activation memory (the residual is hc streams wide); the result is identical.
        constexpr int CHUNK = 512;
        for (int t0 = 0; t0 < n; t0 += CHUNK) {
            const int m = std::min(CHUNK, n - t0);
            chunk(tokens + t0, m, t0 + m == n ? logits : nullptr);
        }
    }

private:
    const Q4Config& c_;
    const Q4Weights& w_;
    int max_ctx_;
    int pos_ = 0;
    ThreadPool pool_;
    QsaIndexer qsa_;
    int slot_[256] = {};
    int n_full_ = 0, n_lin_ = 0, ple_hist_len_ = 0;
    std::vector<float> kcache_, vcache_, conv_state_, rec_state_, ple_hist_, inv_freq_;
    std::vector<float> saved_conv_, saved_rec_, saved_ple_;
    int saved_pos_ = 0;
    std::vector<int> toks_;  // token at every position below pos_ (the PLE n-gram context)
    // x_: residual streams [n, hc, H]; xn_: mixed block input [n, H]; out_: block output [n, H]
    std::vector<float> x_, xnorm_, gate_, lo_, inj_, xn_, out_, a_, b_, c1_, c2_;

    void chunk(const int* tokens, int n, float* logits) {
        const int H = c_.hidden, HC = c_.hc_dim();
        x_.resize((size_t)n * HC);
        for (int t = 0; t < n; t++)
            if (tokens[t] < 0 || tokens[t] >= c_.vocab) throw std::runtime_error("token id out of range");
        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) {
                float* x = &x_[(size_t)t * HC];
                gq::dequant_row(w_.embed.type, w_.embed.row(tokens[t]), H, x);
                for (int s = 1; s < c_.hc; s++) std::memcpy(x + (size_t)s * H, x, H * sizeof(float));
            }
        }, 16);
        for (int t = 0; t < n; t++) toks_[pos_ + t] = tokens[t];

        for (int l = 0; l < c_.n_layers; l++) {
            const Q4Layer& L = w_.layers[l];
            if (L.ple) ple(L, n);
            hc_mix(L.hc_attn, x_.data(), n);
            if (L.full_attn) attention(l, n);
            else deltanet(l, n);
            hc_combine(n);
            hc_mix(L.hc_ffn, x_.data(), n);
            moe(L, n);
            hc_combine(n);
        }
        if (logits) {  // the final mixer is the output norm; only the last token is needed
            hc_mix(w_.hc_head, &x_[(size_t)(n - 1) * HC], 1);
            gq::run_jobs(pool_, {Job{&w_.output, xn_.data(), logits, 1}});
        }
        pos_ += n;
    }

    // ---- Hyper-connections -----------------------------------------------------------------
    // Mixes the hc residual streams x[n, hc, H] into the block input xn_[n, H]; inj_[n, hc] gets the
    // per-stream injection logits (when the mixer has them).
    void hc_mix(const Q4Hc& hc, const float* x, int n) {
        const int H = c_.hidden, HC = c_.hc_dim(), R = c_.hc_rank, S = c_.hc;
        xnorm_.resize((size_t)n * HC);
        gate_.resize((size_t)n * HC);
        lo_.resize((size_t)n * R);
        inj_.resize((size_t)n * S);
        xn_.resize((size_t)n * H);
        pool_.parallel_for((int64_t)n * S, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; i++) rms_mul(x + i * H, &hc.norm[(i % S) * H], &xnorm_[i * H], H, c_.rms_eps);
        }, 4);
        std::vector<Job> jobs{Job{&hc.down, xnorm_.data(), lo_.data(), n}};
        if (hc.inject.data) jobs.push_back(Job{&hc.inject, xnorm_.data(), inj_.data(), n});
        gq::run_jobs(pool_, jobs);
        const float inv = 1.0f / S;
        for (size_t i = 0; i < lo_.size(); i++) lo_[i] = silu(lo_[i] * inv);
        for (size_t i = 0; i < inj_.size(); i++) inj_[i] *= inv;
        gq::run_jobs(pool_, {Job{&hc.up, lo_.data(), gate_.data(), n}});
        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) {
                const float* xs = &xnorm_[(size_t)t * HC];
                const float* g = &gate_[(size_t)t * HC];
                float* o = &xn_[(size_t)t * H];
                for (int i = 0; i < H; i++) {
                    float acc = 0;
                    for (int s = 0; s < S; s++) acc += xs[s * H + i] * sigmoid(g[s * H + i]);
                    o[i] = acc * inv;
                }
            }
        });
    }

    // x[t, s] += 2 * sigmoid(inj[t, s]) * out[t]
    void hc_combine(int n) {
        const int H = c_.hidden, S = c_.hc;
        pool_.parallel_for((int64_t)n * S, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; i++) axpy(&x_[i * H], 2.0f * sigmoid(inj_[i]), &out_[(i / S) * H], H);
        }, 4);
    }

    // ---- PLE: hashed n-gram embeddings -----------------------------------------------------
    void ple(const Q4Layer& L, int n) {
        const int H = c_.hidden, HC = c_.hc_dim(), S = c_.hc;
        std::vector<float> emb((size_t)n * H), key((size_t)n * HC), value((size_t)n * H);
        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) q4_ple_embed(c_, w_, toks_.data(), pos_ + (int)t, &emb[(size_t)t * H]);
        });
        gq::run_jobs(pool_, {Job{&L.ple_key, emb.data(), key.data(), n}, Job{&L.ple_value, emb.data(), value.data(), n}});

        // gate each stream's copy of the value by sigmoid(signed sqrt(<key, query> / sqrt(H)))
        const int hist = ple_hist_len_, K = c_.ple_kernel, dil = c_.ple_ngram;
        std::vector<float> gated((size_t)n * HC), pad((size_t)(hist + n) * HC);
        std::memcpy(pad.data(), ple_hist_.data(), (size_t)hist * HC * sizeof(float));
        pool_.parallel_for((int64_t)n * S, [&](int64_t b, int64_t e) {
            std::vector<float> kn(H), qn(H);
            for (int64_t i = b; i < e; i++) {
                const int64_t t = i / S, s = i % S;
                rms_mul(&key[i * H], &L.ple_norm_key[s * H], kn.data(), H, c_.rms_eps);
                rms_mul(&x_[i * H], &L.ple_norm_query[s * H], qn.data(), H, c_.rms_eps);
                const float d = dot_f32(kn.data(), qn.data(), H) / std::sqrt((float)H);
                const float g = sigmoid((float)((d > 0) - (d < 0)) * std::sqrt(std::max(std::fabs(d), 1e-6f)));
                float* gv = &gated[i * H];
                for (int j = 0; j < H; j++) gv[j] = g * value[(size_t)t * H + j];
                rms_mul(gv, &L.ple_norm_conv[s * H], &pad[((size_t)hist + t) * HC + s * H], H, c_.rms_eps);
            }
        }, 4);
        // x += gated + silu(dilated causal depthwise conv of the normed gated value)
        pool_.parallel_for((int64_t)n * S, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; i++) {
                const int64_t t = i / S, s = i % S;
                float* x = &x_[i * H];
                const float* gv = &gated[i * H];
                const float* wk = &L.ple_conv1d[(size_t)s * H * K];
                for (int j = 0; j < H; j++) {
                    float acc = 0;
                    for (int k = 0; k < K; k++)
                        acc += wk[(size_t)j * K + k] * pad[((size_t)hist + t - (size_t)(K - 1 - k) * dil) * HC + s * H + j];
                    x[j] += gv[j] + silu(acc);
                }
            }
        }, 4);
        std::memcpy(ple_hist_.data(), &pad[(size_t)n * HC], (size_t)hist * HC * sizeof(float));
    }

    // ---- Gated DeltaNet --------------------------------------------------------------------
    void deltanet(int l, int n) {
        const Q4Layer& L = w_.layers[l];
        const int CD = c_.conv_dim(), VD = c_.lin_value_dim(), NV = c_.lin_v_heads, KD = c_.lin_k_dim,
                  DV = c_.lin_v_dim, K = c_.conv_kernel;
        std::vector<float>& qkv = a_;  // [n, CD] -> conv output in place
        std::vector<float>& z = b_;    // [n, VD]
        qkv.resize((size_t)n * CD);
        z.resize((size_t)n * VD);
        out_.resize((size_t)n * c_.hidden);
        std::vector<float> av((size_t)n * NV), bv((size_t)n * NV);
        gq::run_jobs(pool_, {Job{&L.qkv, xn_.data(), qkv.data(), n}, Job{&L.z, xn_.data(), z.data(), n},
                             Job{&L.alpha, xn_.data(), av.data(), n}, Job{&L.beta, xn_.data(), bv.data(), n}});

        // causal depthwise conv1d + silu, sequential over tokens per channel
        float* cs = &conv_state_[(size_t)slot_[l] * (K - 1) * CD];  // [K-1, CD], oldest first
        pool_.parallel_for(CD, [&](int64_t b, int64_t e) {
            for (int64_t ch = b; ch < e; ch++) {
                float wk[8], hist[8];
                for (int k = 0; k < K; k++) wk[k] = L.conv1d[ch * K + k];
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
        const int KEYD = c_.lin_key_dim();
        pool_.parallel_for(NV, [&](int64_t hb, int64_t he) {
            std::vector<float> q(KD), k(KD);
            for (int64_t h = hb; h < he; h++) {
                float* S = S0 + (size_t)h * KD * DV;  // [KD][DV]
                // the GGUF converter reorders the value heads from grouped to tiled: head h pairs with key head h % n_k
                const int kh = (int)h % c_.lin_k_heads;
                const float A = L.ssm_a[h];  // -exp(A_log), folded by the converter
                const float dtb = L.dt_bias[h];
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
                    // gated rmsnorm; unlike Qwen3.5 the output gate is a sigmoid
                    float ss = 0;
                    for (int j = 0; j < DV; j++) ss += o[j] * o[j];
                    const float r = 1.0f / std::sqrt(ss / DV + c_.rms_eps);
                    const float* zs = &z[(size_t)t * VD + h * DV];
                    for (int j = 0; j < DV; j++) o[j] = o[j] * r * L.ssm_norm[j] * sigmoid(zs[j]);
                }
            }
        });
        gq::run_jobs(pool_, {Job{&L.out, core.data(), out_.data(), n}});
    }

    // ---- Gated full attention with QSA selection ---------------------------------------------
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
        const Q4Layer& L = w_.layers[l];
        const int NH = c_.n_heads, NKV = c_.n_kv_heads, D = c_.head_dim, CR = L.compress;
        std::vector<float>& q = a_;  // [n, NH, 2*D]  (query | gate) per head
        std::vector<float>& k = b_;  // [n, NKV*D]
        std::vector<float>& v = c1_;
        q.resize((size_t)n * NH * 2 * D);
        k.resize((size_t)n * NKV * D);
        v.resize((size_t)n * NKV * D);
        out_.resize((size_t)n * c_.hidden);
        gq::run_jobs(pool_, {Job{&L.q, xn_.data(), q.data(), n}, Job{&L.k, xn_.data(), k.data(), n}, Job{&L.v, xn_.data(), v.data(), n}});

        const size_t kvs = (size_t)max_ctx_ * NKV * D;
        float* KC = &kcache_[slot_[l] * kvs];
        float* VC = &vcache_[slot_[l] * kvs];
        pool_.parallel_for(n, [&](int64_t tb, int64_t te) {
            for (int64_t t = tb; t < te; t++) {
                const int p = pos_ + (int)t;
                for (int h = 0; h < NH; h++) {
                    float* qh = &q[((size_t)t * NH + h) * 2 * D];
                    rms_mul(qh, L.q_norm.data(), qh, D, c_.rms_eps);
                    rope(qh, p);
                }
                for (int h = 0; h < NKV; h++) {
                    float* kh = &k[((size_t)t * NKV + h) * D];
                    rms_mul(kh, L.k_norm.data(), kh, D, c_.rms_eps);
                    rope(kh, p);
                    std::memcpy(&KC[((size_t)p * NKV + h) * D], kh, D * sizeof(float));
                    std::memcpy(&VC[((size_t)p * NKV + h) * D], &v[((size_t)t * NKV + h) * D], D * sizeof(float));
                }
            }
        }, 8);

        // per token: the cache rows it attends to, as [begin, end) runs (dense: one run)
        std::vector<std::vector<std::pair<int, int>>> runs(n);
        std::vector<std::vector<int>> blocks(n);
        if (CR) {
            qsa_.update(l, xn_.data(), n, pos_);
            if (qsa_.needs_select(l, pos_, n)) qsa_.select(l, xn_.data(), n, pos_, blocks);
        }
        for (int t = 0; t < n; t++) {
            const int len = pos_ + t + 1;
            auto& r = runs[t];
            if (blocks[t].empty()) { r.push_back({0, len}); continue; }
            for (int b : blocks[t]) {
                if (!r.empty() && r.back().second == b * CR) r.back().second = (b + 1) * CR;
                else r.push_back({b * CR, (b + 1) * CR});
            }
            const int tail = len / CR * CR;  // the incomplete tail block is always visible
            if (tail < len) {
                if (r.back().second == tail) r.back().second = len;
                else r.push_back({tail, len});
            }
        }

        std::vector<float>& att = c2_;  // [n, NH*D]
        att.resize((size_t)n * NH * D);
        const float scale = 1.0f / std::sqrt((float)D);
        const int group = NH / NKV;
        pool_.parallel_for((int64_t)n * NH, [&](int64_t b, int64_t e) {
            std::vector<float> sc;
            for (int64_t i = b; i < e; i++) {
                const int t = (int)(i / NH), h = (int)(i % NH), kvh = h / group;
                const float* qh = &q[((size_t)t * NH + h) * 2 * D];
                const float* gate = qh + D;
                sc.clear();
                float mx = -1e30f;
                for (auto [r0, r1] : runs[t])
                    for (int j = r0; j < r1; j++) {
                        sc.push_back(dot_f32(qh, &KC[((size_t)j * NKV + kvh) * D], D) * scale);
                        mx = std::max(mx, sc.back());
                    }
                float sum = 0;
                for (float& s : sc) { s = std::exp(s - mx); sum += s; }
                float* o = &att[((size_t)t * NH + h) * D];
                std::fill(o, o + D, 0.f);
                size_t m = 0;
                for (auto [r0, r1] : runs[t])
                    for (int j = r0; j < r1; j++) axpy(o, sc[m++] / sum, &VC[((size_t)j * NKV + kvh) * D], D);
                for (int d = 0; d < D; d++) o[d] *= sigmoid(gate[d]);
            }
        });
        gq::run_jobs(pool_, {Job{&L.o, att.data(), out_.data(), n}});
    }

    // ---- Mixture of experts ----------------------------------------------------------------
    void moe(const Q4Layer& L, int n) {
        const int H = c_.hidden, E = c_.n_experts, TK = c_.top_k, I = c_.moe_inter, SI = c_.shared_inter;
        out_.resize((size_t)n * H);
        std::vector<float> logits((size_t)n * E), sg((size_t)n * SI), su((size_t)n * SI);
        gq::run_jobs(pool_, {Job{&L.router, xn_.data(), logits.data(), n}, Job{&L.sh_gate, xn_.data(), sg.data(), n},
                             Job{&L.sh_up, xn_.data(), su.data(), n}});

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

        std::vector<GMat> mats(acts.size() * 3);  // gate, up, down views of every active expert
        std::vector<Job> jobs;
        for (size_t a = 0; a < acts.size(); a++) {
            mats[a * 3] = L.ex_gate.slice(acts[a].e);
            mats[a * 3 + 1] = L.ex_up.slice(acts[a].e);
            mats[a * 3 + 2] = L.ex_down.slice(acts[a].e);
            jobs.push_back({&mats[a * 3], &xin[acts[a].off * H], &g[acts[a].off * I], acts[a].n});
            jobs.push_back({&mats[a * 3 + 1], &xin[acts[a].off * H], &u[acts[a].off * I], acts[a].n});
        }
        gq::run_jobs(pool_, jobs);
        pool_.parallel_for((int64_t)g.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; i++) g[i] = silu(g[i]) * u[i];
        }, 4096);
        for (size_t i = 0; i < sg.size(); i++) sg[i] = silu(sg[i]) * su[i];

        std::vector<float> sy((size_t)n * H);
        jobs.clear();
        for (size_t a = 0; a < acts.size(); a++) jobs.push_back({&mats[a * 3 + 2], &g[acts[a].off * I], &y[acts[a].off * H], acts[a].n});
        jobs.push_back({&L.sh_down, sg.data(), sy.data(), n});
        gq::run_jobs(pool_, jobs);

        // per token: (row in y, weight) of its experts
        std::vector<std::vector<std::pair<size_t, float>>> per_tok(n);
        for (auto& a : acts)
            for (int i = 0; i < a.n; i++) per_tok[per_expert[a.e][i].first].push_back({a.off + i, per_expert[a.e][i].second});
        pool_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) {
                const float gs = sigmoid(dot_f32(L.shared_gate.data(), &xn_[(size_t)t * H], H));
                float* o = &out_[(size_t)t * H];
                for (int i = 0; i < H; i++) o[i] = gs * sy[(size_t)t * H + i];
                for (auto [r, wgt] : per_tok[t]) axpy(o, wgt, &y[r * H], H);
            }
        });
    }
};

}  // namespace

std::unique_ptr<Backend> make_qwen4_cpu_backend(const Q4Config& cfg, const Q4Weights& w, int max_ctx, int threads) {
    return std::make_unique<Qwen4Cpu>(cfg, w, max_ctx, threads);
}
