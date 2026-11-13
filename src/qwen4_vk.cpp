// Vulkan compute backend for the qwen4exp architecture (Qwen3.8-Flash-Next GGUF).
//
// Memory placement (tuned for a machine with far more RAM than VRAM):
//   * Dense weights (attention, DeltaNet, shared experts, hyper-connections, output head) live in VRAM.
//   * The routed experts stay in the memory-mapped GGUF. A VRAM cache of slots holds the most used ones;
//     every other expert is computed by the CPU straight from RAM, which has several times the bandwidth of
//     reading the weights over PCIe. Each layer syncs once after routing: the GPU runs the cached experts
//     while the CPU threads compute the rest, then the GPU combines both.
// What else differs from vk_backend.cpp:
//   * weights stay GGUF-quantized: experts keep their native Q4_K / Q5_K / Q5_1 blocks (decoded by the
//     shaders), Q8_0 is repacked planar (int8 weights + one f32 scale per 32), other types become f32
//   * expert slots are sized per layer (the quantization mix differs between layers); the VRAM cache is
//     split into one pool per slot size
//   * hyper-connections replace the layer norms; PLE injects hashed n-gram embeddings (gathered on the CPU)
//   * QSA: the indexer runs on the CPU from the block input the GPU copies out; beyond the token budget each
//     full-attention layer syncs so the CPU can hand the selection mask to the attention shader.
#include "expert_predictor.h"
#include <future>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cpu_kernels.h"
#include "qwen4.h"
#include "qwen4_qsa.h"
#include "vk_common.h"

namespace {

using namespace vkc;

constexpr int NMAX = 1024;          // max tokens per forward chunk
constexpr int NT = 8;               // tokens per matvec pass (matches q4_matmul)
constexpr int GEMM_MIN = 9;         // dense matrices use the tiled GEMM from this many tokens
constexpr int EXPERT_GEMM_MIN = 33; // grouped experts use the tiled GEMM from this many tokens
constexpr int ATT_PART_MAX = 2048;  // max (tokens x key splits) for split-K attention scratch
constexpr int MAX_FILLS = 64;        // cache fills per forward pass (each is a CPU repack + upload)
constexpr uint32_t FILL_MARGIN = 2; // session uses an expert needs over the cached one it replaces
// VRAM budget for cached experts when the driver cannot say how much VRAM is really free (see vram_query.cpp).
// Deliberately modest: the free-VRAM figure Vulkan reports does not account for what other programs hold, and
// an overcommitted card is far slower than a small cache. With a real figure the cache takes all free VRAM.
constexpr VkDeviceSize AUTO_CACHE = 6ull << 30;
constexpr int KV_START = 16384;      // tokens the K/V caches start with
constexpr VkDeviceSize VRAM_MARGIN = 2048ull << 20;  // left free for the driver, the display and other programs
constexpr float DRAFT_P_MIN = 0.5f; // stop drafting when the draft head's best token is less likely than this

// Push-constant layouts (must match the shaders, std430)
struct PcMat { uint64_t w1, s1, w2, s2, x, y; uint32_t rows, cols, n, x_stride, y_stride, post; float post_scale; };
struct PcExpert { uint64_t table, toks, x, y; uint32_t woff1, soff1, woff2, soff2, rows, cols, x_stride, y_stride, x_mode; };
struct PcHc { uint64_t a, b, c; uint32_t dim, hc; float eps; };
struct PcPleGate { uint64_t key, value, x, wk, wq, wc, gated, pad; uint32_t dim, hc, hist; float eps; };
struct PcPleConv { uint64_t x, gated, pad, w; uint32_t width, kernel, dil, hist; };
struct PcNorm { uint64_t x, delta, w, out; uint32_t dim; float eps; };
struct PcConv { uint64_t data, w, state; uint32_t n, channels; uint64_t snap; uint32_t snap_stride; };
struct PcDelta { uint64_t qkv, a, b, neg_a, dt_bias, state, out; uint32_t n, conv_dim, key_dim, n_v_heads, ratio; uint64_t snap; uint32_t snap_stride; };
struct PcGatedNorm { uint64_t o, z, w; uint32_t n_heads; float eps; };
struct PcQkRope { uint64_t qg, k, v, q_out, kcache, vcache, qn, kn, rope; uint32_t pos, n_heads, n_kv, rope_dim; float eps; };
struct PcAttn { uint64_t q, kcache, vcache, part; uint32_t pos, n_heads, n_kv, chunk, n_split; float scale; };
struct PcAttnQsa { uint64_t q, kcache, vcache, part; uint32_t pos, n_heads, n_kv, chunk, n_split; float scale; uint64_t mask; uint32_t cr, mask_words; };
struct PcAttnCombine { uint64_t part, qg, out; uint32_t n_heads, n_split; };
struct PcRoute { uint64_t logits, sel, selw; uint32_t n_experts, top_k; };
struct PcGroup { uint64_t sel, selw, loc, entries, toks, comb, args; uint32_t n_sel, top_k, n_experts, chunk, gx_up, gx_down; };
struct PcCombine { uint64_t x, sgate, sy, ey, table; uint32_t dim, top_k; };
struct PcCombW { uint64_t comb, selw; uint32_t n; };

struct ExpertEntry { uint64_t base; uint32_t start, count; };  // matches `Entry` in the expert shaders
static_assert(sizeof(ExpertEntry) == 16, "ExpertEntry layout");

// GPU-side dense matrix: planar Q8 (w = int8 weights, s = f32 scales) or plain f32 (w only).
struct GM { Ptr w, s; int rows = 0, cols = 0; bool q8 = false; };
struct GHc { Ptr norm; GM down, up, inject; bool has_inject = false; };

struct GLayer {
    GHc hc_attn, hc_ffn;
    GM q, k, v, o; Ptr qn, kn;                                   // attention
    GM qkv, z, out, alpha, beta; Ptr conv, neg_a, dt_bias, lnorm;  // deltanet
    GM ple_key, ple_value; Ptr ple_wk, ple_wq, ple_wc, ple_conv;   // PLE
    GM router, sgate, sh_gate, sh_up, sh_down;
    // expert slot layout (bytes from the slot base) and the pipelines that decode it
    uint32_t off_up = 0, off_down = 0, s_gate = 0, s_up = 0, s_down = 0;
    VkDeviceSize slot_size = 0;
    int cls = 0;
    std::string pipe_up, pipe_down, gemm_up, gemm_down;
};

uint32_t groups(uint32_t n, uint32_t per) { return (n + per - 1) / per; }
VkDeviceSize align256(VkDeviceSize v) { return (v + 255) & ~VkDeviceSize(255); }

// One Q8_0 row -> planar int8 weights + f32 scales.
void unpack_q8_row(const uint8_t* src, int cols, uint8_t* w, float* s) {
    for (int j = 0; j < cols; j += 32, src += 34) {
        uint16_t h;
        std::memcpy(&h, src, 2);
        s[j / 32] = ck::half_to_f32(h);
        std::memcpy(w + j, src + 2, 32);
    }
}

// Bytes of one expert matrix inside a slot: native blocks, or planar weights + scales for Q8_0.
VkDeviceSize expert_part_bytes(const gq::GMat& m) {
    if (m.type == GType::Q8_0) return align256((VkDeviceSize)m.rows * m.cols) + align256((VkDeviceSize)m.rows * (m.cols / 32) * 4);
    return align256(m.nbytes());
}

// ---------------------------------------------------------------------------------------------
class Qwen4Vulkan final : public Backend {
public:
    Qwen4Vulkan(const Q4Config& c, const Q4Weights& w, int max_ctx, const VulkanOptions& opt)
        : c_(c), w_(w), max_ctx_(max_ctx), opt_(opt), arena_(gpu_), up_(gpu_), cpu_(expert_threads(opt.threads)), qsa_(c, w, max_ctx, cpu_) {
        if (c.head_dim != 256 || c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.conv_kernel != 4 || c.hidden % 256 ||
            c.n_experts > 1024 || c.rope_dim > 256)
            throw std::runtime_error("model dimensions not supported by the Vulkan shaders");
        // tokens the MTP head drafts per pass; with them a pass is draft_ + 1 tokens, one matvec chunk at most
        draft_ = w.has_mtp ? std::max(0, std::min(opt.draft, NT - 1)) : 0;
        if (const char* tp = std::getenv("QW_TRACE")) trace_ = std::fopen(tp, "ab");
        sys_free0_ = qw_system_free_vram(gpu_.device_name());  // before our buffers: Windows only counts them once used
        upload_weights();
        alloc_state();
        setup_experts();
        reset();
        warm_experts();
    }

    // Touches the pages of the experts the CPU will compute, most used first: the first access to a page of the
    // mapped GGUF is a page fault (and a disk read if the file is not cached), which would otherwise land in
    // the middle of the first replies.
    void warm_experts() {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> keys;
        for (int k = 0; k < (int)n_keys(); k++)
            if (where_[k].first < 0) keys.push_back(k);
        std::stable_sort(keys.begin(), keys.end(), [&](int a, int b) { return usage_[a] > usage_[b]; });
        std::atomic<size_t> next{0}, bytes{0};
        std::atomic<unsigned> sink{0};
        cpu_.parallel_for(cpu_.size(), [&](int64_t, int64_t) {
            unsigned sum = 0;
            size_t done = 0;
            for (size_t i; (i = next.fetch_add(1)) < keys.size();) {
                const int l = keys[i] / c_.n_experts, e = keys[i] % c_.n_experts;
                for (const gq::GMat* st : {&w_.layers[l].ex_gate, &w_.layers[l].ex_up, &w_.layers[l].ex_down}) {
                    const gq::GMat m = st->slice(e);
                    for (size_t o = 0; o < m.nbytes(); o += 4096) sum += m.data[o];
                    done += m.nbytes();
                }
            }
            sink += sum;
            bytes += done;
        });
        std::fprintf(stderr, "[vulkan] touched %.1f GB of experts in RAM in %.1fs\n", bytes / 1073741824.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // The GPU round trip of every layer needs a free core for the driver, and the experts are bound by memory
    // bandwidth well before all hardware threads are busy (16 of 32 measured fastest on a 16-core CPU).
    static int expert_threads(int requested) {
        if (requested > 0) return requested;
        const int hw = (int)std::thread::hardware_concurrency();
        return std::max(1, hw >= 16 ? hw / 2 : hw - 2);
    }

    ~Qwen4Vulkan() override {
        std::fprintf(stderr, "[vulkan] expert lookups: %llu on the GPU (cached), %llu on the CPU; %llu cache fills\n",
                     (unsigned long long)hits_, (unsigned long long)misses_, (unsigned long long)fills_);
        if (std::getenv("QW_PROFILE"))
        {
            std::fprintf(stderr, "[vulkan] moe time: %.2fs waiting for the GPU, %.2fs CPU experts, %.2fs cache fills\n", t_gpu_, t_cpu_, t_fill_);
            // the hit rate a cache of this size would have had with this session's hottest experts in it
            uint64_t best = 0, tot = 0, distinct = 0;
            for (size_t k = 0; k < slots_.size(); k++) {
                std::vector<uint32_t> v;
                for (int key = 0; key < n_keys(); key++)
                    if (L_[key / c_.n_experts].cls == (int)k && recent_[key]) v.push_back(recent_[key]);
                std::sort(v.begin(), v.end(), std::greater<uint32_t>());
                distinct += v.size();
                for (size_t i = 0; i < v.size(); i++) { tot += v[i]; if (i < slots_[k].size()) best += v[i]; }
            }
            std::fprintf(stderr, "[vulkan] %llu distinct experts used; a cache holding the hottest would serve %.0f%% of lookups\n",
                         (unsigned long long)distinct, tot ? 100.0 * best / tot : 0.0);
        }
        save_stats();
        if (trace_) std::fclose(trace_);
        if (pred_ready_) {
            if (pred_job_.valid()) pred_job_.get();
            pred_.save();
            std::fprintf(stderr, "[vulkan] expert predictor: %llu training steps, loss %.3f, %.2fs\n",
                         (unsigned long long)pred_.trained(), pred_.loss(), t_pred_);
        }
    }

    void reset() override {
        pos_ = 0;
        gpu_.begin();
        gpu_.fill_zero(conv_state_);
        gpu_.fill_zero(rec_state_);
        gpu_.fill_zero(ple_state_);
        if (draft_) gpu_.fill_zero(hlast_);
        gpu_.submit_wait();
    }
    int pos() const override { return pos_; }
    int max_ctx() const override { return max_ctx_; }
    const char* name() const override { return "vulkan"; }

    void save_state() override {
        gpu_.begin();
        gpu_.copy(conv_state_, 0, conv_saved_, 0, conv_state_->size);
        gpu_.copy(rec_state_, 0, rec_saved_, 0, rec_state_->size);
        gpu_.copy(ple_state_, 0, ple_saved_, 0, ple_state_->size);
        if (draft_) gpu_.copy(hlast_, 0, hlast_saved_, 0, hlast_->size);
        gpu_.submit_wait();
        saved_pos_ = pos_;
    }
    void restore_state() override {
        gpu_.begin();
        gpu_.copy(conv_saved_, 0, conv_state_, 0, conv_state_->size);
        gpu_.copy(rec_saved_, 0, rec_state_, 0, rec_state_->size);
        gpu_.copy(ple_saved_, 0, ple_state_, 0, ple_state_->size);
        if (draft_) gpu_.copy(hlast_saved_, 0, hlast_, 0, hlast_->size);
        gpu_.submit_wait();
        pos_ = saved_pos_;
    }

    void forward(const int* tokens, int n, float* logits) override {
        check_tokens(tokens, n);
        for (int done = 0; done < n;) {
            const int m = std::min(NMAX, n - done), p0 = pos_;
            forward_chunk(tokens + done, m, done + m == n ? logits : nullptr, false);
            if (draft_) mtp_run(tokens + done, m, p0, nullptr, true, true);  // keep the draft block's KV cache in step
            done += m;
        }
    }

    // ---- speculative decoding with the MTP draft block ----------------------------------------------------
    // The block sees, at position p + 1, the trunk's residual streams of position p and the token at p + 1. Its
    // KV cache is filled from the real trunk states (after every pass); a draft chains its own output instead.
    int max_draft() const override { return draft_; }

    int draft(int token, int n, int* out) override {
        n = std::min({n, draft_, max_ctx_ - pos_ - 2});
        int cnt = 0;
        mlogits_.resize(c_.vocab);
        for (int k = 0; k < n; k++) {
            mtp_run(&token, 1, pos_ + k, mlogits_.data(), k == 0, false);
            // probability of the best token under the draft head
            const int best = (int)(std::max_element(mlogits_.begin(), mlogits_.end()) - mlogits_.begin());
            const float mx = mlogits_[best];
            double sum = 0;
            for (float v : mlogits_) sum += std::exp((double)(v - mx));
            if (1.0 / sum < DRAFT_P_MIN) break;
            out[cnt++] = token = best;
        }
        return cnt;
    }

    void forward_all(const int* tokens, int n, float* logits) override {
        if (n < 1 || n > draft_ + 1) throw std::runtime_error("forward_all: bad token count");
        check_tokens(tokens, n);
        spec_pos0_ = pos_;
        spec_toks_.assign(tokens, tokens + n);
        forward_chunk(tokens, n, logits, true);
    }

    void accept(int keep) override {
        const int n = (int)spec_toks_.size();
        if (keep < 1 || keep > n) throw std::runtime_error("accept: bad count");
        if (keep < n) {  // roll the recurrent state back to the snapshot taken after token keep - 1
            gpu_.begin();
            gpu_.copy(conv_snap_, (VkDeviceSize)(keep - 1) * conv_state_->size, conv_state_, 0, conv_state_->size);
            gpu_.copy(rec_snap_, (VkDeviceSize)(keep - 1) * rec_state_->size, rec_state_, 0, rec_state_->size);
            if (ple_hist_)  // the normed rows of this pass are still in the conv input buffer
                gpu_.copy(ppad_.b, ppad_.off + (VkDeviceSize)keep * c_.hc_dim() * 4, ple_state_, 0, (VkDeviceSize)ple_hist_ * c_.hc_dim() * 4);
            gpu_.submit_wait();
            pos_ = spec_pos0_ + keep;
        }
        mtp_run(spec_toks_.data(), keep, spec_pos0_, nullptr, true, true);
        spec_toks_.clear();
    }

private:
    void check_tokens(const int* tokens, int n) const {
        if (pos_ + n > max_ctx_) throw std::runtime_error("context overflow");
        for (int t = 0; t < n; t++)
            if (tokens[t] < 0 || tokens[t] >= c_.vocab) throw std::runtime_error("token id out of range");
    }

    const Q4Config& c_;
    const Q4Weights& w_;
    int max_ctx_;
    VkDeviceSize sys_free0_ = 0, kv_bytes_ = 0;
    int kv_cap_ = 0, n_kv_slots_ = 0;  // tokens the K/V caches hold now, full-attention layers in them
    VulkanOptions opt_;
    int pos_ = 0, saved_pos_ = 0;
    Gpu gpu_;
    Arena arena_;
    Uploader up_;
    ThreadPool cpu_;
    QsaIndexer qsa_;

    std::vector<GLayer> L_;
    GM output_;
    GHc hc_head_;
    // MTP draft block (layer index c_.n_layers of L_ / w_.layers)
    int draft_ = 0, spec_pos0_ = 0;
    std::vector<int> spec_toks_;
    std::vector<float> mlogits_;
    GM mtp_eh_;
    GHc mtp_head_;
    Ptr mtp_enorm_, mtp_hnorm_, mx_, mh_, mcat_, men_, memb_;
    Buf* hme_ = nullptr;          // token embeddings of the draft block's rows (CPU -> GPU)
    Buf* hlast_ = nullptr;        // trunk residual of the last position, waiting for the token that follows it
    Buf* hlast_saved_ = nullptr;
    Buf* conv_snap_ = nullptr;    // recurrent state after each token of a speculative pass, [draft_][state]
    Buf* rec_snap_ = nullptr;
    Ptr rope_;
    std::vector<int> slot_;      // per layer: index into the full-attn or linear-attn state arrays
    std::vector<int> qsa_slot_;  // per layer: index into hqsa_ or -1
    std::vector<int> toks_;      // token at every position (PLE n-gram context)
    int ple_hist_ = 0, mask_words_ = 1;

    // activations (device local)
    Ptr x_, xnorm_, gate_, lo_, inj_, xn_, mix_, qkv_, z_, a_, b_, core_, qg_, kb_, vb_, qo_, att_, part_, sact_, sy_, sgate_, act_, ey_;
    Ptr rlogits_, selw_, comb_, sel_, logits_, emb_, pkey_, pval_, pgated_, ppad_;
    Buf* kcache_ = nullptr;
    Buf* vcache_ = nullptr;
    Buf* conv_state_ = nullptr;
    Buf* rec_state_ = nullptr;
    Buf* ple_state_ = nullptr;
    Buf* conv_saved_ = nullptr;
    Buf* rec_saved_ = nullptr;
    Buf* ple_saved_ = nullptr;
    // host-visible
    Buf* hx_ = nullptr;     // residual streams of the current chunk (hc copies of the token embedding)
    Buf* hemb_ = nullptr;   // PLE n-gram embeddings of the current chunk
    Buf* hsel_ = nullptr;   // selected expert ids of the current layer, [NMAX * top_k]
    Buf* hlog_ = nullptr;   // final logits
    Buf* hqsa_ = nullptr;   // block inputs of the QSA layers, [qsa slot][NMAX][hidden] (GPU -> CPU)
    Buf* hmask_ = nullptr;  // QSA selection bitmask, [NMAX][mask_words] (CPU -> GPU)

    // experts
    struct Slot { Buf* buf = nullptr; VkDeviceSize off = 0; int key = -1; uint64_t tick = 0; };
    std::vector<std::vector<Slot>> slots_;    // VRAM cache, one pool per slot-size class
    std::vector<VkDeviceSize> cls_size_;
    std::vector<std::pair<int, int>> where_;  // key (layer * n_experts + expert) -> (class, slot) or (-1, -1)
    std::vector<int> pending_;                // keys to bring into the cache at the start of the next pass
    std::vector<std::vector<int>> per_;       // scratch: selections per expert
    std::vector<int> cpu_experts_;
    std::vector<float> cg_, cu_, cx_;
    std::vector<gq::GMat> cmats_;
    std::vector<uint64_t> usage_;             // selection counts per key (persisted)
    std::vector<uint32_t> recent_;            // the same for this session only
    std::vector<float> heat_;                 // QW_EMA=passes: use counts decayed with that time constant
    float ema_ = 0, ema_margin_ = 2.5f;
    // QW_PRED=1: the cache follows a model trained while the engine runs (expert_predictor.h)
    ExpertPredictor pred_;
    bool use_pred_ = false, pred_ready_ = false;
    float pred_margin_ = 1.f;
    double t_pred_ = 0;
    std::future<void> pred_job_;
    std::vector<std::pair<int, uint32_t>> pass_used_;
    float heat(int key) const { return use_pred_ ? pred_.score(key) : ema_ > 0 ? heat_[key] : (float)recent_[key]; }
    float fill_margin() const { return use_pred_ ? pred_margin_ : ema_ > 0 ? ema_margin_ : (float)FILL_MARGIN; }
    FILE* trace_ = nullptr;                   // QW_TRACE=path: tokens and expert selections of every pass
    Buf* hxn_ = nullptr;                      // block input of the layer (GPU -> CPU)
    Buf* htab_ = nullptr;                     // entries | toks | comb tables written by the CPU
    Buf* hcy_ = nullptr;                      // expert outputs computed on the CPU (CPU -> GPU)
    VkDeviceSize htab_toks_ = 0, htab_comb_ = 0;
    double t_gpu_ = 0, t_cpu_ = 0, t_fill_ = 0;  // seconds per stage of the MoE sync (QW_PROFILE)
    uint64_t tick_ = 0, hits_ = 0, misses_ = 0, fills_ = 0;

    // ---- setup ----------------------------------------------------------------------------
    // Q8_0 goes up planar; anything else is dequantized to f32 (only small / float tensors in practice).
    GM up_dense(const gq::GMat& m, bool force_f32 = false) {
        GM g;
        g.rows = m.rows;
        g.cols = m.cols;
        if (m.cols % 32) throw std::runtime_error("matrix width not a multiple of 32");
        g.q8 = m.type == GType::Q8_0 && !force_f32;
        const size_t nb = (size_t)m.cols / 32, row_bytes = g.q8 ? (size_t)m.cols : (size_t)m.cols * 4;
        g.w = arena_.alloc((VkDeviceSize)m.rows * row_bytes);
        if (g.q8) g.s = arena_.alloc((VkDeviceSize)m.rows * nb * 4);
        const int batch = std::max<int>(1, (int)(((size_t)48 << 20) / row_bytes));
        std::vector<uint8_t> wbuf;
        std::vector<float> sbuf;
        for (int r0 = 0; r0 < m.rows; r0 += batch) {
            const int nr = std::min(batch, m.rows - r0);
            wbuf.resize((size_t)nr * row_bytes);
            sbuf.resize((size_t)nr * nb);
            cpu_.parallel_for(nr, [&](int64_t b, int64_t e) {
                for (int64_t r = b; r < e; r++) {
                    if (g.q8) unpack_q8_row(m.row(r0 + r), m.cols, &wbuf[(size_t)r * row_bytes], &sbuf[(size_t)r * nb]);
                    else gq::dequant_row(m.type, m.row(r0 + r), m.cols, (float*)&wbuf[(size_t)r * row_bytes]);
                }
            }, 16);
            up_.put({g.w.b, g.w.off + (VkDeviceSize)r0 * row_bytes}, wbuf.data(), wbuf.size());
            if (g.q8) up_.put({g.s.b, g.s.off + (VkDeviceSize)r0 * nb * 4}, sbuf.data(), sbuf.size() * 4);
        }
        return g;
    }
    Ptr up_f32(const std::vector<float>& v) {
        Ptr p = arena_.alloc(v.size() * 4);
        up_.put(p, v.data(), v.size() * 4);
        return p;
    }
    // A float vector as a 1-row f32 matrix.
    GM up_row(const std::vector<float>& v) {
        GM g;
        g.rows = 1;
        g.cols = (int)v.size();
        g.w = up_f32(v);
        return g;
    }
    GHc up_hc(const Q4Hc& h) {
        GHc g;
        g.norm = up_f32(h.norm);
        g.down = up_dense(h.down);
        g.up = up_dense(h.up);
        g.has_inject = h.inject.data != nullptr;
        if (g.has_inject) g.inject = up_dense(h.inject);
        return g;
    }
    // qk_norm_rope multiplies by (1 + w); the GGUF gammas already contain the 1.
    Ptr up_norm_minus1(std::vector<float> v) {
        for (float& x : v) x -= 1.0f;
        return up_f32(v);
    }

    void upload_weights() {
        auto t0 = std::chrono::steady_clock::now();
        output_ = up_dense(w_.output);
        hc_head_ = up_hc(w_.hc_head);
        int nf = 0, nl = 0, nq = 0;
        const int n_blocks = c_.n_layers + (draft_ ? 1 : 0);
        if (draft_) {
            mtp_eh_ = up_dense(w_.mtp_eh);
            mtp_enorm_ = up_norm_minus1(w_.mtp_enorm);  // add_rmsnorm multiplies by (1 + w)
            mtp_hnorm_ = up_f32(w_.mtp_hnorm);
            mtp_head_ = up_hc(w_.mtp_head);
        }
        for (int l = 0; l < n_blocks; l++) {
            const Q4Layer& W = w_.layers[l];
            GLayer g;
            g.hc_attn = up_hc(W.hc_attn);
            g.hc_ffn = up_hc(W.hc_ffn);
            if (W.full_attn) {
                slot_.push_back(nf++);
                g.q = up_dense(W.q); g.k = up_dense(W.k); g.v = up_dense(W.v); g.o = up_dense(W.o);
                g.qn = up_norm_minus1(W.q_norm);
                g.kn = up_norm_minus1(W.k_norm);
            } else {
                slot_.push_back(nl++);
                g.qkv = up_dense(W.qkv); g.z = up_dense(W.z); g.out = up_dense(W.out);
                g.alpha = up_dense(W.alpha); g.beta = up_dense(W.beta);
                g.conv = up_f32(W.conv1d);
                g.neg_a = up_f32(W.ssm_a);
                g.dt_bias = up_f32(W.dt_bias);
                g.lnorm = up_f32(W.ssm_norm);
            }
            qsa_slot_.push_back(W.full_attn && W.compress > 0 ? nq++ : -1);
            if (W.ple) {
                g.ple_key = up_dense(W.ple_key); g.ple_value = up_dense(W.ple_value);
                g.ple_wk = up_f32(W.ple_norm_key); g.ple_wq = up_f32(W.ple_norm_query); g.ple_wc = up_f32(W.ple_norm_conv);
                g.ple_conv = up_f32(W.ple_conv1d);
            }
            g.router = up_dense(W.router);
            g.sgate = up_row(W.shared_gate);
            // the swiglu pipeline decodes gate and up with one codec
            const bool sq8 = W.sh_gate.type == GType::Q8_0 && W.sh_up.type == GType::Q8_0;
            g.sh_gate = up_dense(W.sh_gate, !sq8); g.sh_up = up_dense(W.sh_up, !sq8); g.sh_down = up_dense(W.sh_down);
            L_.push_back(g);
        }
        // rope table [max_ctx][rope_dim/2] of (cos, sin), computed like the CPU path
        const int half = c_.rope_dim / 2;
        std::vector<float> rope((size_t)max_ctx_ * c_.rope_dim);
        for (int p = 0; p < max_ctx_; p++)
            for (int i = 0; i < half; i++) {
                float inv = (float)(1.0 / std::pow((double)c_.rope_theta, (2.0 * i) / c_.rope_dim));
                float ang = (float)p * inv;
                rope[(size_t)p * c_.rope_dim + 2 * i] = std::cos(ang);
                rope[(size_t)p * c_.rope_dim + 2 * i + 1] = std::sin(ang);
            }
        rope_ = up_f32(rope);
        up_.flush();
        std::fprintf(stderr, "[vulkan] dense weights uploaded in %.1fs\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    void alloc_state() {
        const int H = c_.hidden, HC = c_.hc_dim(), TK = c_.top_k;
        const int n_lin = (int)std::count(c_.is_full_attn.begin(), c_.is_full_attn.end(), false);
        const int n_full = (int)L_.size() - n_lin;  // includes the draft block
        const int n_qsa = (int)std::count_if(qsa_slot_.begin(), qsa_slot_.end(), [](int s) { return s >= 0; });
        ple_hist_ = c_.ple_ngram > 0 ? (c_.ple_kernel - 1) * c_.ple_ngram : 0;
        int max_blocks = 1;
        for (size_t l = 0; l < L_.size(); l++)
            if (w_.layers[l].compress > 0) max_blocks = std::max(max_blocks, max_ctx_ / w_.layers[l].compress + 1);
        mask_words_ = (max_blocks + 31) / 32;
        toks_.assign(max_ctx_, 0);

        auto act = [&](size_t floats) { return arena_.alloc(floats * 4); };
        x_ = act((size_t)NMAX * HC);
        xnorm_ = act((size_t)NMAX * HC);
        gate_ = act((size_t)NMAX * HC);
        lo_ = act((size_t)NMAX * c_.hc_rank);
        inj_ = act((size_t)NMAX * c_.hc);
        xn_ = act((size_t)NMAX * H);
        mix_ = act((size_t)NMAX * H);
        qkv_ = act((size_t)NMAX * c_.conv_dim());
        z_ = act((size_t)NMAX * c_.lin_value_dim());
        a_ = act((size_t)NMAX * c_.lin_v_heads);
        b_ = act((size_t)NMAX * c_.lin_v_heads);
        core_ = act((size_t)NMAX * c_.lin_value_dim());
        qg_ = act((size_t)NMAX * c_.n_heads * c_.head_dim * 2);
        kb_ = act((size_t)NMAX * c_.n_kv_heads * c_.head_dim);
        vb_ = act((size_t)NMAX * c_.n_kv_heads * c_.head_dim);
        qo_ = act((size_t)NMAX * c_.n_heads * c_.head_dim);
        att_ = act((size_t)NMAX * c_.n_heads * c_.head_dim);
        part_ = act((size_t)ATT_PART_MAX * c_.n_heads * (c_.head_dim + 2));
        sact_ = act((size_t)NMAX * c_.shared_inter);
        sy_ = act((size_t)NMAX * H);
        sgate_ = act(NMAX);
        act_ = act((size_t)NMAX * TK * c_.moe_inter);
        ey_ = act((size_t)NMAX * TK * H);
        rlogits_ = act((size_t)NMAX * c_.n_experts);
        selw_ = act((size_t)NMAX * TK);
        comb_ = act((size_t)NMAX * TK * 2);
        sel_ = arena_.alloc((VkDeviceSize)NMAX * TK * 4);
        logits_ = arena_.alloc((VkDeviceSize)c_.vocab * 4 * (draft_ + 1));
        if (draft_) {
            mx_ = act((size_t)NMAX * HC);
            mh_ = act((size_t)NMAX * HC);
            mcat_ = act((size_t)NMAX * HC * 2);
            men_ = act((size_t)NMAX * H);
            memb_ = act((size_t)NMAX * H);
        }
        if (ple_hist_) {
            emb_ = act((size_t)NMAX * H);
            pkey_ = act((size_t)NMAX * HC);
            pval_ = act((size_t)NMAX * H);
            pgated_ = act((size_t)NMAX * HC);
            ppad_ = act((size_t)(NMAX + ple_hist_) * HC);
        }
        // the K/V caches hold half floats (the kv16 shader variants), which is what makes a 256K context fit
        // They cover kv_cap_ tokens and grow with the conversation (ensure_kv), so the context limit costs no
        // VRAM until it is used.
        n_kv_slots_ = n_full;
        const char* ks = std::getenv("QW_KV_START");  // tests force growth with a small start
        kv_cap_ = std::min(max_ctx_, ks ? std::max(1, std::atoi(ks)) : KV_START);
        const VkDeviceSize kv = (VkDeviceSize)n_full * kv_cap_ * c_.n_kv_heads * c_.head_dim * 2;
        kv_bytes_ = 2 * kv;
        kcache_ = gpu_.create(std::max<VkDeviceSize>(kv, 256), Mem::Device);
        vcache_ = gpu_.create(std::max<VkDeviceSize>(kv, 256), Mem::Device);
        const VkDeviceSize cs = std::max<VkDeviceSize>((VkDeviceSize)n_lin * (c_.conv_kernel - 1) * c_.conv_dim() * 4, 256);
        const VkDeviceSize rs = std::max<VkDeviceSize>((VkDeviceSize)n_lin * c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4, 256);
        const VkDeviceSize ps = std::max<VkDeviceSize>((VkDeviceSize)ple_hist_ * HC * 4, 256);
        conv_state_ = gpu_.create(cs, Mem::Device);
        rec_state_ = gpu_.create(rs, Mem::Device);
        ple_state_ = gpu_.create(ps, Mem::Device);
        conv_saved_ = gpu_.create(cs, Mem::Device);
        rec_saved_ = gpu_.create(rs, Mem::Device);
        ple_saved_ = gpu_.create(ps, Mem::Device);
        if (draft_) {
            hme_ = gpu_.create((VkDeviceSize)NMAX * H * 4, Mem::HostUpload);
            hlast_ = gpu_.create((VkDeviceSize)HC * 4, Mem::Device);
            hlast_saved_ = gpu_.create((VkDeviceSize)HC * 4, Mem::Device);
            conv_snap_ = gpu_.create(cs * draft_, Mem::Device);
            rec_snap_ = gpu_.create(rs * draft_, Mem::Device);
        }
        hx_ = gpu_.create((VkDeviceSize)NMAX * HC * 4, Mem::HostUpload);
        hemb_ = gpu_.create((VkDeviceSize)NMAX * H * 4, Mem::HostUpload);
        hsel_ = gpu_.create((VkDeviceSize)NMAX * TK * 4, Mem::HostRead);
        hxn_ = gpu_.create((VkDeviceSize)NMAX * H * 4, Mem::HostRead);
        hcy_ = gpu_.create((VkDeviceSize)NMAX * TK * H * 4, Mem::HostUpload);
        const VkDeviceSize ne = (VkDeviceSize)NMAX * TK;
        htab_toks_ = align256(ne * sizeof(ExpertEntry));
        htab_comb_ = align256(htab_toks_ + ne * 4);
        htab_ = gpu_.create(htab_comb_ + ne * 8, Mem::HostUpload);
        hlog_ = gpu_.create((VkDeviceSize)c_.vocab * 4 * (draft_ + 1), Mem::HostRead);
        hqsa_ = gpu_.create(std::max<VkDeviceSize>((VkDeviceSize)n_qsa * NMAX * H * 4, 256), Mem::HostRead);
        hmask_ = gpu_.create((VkDeviceSize)NMAX * mask_words_ * 4, Mem::HostUpload);
    }

    // ---- experts ------------------------------------------------------------------------------
    size_t n_keys() const { return L_.size() * c_.n_experts; }
    uint64_t slot_addr(std::pair<int, int> cs) const { return slots_[cs.first][cs.second].buf->addr + slots_[cs.first][cs.second].off; }

    static void write_part(const gq::GMat& m, uint8_t* d, uint32_t s_off_rel) {
        if (m.type != GType::Q8_0) { std::memcpy(d, m.data, m.nbytes()); return; }
        float* s = (float*)(d + s_off_rel);
        for (int r = 0; r < m.rows; r++) unpack_q8_row(m.row(r), m.cols, d + (size_t)r * m.cols, s + (size_t)r * (m.cols / 32));
    }
    void write_expert(int key, uint8_t* d) const {
        const int l = key / c_.n_experts, e = key % c_.n_experts;
        const Q4Layer& W = w_.layers[l];
        const GLayer& g = L_[l];
        write_part(W.ex_gate.slice(e), d, g.s_gate);
        write_part(W.ex_up.slice(e), d + g.off_up, g.s_up - g.off_up);
        write_part(W.ex_down.slice(e), d + g.off_down, g.s_down - g.off_down);
    }

    static const char* codec(GType t) {
        switch (t) {
            case GType::Q4_K: return "q4k";
            case GType::Q5_K: return "q5k";
            case GType::Q8_0: return "q8";
            case GType::Q5_1: return "q51";
            default: return nullptr;
        }
    }

    // VRAM we may still take: what was free on the card at start (over all processes, when the driver can tell)
    // less our own buffers and a margin. Windows only counts a buffer once it is used, hence the bookkeeping.
    VkDeviceSize vram_left() const {
        const VkDeviceSize own = gpu_.allocated_device();
        VkDeviceSize free = gpu_.device_free();
        if (sys_free0_) free = std::min<VkDeviceSize>(free, sys_free0_ > own ? sys_free0_ - own : 0);
        return free > VRAM_MARGIN ? free - VRAM_MARGIN : 0;
    }

    // Gives one buffer of expert slots back (of the class with the most slots). False when the cache is empty.
    bool release_expert_chunk() {
        size_t k = 0;
        for (size_t i = 1; i < slots_.size(); i++)
            if (slots_[i].size() > slots_[k].size()) k = i;
        if (slots_.empty() || slots_[k].empty()) return false;
        Buf* b = slots_[k].back().buf;
        while (!slots_[k].empty() && slots_[k].back().buf == b) {
            if (slots_[k].back().key >= 0) where_[slots_[k].back().key] = {-1, -1};
            slots_[k].pop_back();
        }
        gpu_.destroy(b);
        return true;
    }

    // Grows the K/V caches to hold `tokens` positions (doubling, up to the context limit). The VRAM comes out
    // of the expert cache when the card has none left. Call between passes: nothing may be recorded or running.
    void ensure_kv(int tokens) {
        if (tokens <= kv_cap_) return;
        const int cap = (int)std::min<int64_t>(max_ctx_, std::max<int64_t>(2ll * kv_cap_, tokens));
        const VkDeviceSize per = (VkDeviceSize)c_.n_kv_heads * c_.head_dim * 2;
        const VkDeviceSize size = (VkDeviceSize)n_kv_slots_ * cap * per;
        Buf* nb[2] = {nullptr, nullptr};
        for (Buf*& b : nb) {
            while (vram_left() < size && release_expert_chunk()) {}
            while (!(b = gpu_.create(size, Mem::Device, true)))
                if (!release_expert_chunk()) throw std::runtime_error("out of VRAM growing the K/V caches");
        }
        gpu_.begin();
        for (int s = 0; s < n_kv_slots_; s++) {
            gpu_.copy(kcache_, (VkDeviceSize)s * kv_cap_ * per, nb[0], (VkDeviceSize)s * cap * per, (VkDeviceSize)kv_cap_ * per);
            gpu_.copy(vcache_, (VkDeviceSize)s * kv_cap_ * per, nb[1], (VkDeviceSize)s * cap * per, (VkDeviceSize)kv_cap_ * per);
        }
        gpu_.submit_wait();
        gpu_.destroy(kcache_);
        gpu_.destroy(vcache_);
        kcache_ = nb[0];
        vcache_ = nb[1];
        kv_cap_ = cap;
        size_t n_slots = 0;
        for (auto& s : slots_) n_slots += s.size();
        std::fprintf(stderr, "[vulkan] K/V caches grown to %d tokens (%.1f GB); %zu expert slots cached\n", cap,
                     2 * size / 1073741824.0, n_slots);
    }

    void setup_experts() {
        // per-layer slot layout: [gate][up][down], Q8_0 parts as planar weights followed by their scales
        for (int l = 0; l < (int)L_.size(); l++) {
            const Q4Layer& W = w_.layers[l];
            GLayer& g = L_[l];
            const char* cu = codec(W.ex_gate.type);
            const char* cd = codec(W.ex_down.type);
            if (!cu || !cd || W.ex_gate.type != W.ex_up.type || W.ex_gate.type == GType::Q5_1)
                throw std::runtime_error(std::string("layer ") + std::to_string(l) + ": unsupported expert quantization " +
                                         gtype_name(W.ex_gate.type) + "/" + gtype_name(W.ex_up.type) + "/" + gtype_name(W.ex_down.type));
            g.pipe_up = std::string("q4_expert_") + cu + "_swiglu";
            g.pipe_down = std::string("q4_expert_") + cd;
            g.gemm_up = std::string("q4_gemm_expert_") + cu + "_swiglu";
            g.gemm_down = std::string("q4_gemm_expert_") + cd;
            auto planar_w = [](const gq::GMat& m) { return (uint32_t)align256((VkDeviceSize)m.rows * m.cols); };
            const gq::GMat ga = W.ex_gate.slice(0), ua = W.ex_up.slice(0), da = W.ex_down.slice(0);
            VkDeviceSize o = 0;
            g.s_gate = (uint32_t)o + (ga.type == GType::Q8_0 ? planar_w(ga) : 0);
            o += expert_part_bytes(ga);
            g.off_up = (uint32_t)o;
            g.s_up = (uint32_t)o + (ua.type == GType::Q8_0 ? planar_w(ua) : 0);
            o += expert_part_bytes(ua);
            g.off_down = (uint32_t)o;
            g.s_down = (uint32_t)o + (da.type == GType::Q8_0 ? planar_w(da) : 0);
            o += expert_part_bytes(da);
            g.slot_size = o;
            auto it = std::find(cls_size_.begin(), cls_size_.end(), o);
            g.cls = (int)(it - cls_size_.begin());
            if (it == cls_size_.end()) cls_size_.push_back(o);
        }
        usage_.assign(n_keys(), 0);
        recent_.assign(n_keys(), 0);
        if (const char* ev = std::getenv("QW_EMA")) ema_ = (float)std::atof(ev);
        if (const char* ev = std::getenv("QW_EMA_MARGIN")) ema_margin_ = (float)std::atof(ev);
        heat_.assign(n_keys(), 0.f);
        use_pred_ = std::getenv("QW_PRED") != nullptr;
        if (const char* ev = std::getenv("QW_PRED_MARGIN")) pred_margin_ = (float)std::atof(ev);
        load_stats();

        // VRAM cache: the budget is split between the slot-size classes in proportion to their experts' bytes.
        // It never takes VRAM that is not free right now: an overcommitted card makes Windows move buffers
        // to system memory, which slows every shader down.
        const VkDeviceSize own = gpu_.allocated_device(), left = vram_left();
        VkDeviceSize budget = opt_.expert_cache_set ? (VkDeviceSize)opt_.expert_cache_mb << 20 : sys_free0_ ? ~0ull : AUTO_CACHE;
        budget = std::min(budget, left);
        if (sys_free0_)
            std::fprintf(stderr, "[vulkan] VRAM: %.1f GB free at start, %.1f GB dense weights and state, %.1f GB K/V caches "
                                 "for the first %d tokens of context, %.1f GB left for experts\n",
                         sys_free0_ / 1073741824.0, (own - kv_bytes_) / 1073741824.0, kv_bytes_ / 1073741824.0, kv_cap_,
                         left / 1073741824.0);
        const size_t nc = cls_size_.size();
        std::vector<size_t> cls_keys(nc, 0);
        VkDeviceSize total = 0;
        for (int l = 0; l < (int)L_.size(); l++) {
            cls_keys[L_[l].cls] += c_.n_experts;
            total += L_[l].slot_size * c_.n_experts;
        }
        slots_.resize(nc);
        size_t n_slots = 0;
        VkDeviceSize slot_bytes = 0;
        for (size_t k = 0; k < nc; k++) {
            const double share = (double)(cls_size_[k] * cls_keys[k]) / (double)total;
            const size_t want = std::min<size_t>(cls_keys[k], (size_t)(budget * share / cls_size_[k]));
            const size_t per_chunk = CHUNK / cls_size_[k];
            while (slots_[k].size() < want) {
                const size_t cnt = std::min(per_chunk, want - slots_[k].size());
                Buf* b = gpu_.create(cnt * cls_size_[k], Mem::Device, true);
                if (!b) break;
                for (size_t i = 0; i < cnt; i++) slots_[k].push_back({b, i * cls_size_[k], -1, 0});
            }
            n_slots += slots_[k].size();
            slot_bytes += slots_[k].size() * cls_size_[k];
        }
        where_.assign(n_keys(), {-1, -1});
        std::fprintf(stderr, "[vulkan] expert cache: %zu slots = %.1f GB of VRAM (%.0f%% of %zu experts, %.1f GB); "
                             "%d CPU threads compute the rest from RAM\n",
                     n_slots, slot_bytes / 1073741824.0, 100.0 * n_slots / n_keys(), n_keys(), total / 1073741824.0, cpu_.size());
        preload();
    }

    // Fill the cache with the most used experts (persisted counts), or evenly across layers.
    void preload() {
        std::vector<int> order(n_keys());
        std::iota(order.begin(), order.end(), 0);
        // stable: equal counts keep "expert-major" order (expert 0 of all layers, then expert 1, ...)
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const int ea = a % c_.n_experts, eb = b % c_.n_experts;
            return usage_[a] != usage_[b] ? usage_[a] > usage_[b] : ea < eb;
        });
        auto t0 = std::chrono::steady_clock::now();
        std::vector<size_t> used(slots_.size(), 0);
        std::vector<std::pair<int, int>> todo;  // (key, slot)
        for (int key : order) {
            const int k = L_[key / c_.n_experts].cls;
            if (used[k] == slots_[k].size()) continue;
            const int sidx = (int)used[k]++;
            slots_[k][sidx].key = key;
            where_[key] = {k, sidx};
            todo.push_back({key, sidx});
        }
        // repack on all threads, a staging buffer's worth at a time
        const size_t batch = std::max<size_t>(1, (STAGING / 2) / cls_size_[0] / 2);
        std::vector<std::vector<uint8_t>> tmp(batch);
        for (size_t i = 0; i < todo.size(); i += batch) {
            const size_t cnt = std::min(batch, todo.size() - i);
            cpu_.parallel_for((int64_t)cnt, [&](int64_t b, int64_t e) {
                for (int64_t j = b; j < e; j++) {
                    tmp[j].resize(L_[todo[i + j].first / c_.n_experts].slot_size);
                    write_expert(todo[i + j].first, tmp[j].data());
                }
            });
            for (size_t j = 0; j < cnt; j++) {
                const Slot& sl = slots_[where_[todo[i + j].first].first][todo[i + j].second];
                up_.put({sl.buf, sl.off}, tmp[j].data(), tmp[j].size());
            }
        }
        up_.flush();
        std::fprintf(stderr, "[vulkan] preloaded %zu experts (%s) in %.1fs\n", todo.size(),
                     std::any_of(usage_.begin(), usage_.end(), [](uint64_t u) { return u > 0; }) ? "by usage stats" : "uniform",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // Moves `key` into the cache if it has been used more in this session than the least used cached expert of
    // its class (not counting slots touched in the current pass). Called between passes. The persisted counts
    // only order the preload: what is hot depends on the conversation.
    void try_fill(int key) {
        const int k = L_[key / c_.n_experts].cls;
        auto& sl = slots_[k];
        int v = -1;
        for (int i = 0; i < (int)sl.size(); i++) {
            if (sl[i].key < 0) { v = i; break; }
            if (sl[i].tick == tick_) continue;
            if (v < 0 || heat(sl[i].key) < heat(sl[v].key)) v = i;
        }
        if (v < 0) return;
        if (sl[v].key >= 0) {
            if (heat(key) < heat(sl[v].key) + fill_margin()) return;
            where_[sl[v].key] = {-1, -1};
        }
        up_.put_fn({sl[v].buf, sl[v].off}, cls_size_[k], [&](void* d) { write_expert(key, (uint8_t*)d); });
        sl[v].key = key;
        sl[v].tick = tick_;
        where_[key] = {k, v};
        fills_++;
    }

    void load_stats() {
        if (opt_.stats_path.empty()) return;
        std::ifstream f(opt_.stats_path, std::ios::binary);
        if (!f) return;
        std::vector<uint64_t> v(n_keys(), 0);  // the draft block's counts are the tail and may be missing
        f.read((char*)v.data(), v.size() * 8);
        if ((size_t)f.gcount() >= (size_t)c_.n_layers * c_.n_experts * 8) usage_ = v;
    }
    void save_stats() {
        if (opt_.stats_path.empty()) return;
        std::ofstream f(opt_.stats_path, std::ios::binary);
        f.write((const char*)usage_.data(), usage_.size() * 8);
    }

    // ---- recording helpers --------------------------------------------------------------------
    // y[n][rows] = x[n][x_stride >= cols] W^T; post 1 applies silu(y * post_scale).
    void mat(const GM& m, uint64_t x, uint64_t y, int n, int x_stride, uint32_t post = 0, float post_scale = 0, const char* label = nullptr) {
        PcMat pc{m.w.addr(), m.s.addr(), 0, 0, x, y, (uint32_t)m.rows, (uint32_t)m.cols, (uint32_t)n, (uint32_t)x_stride,
                 (uint32_t)m.rows, post, post_scale};
        if (n >= GEMM_MIN && m.rows % 64 == 0)
            gpu_.dispatch(m.q8 ? "q4_gemm_q8" : "q4_gemm_f32", pc, groups(m.rows, 64), groups(n, 64), 1, label);
        else
            gpu_.dispatch(m.q8 ? "q4_matmul_q8" : "q4_matmul_f32", pc, groups(m.rows, 8), groups(n, NT), 1, label);
    }
    void mat_swiglu(const GM& g, const GM& u, uint64_t x, uint64_t y, int n) {
        PcMat pc{g.w.addr(), g.s.addr(), u.w.addr(), u.s.addr(), x, y, (uint32_t)g.rows, (uint32_t)g.cols, (uint32_t)n,
                 (uint32_t)g.cols, (uint32_t)g.rows, 0, 0};
        if (n >= GEMM_MIN && g.rows % 64 == 0)
            gpu_.dispatch(g.q8 ? "q4_gemm_q8_swiglu" : "q4_gemm_f32_swiglu", pc, groups(g.rows, 64), groups(n, 64));
        else
            gpu_.dispatch(g.q8 ? "q4_matmul_q8_swiglu" : "q4_matmul_f32_swiglu", pc, groups(g.rows, 8), groups(n, NT));
    }

    // Mixes the residual streams at x[n][hc][H] into the block input xn_[n][H]; inj_ gets the injection logits.
    void hc_mix(const GHc& h, uint64_t x, int n) {
        const int H = c_.hidden, HC = c_.hc_dim();
        PcHc pn{x, h.norm.addr(), xnorm_.addr(), (uint32_t)H, (uint32_t)c_.hc, c_.rms_eps};
        gpu_.dispatch("q4_hc_norm", pn, n, c_.hc);
        gpu_.barrier();
        mat(h.down, xnorm_.addr(), lo_.addr(), n, HC, 1, 1.0f / c_.hc, "hc_down");
        if (h.has_inject) mat(h.inject, xnorm_.addr(), inj_.addr(), n, HC, 0, 0, "hc_inject");
        gpu_.barrier();
        mat(h.up, lo_.addr(), gate_.addr(), n, c_.hc_rank, 0, 0, "hc_up");
        gpu_.barrier();
        PcHc pm{xnorm_.addr(), gate_.addr(), xn_.addr(), (uint32_t)H, (uint32_t)c_.hc, c_.rms_eps};
        gpu_.dispatch("q4_hc_mix", pm, groups(H, 256), n);
        gpu_.barrier();
    }
    // x[t][s] += 2 * sigmoid(inj_[t][s] / hc) * mix_[t]
    void hc_combine(uint64_t x, int n) {
        PcHc pc{x, inj_.addr(), mix_.addr(), (uint32_t)c_.hidden, (uint32_t)c_.hc, c_.rms_eps};
        gpu_.dispatch("q4_hc_combine", pc, groups(c_.hidden, 256), n);
        gpu_.barrier();
    }

    void ple(int l, int n) {
        const GLayer& g = L_[l];
        const int H = c_.hidden, HC = c_.hc_dim();
        const VkDeviceSize hb = (VkDeviceSize)ple_hist_ * HC * 4;
        mat(g.ple_key, emb_.addr(), pkey_.addr(), n, H, 0, 0, "ple_key");
        mat(g.ple_value, emb_.addr(), pval_.addr(), n, H, 0, 0, "ple_value");
        gpu_.copy(ple_state_, 0, ppad_.b, ppad_.off, hb);
        gpu_.barrier();
        PcPleGate pg{pkey_.addr(), pval_.addr(), x_.addr(), g.ple_wk.addr(), g.ple_wq.addr(), g.ple_wc.addr(), pgated_.addr(),
                     ppad_.addr(), (uint32_t)H, (uint32_t)c_.hc, (uint32_t)ple_hist_, c_.rms_eps};
        gpu_.dispatch("q4_ple_gate", pg, n, c_.hc);
        gpu_.barrier();
        PcPleConv pv{x_.addr(), pgated_.addr(), ppad_.addr(), g.ple_conv.addr(), (uint32_t)HC, (uint32_t)c_.ple_kernel,
                     (uint32_t)c_.ple_ngram, (uint32_t)ple_hist_};
        gpu_.dispatch("q4_ple_conv", pv, groups(HC, 256), n);
        gpu_.copy(ppad_.b, ppad_.off + (VkDeviceSize)n * HC * 4, ple_state_, 0, hb);
        gpu_.barrier();
    }

    // snap: also store the recurrent state after every token but the last (speculative pass)
    void deltanet(int l, int n, bool snap) {
        const GLayer& g = L_[l];
        const int s = slot_[l], H = c_.hidden;
        mat(g.qkv, xn_.addr(), qkv_.addr(), n, H, 0, 0, "dn_qkv");
        mat(g.z, xn_.addr(), z_.addr(), n, H, 0, 0, "dn_z");
        mat(g.alpha, xn_.addr(), a_.addr(), n, H, 0, 0, "dn_alpha");
        mat(g.beta, xn_.addr(), b_.addr(), n, H, 0, 0, "dn_beta");
        gpu_.barrier();
        const VkDeviceSize cs = (VkDeviceSize)(c_.conv_kernel - 1) * c_.conv_dim() * 4;
        PcConv pcc{qkv_.addr(), g.conv.addr(), conv_state_->addr + s * cs, (uint32_t)n, (uint32_t)c_.conv_dim(),
                   snap ? conv_snap_->addr + s * cs : 0, (uint32_t)conv_state_->size};
        gpu_.dispatch(snap ? "conv1d_snap" : "conv1d", pcc, groups(c_.conv_dim(), 256));
        gpu_.barrier();
        const VkDeviceSize rs = (VkDeviceSize)c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4;
        PcDelta pcd{qkv_.addr(), a_.addr(), b_.addr(), g.neg_a.addr(), g.dt_bias.addr(), rec_state_->addr + s * rs,
                    core_.addr(), (uint32_t)n, (uint32_t)c_.conv_dim(), (uint32_t)c_.lin_key_dim(),
                    (uint32_t)c_.lin_v_heads, (uint32_t)(c_.lin_v_heads / c_.lin_k_heads),
                    snap ? rec_snap_->addr + s * rs : 0, (uint32_t)rec_state_->size};
        gpu_.dispatch(snap ? "deltanet_tiled_snap" : "deltanet_tiled", pcd, c_.lin_v_heads, c_.lin_v_dim / 16);
        gpu_.barrier();
        PcGatedNorm pcg{core_.addr(), z_.addr(), g.lnorm.addr(), (uint32_t)c_.lin_v_heads, c_.rms_eps};
        gpu_.dispatch("gated_norm_sigmoid", pcg, n, c_.lin_v_heads);
        gpu_.barrier();
        mat(g.out, core_.addr(), mix_.addr(), n, c_.lin_value_dim(), 0, 0, "dn_out");
        gpu_.barrier();
    }

    // Returns with a command buffer open; `deferred` collects the QSA layers whose keys the CPU caches after
    // the pass (dense attention), the others sync here to get their selection mask.
    void attention(int l, int n, int pos, std::vector<int>& deferred) {
        const GLayer& g = L_[l];
        const int H = c_.hidden, AD = c_.n_heads * c_.head_dim;
        const VkDeviceSize kv = (VkDeviceSize)slot_[l] * kv_cap_ * c_.n_kv_heads * c_.head_dim * 2;
        bool masked = false;
        if (qsa_slot_[l] >= 0) {
            const VkDeviceSize ho = (VkDeviceSize)qsa_slot_[l] * NMAX * H * 4;
            gpu_.copy(xn_.b, xn_.off, hqsa_, ho, (VkDeviceSize)n * H * 4);
            if (qsa_.needs_select(l, pos, n)) {
                gpu_.barrier();
                gpu_.submit_wait();
                const float* xn = (const float*)(hqsa_->map + ho);
                std::vector<std::vector<int>> blocks;
                qsa_.update(l, xn, n, pos);
                qsa_.select(l, xn, n, pos, blocks);
                uint32_t* mask = (uint32_t*)hmask_->map;
                for (int t = 0; t < n; t++) {
                    uint32_t* m = mask + (size_t)t * mask_words_;
                    // no selection for this token: every complete block is visible
                    std::fill(m, m + mask_words_, blocks[t].empty() ? 0xFFFFFFFFu : 0u);
                    for (int b : blocks[t]) m[b >> 5] |= 1u << (b & 31);
                }
                gpu_.begin();
                masked = true;
            } else {
                deferred.push_back(l);
            }
        }
        mat(g.q, xn_.addr(), qg_.addr(), n, H, 0, 0, "attn_q");
        mat(g.k, xn_.addr(), kb_.addr(), n, H, 0, 0, "attn_k");
        mat(g.v, xn_.addr(), vb_.addr(), n, H, 0, 0, "attn_v");
        gpu_.barrier();
        PcQkRope pr{qg_.addr(), kb_.addr(), vb_.addr(), qo_.addr(), kcache_->addr + kv, vcache_->addr + kv,
                    g.qn.addr(), g.kn.addr(), rope_.addr(), (uint32_t)pos, (uint32_t)c_.n_heads,
                    (uint32_t)c_.n_kv_heads, (uint32_t)c_.rope_dim, c_.rms_eps};
        gpu_.dispatch("qk_norm_rope_kv16", pr, n, c_.n_heads + c_.n_kv_heads);
        gpu_.barrier();
        // split-K over keys: many splits when few tokens (decode), few when the batch is already wide
        const int maxlen = pos + n;
        int splits = n <= NT ? (maxlen + 255) / 256 : (maxlen + 1023) / 1024;
        splits = std::max(1, std::min(splits, std::min(64, ATT_PART_MAX / n)));
        const int chunk = (maxlen + splits - 1) / splits;
        const float scale = 1.0f / std::sqrt((float)c_.head_dim);
        if (masked) {
            PcAttnQsa pa{qo_.addr(), kcache_->addr + kv, vcache_->addr + kv, part_.addr(), (uint32_t)pos, (uint32_t)c_.n_heads,
                         (uint32_t)c_.n_kv_heads, (uint32_t)chunk, (uint32_t)splits, scale, hmask_->addr,
                         (uint32_t)w_.layers[l].compress, (uint32_t)mask_words_};
            gpu_.dispatch("attention_qsa_kv16", pa, n, c_.n_heads, splits);
        } else {
            PcAttn pa{qo_.addr(), kcache_->addr + kv, vcache_->addr + kv, part_.addr(), (uint32_t)pos, (uint32_t)c_.n_heads,
                      (uint32_t)c_.n_kv_heads, (uint32_t)chunk, (uint32_t)splits, scale};
            gpu_.dispatch("attention_kv16", pa, n, c_.n_heads, splits);
        }
        gpu_.barrier();
        PcAttnCombine pc{part_.addr(), qg_.addr(), att_.addr(), (uint32_t)c_.n_heads, (uint32_t)splits};
        gpu_.dispatch("attn_combine", pc, n, c_.n_heads);
        gpu_.barrier();
        mat(g.o, att_.addr(), mix_.addr(), n, AD, 0, 0, "attn_o");
        gpu_.barrier();
    }

    // Routed experts of layer l. Syncs once after routing: the GPU runs the experts it has cached while the CPU
    // computes the others from the mapped GGUF; the block output lands in mix_. Leaves a command buffer open.
    void moe(int l, int n) {
        const GLayer& g = L_[l];
        const Q4Layer& W = w_.layers[l];
        const int H = c_.hidden, TK = c_.top_k, E = c_.n_experts, I = c_.moe_inter;
        mat(g.router, xn_.addr(), rlogits_.addr(), n, H, 0, 0, "router");
        mat(g.sgate, xn_.addr(), sgate_.addr(), n, H, 0, 0, "shared_gate");
        mat_swiglu(g.sh_gate, g.sh_up, xn_.addr(), sact_.addr(), n);
        gpu_.barrier();
        mat(g.sh_down, sact_.addr(), sy_.addr(), n, c_.shared_inter, 0, 0, "shared_down");
        PcRoute pr{rlogits_.addr(), sel_.addr(), selw_.addr(), (uint32_t)E, (uint32_t)TK};
        gpu_.dispatch("moe_route", pr, n);
        gpu_.barrier();
        gpu_.copy(sel_.b, sel_.off, hsel_, 0, (VkDeviceSize)n * TK * 4);
        gpu_.copy(xn_.b, xn_.off, hxn_, 0, (VkDeviceSize)n * H * 4);
        gpu_.barrier();
        auto tp0 = std::chrono::steady_clock::now();
        gpu_.submit_wait();
        auto tp1 = std::chrono::steady_clock::now();

        // split the selections into cached (GPU) and other (CPU) experts, build the GPU tables
        const uint32_t* sel = (const uint32_t*)hsel_->map;
        if (trace_) {  // QW_TRACE: [1, layer, n, selections[n * top_k]]
            const uint32_t h[3] = {1, (uint32_t)l, (uint32_t)n};
            std::fwrite(h, 4, 3, trace_);
            std::fwrite(sel, 4, (size_t)n * TK, trace_);
        }
        per_.resize(E);
        for (auto& p : per_) p.clear();
        for (int i = 0; i < n * TK; i++) per_[sel[i]].push_back(i);  // selection index = t * TK + k
        const bool gemm = n >= EXPERT_GEMM_MIN && I % 64 == 0 && H % 64 == 0;
        const uint32_t chunk = gemm ? 64 : NT, rpw = gemm ? 64 : 8;
        ExpertEntry* ent = (ExpertEntry*)htab_->map;
        uint32_t* toks = (uint32_t*)(htab_->map + htab_toks_);
        uint32_t* comb = (uint32_t*)(htab_->map + htab_comb_);
        uint32_t ne = 0, row = 0;
        cpu_experts_.clear();
        for (int e = 0; e < E; e++) {
            if (per_[e].empty()) continue;
            const int key = l * E + e;
            usage_[key] += per_[e].size();
            recent_[key] += (uint32_t)per_[e].size();
            if (ema_ > 0) heat_[key] += (float)per_[e].size();
            if (use_pred_) pass_used_.push_back({key, (uint32_t)per_[e].size()});
            if (where_[key].first < 0) {
                misses_ += per_[e].size();
                cpu_experts_.push_back(e);
                pending_.push_back(key);
                continue;
            }
            hits_ += per_[e].size();
            slots_[where_[key].first][where_[key].second].tick = tick_;
            const uint64_t base = slot_addr(where_[key]);
            for (size_t i0 = 0; i0 < per_[e].size(); i0 += chunk) {
                const uint32_t cnt = (uint32_t)std::min<size_t>(chunk, per_[e].size() - i0);
                ent[ne++] = {base, row, cnt};
                for (uint32_t j = 0; j < cnt; j++) {
                    const int i = per_[e][i0 + j];
                    toks[row + j] = (uint32_t)(i / TK);
                    comb[2 * i] = row + j;
                }
                row += cnt;
            }
        }
        const uint32_t gpu_rows = row;

        // GPU: cached experts, submitted right away so they run while the CPU computes the others; everything
        // recorded after that is queued behind them and waited for at the next sync
        gpu_.begin();
        if (ne) {
            const uint64_t tab = htab_->addr, tk = htab_->addr + htab_toks_;
            PcExpert pu{tab, tk, xn_.addr(), act_.addr(), 0, g.s_gate, g.off_up, g.s_up, (uint32_t)I, (uint32_t)H,
                        (uint32_t)H, (uint32_t)I, 0};
            gpu_.dispatch((gemm ? g.gemm_up : g.pipe_up).c_str(), pu, groups(I, rpw), ne);
            gpu_.barrier();
            PcExpert pd{tab, tk, act_.addr(), ey_.addr(), g.off_down, g.s_down, 0, 0, (uint32_t)H, (uint32_t)I,
                        (uint32_t)I, (uint32_t)H, 1};
            gpu_.dispatch((gemm ? g.gemm_down : g.pipe_down).c_str(), pd, groups(H, rpw), ne);
            gpu_.barrier();
            gpu_.submit_chain();
        }

        // CPU: the other experts, straight from the mapped weights
        size_t rows_cpu = 0;
        for (int e : cpu_experts_) rows_cpu += per_[e].size();
        if (rows_cpu) {
            const float* xn = (const float*)hxn_->map;
            float* cy = (float*)hcy_->map;
            cg_.resize(rows_cpu * I);
            cu_.resize(rows_cpu * I);
            cx_.resize(rows_cpu * H);
            cmats_.resize(cpu_experts_.size() * 3);
            std::vector<gq::Job> jobs;
            size_t r = 0;
            for (size_t a = 0; a < cpu_experts_.size(); a++) {
                const int e = cpu_experts_[a], cnt = (int)per_[e].size();
                cmats_[a * 3] = W.ex_gate.slice(e);
                cmats_[a * 3 + 1] = W.ex_up.slice(e);
                cmats_[a * 3 + 2] = W.ex_down.slice(e);
                // a single token's input is used in place; several are gathered into consecutive rows
                const float* x = xn + (size_t)(per_[e][0] / TK) * H;
                if (cnt > 1) {
                    for (int j = 0; j < cnt; j++) std::memcpy(&cx_[(r + j) * H], xn + (size_t)(per_[e][j] / TK) * H, (size_t)H * 4);
                    x = &cx_[r * H];
                }
                for (int j = 0; j < cnt; j++) comb[2 * per_[e][j]] = gpu_rows + (uint32_t)(r + j);
                jobs.push_back({&cmats_[a * 3], x, &cg_[r * I], cnt});
                jobs.push_back({&cmats_[a * 3 + 1], x, &cu_[r * I], cnt});
                r += cnt;
            }
            gq::run_jobs(cpu_, jobs);
            cpu_.parallel_for((int64_t)cg_.size(), [&](int64_t b, int64_t e2) {
                for (int64_t i = b; i < e2; i++) cg_[i] = ck::silu(cg_[i]) * cu_[i];
            }, 4096);
            jobs.clear();
            r = 0;
            for (size_t a = 0; a < cpu_experts_.size(); a++) {
                const int cnt = (int)per_[cpu_experts_[a]].size();
                jobs.push_back({&cmats_[a * 3 + 2], &cg_[r * I], cy + r * H, cnt});
                r += cnt;
            }
            gq::run_jobs(cpu_, jobs);
        }
        t_gpu_ += std::chrono::duration<double>(tp1 - tp0).count();
        t_cpu_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tp1).count();

        // GPU: bring the CPU rows next to the GPU rows, then combine (weights from moe_route via selw_)
        if (rows_cpu) gpu_.copy(hcy_, 0, ey_.b, ey_.off + (VkDeviceSize)gpu_rows * H * 4, (VkDeviceSize)rows_cpu * H * 4);
        gpu_.copy(htab_, htab_comb_, comb_.b, comb_.off, (VkDeviceSize)n * TK * 8);
        gpu_.barrier();
        PcCombW pw{comb_.addr(), selw_.addr(), (uint32_t)(n * TK)};
        gpu_.dispatch("comb_weights", pw, groups(n * TK, 256));
        gpu_.barrier();
        PcCombine pc{mix_.addr(), sgate_.addr(), sy_.addr(), ey_.addr(), comb_.addr(), (uint32_t)H, (uint32_t)TK};
        gpu_.dispatch("moe_combine_set", pc, groups(H, 256), n);
        gpu_.barrier();
    }

    // One trunk pass over n tokens. logits: of the last token, or with `spec` (a speculative pass of at most
    // draft_ + 1 tokens, see forward_all) of every token, with per-token snapshots of the recurrent state.
    void forward_chunk(const int* tokens, int n, float* logits, bool spec) {
        ensure_kv(pos_ + n);
        const int H = c_.hidden, HC = c_.hc_dim();
        float* hx = (float*)hx_->map;
        for (int t = 0; t < n; t++) toks_[pos_ + t] = tokens[t];
        if (trace_) {  // QW_TRACE: [0, position, n, tokens[n]]
            const uint32_t h[3] = {0, (uint32_t)pos_, (uint32_t)n};
            std::fwrite(h, 4, 3, trace_);
            std::fwrite(tokens, 4, (size_t)n, trace_);
        }
        cpu_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) {
                float* x = hx + (size_t)t * HC;
                gq::dequant_row(w_.embed.type, w_.embed.row(tokens[t]), H, x);
                for (int s = 1; s < c_.hc; s++) std::memcpy(x + (size_t)s * H, x, H * sizeof(float));
                if (ple_hist_) q4_ple_embed(c_, w_, toks_.data(), pos_ + (int)t, (float*)hemb_->map + (size_t)t * H);
            }
        }, 4);
        if (use_pred_) {  // learn from the previous pass and rescore the experts
            auto tq0 = std::chrono::steady_clock::now();
            if (!pred_ready_) {
                const char* pf = std::getenv("QW_PRED_FILE");
                pred_.init(n_keys(), usage_, pf ? pf : opt_.stats_path.empty() ? "" : opt_.stats_path + ".pred");
                pred_ready_ = true;
                std::fprintf(stderr, "[vulkan] expert predictor: %llu training steps loaded\n",(unsigned long long)pred_.trained());
            }
            // The step for a pass runs on its own thread during the following pass (the trunk mostly waits
            // for the GPU), so the scores the fills below read are one pass behind.
            if (pred_job_.valid()) pred_job_.get();
            t_pred_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tq0).count();
        }
        // cache fills queued by the previous pass
        // (the most used of the experts the previous pass missed, over all layers)
        {
            const size_t nf = std::min<size_t>(MAX_FILLS, pending_.size());
            std::partial_sort(pending_.begin(), pending_.begin() + nf, pending_.end(),
                              [&](int a, int b) { return heat(a) != heat(b) ? heat(a) > heat(b) : a < b; });
            auto tf0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < nf; i++)
                if (where_[pending_[i]].first < 0) try_fill(pending_[i]);
            t_fill_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
            pending_.clear();
        }
        if (use_pred_ && !pass_used_.empty()) {
            pred_job_ = std::async(std::launch::async, [this, used = std::move(pass_used_)] { pred_.step(used); });
            pass_used_.clear();
        }
        up_.flush();
        tick_++;
        if (ema_ > 0) { const float k = 1.f - 1.f / ema_; for (float& h : heat_) h *= k; }
        gpu_.begin();
        gpu_.copy(hx_, 0, x_.b, x_.off, (VkDeviceSize)n * HC * 4);
        if (ple_hist_) gpu_.copy(hemb_, 0, emb_.b, emb_.off, (VkDeviceSize)n * H * 4);
        gpu_.barrier();
        std::vector<int> deferred;
        for (int l = 0; l < c_.n_layers; l++) {
            const GLayer& g = L_[l];
            if (w_.layers[l].ple) ple(l, n);
            hc_mix(g.hc_attn, x_.addr(), n);
            if (w_.layers[l].full_attn) attention(l, n, pos_, deferred);
            else deltanet(l, n, spec && n > 1);
            hc_combine(x_.addr(), n);
            hc_mix(g.hc_ffn, x_.addr(), n);
            moe(l, n);
            hc_combine(x_.addr(), n);
        }
        const int n_out = spec ? n : 1;
        if (logits) {  // the final mixer is the output norm
            hc_mix(hc_head_, x_.addr() + (uint64_t)(n - n_out) * HC * 4, n_out);
            mat(output_, xn_.addr(), logits_.addr(), n_out, H, 0, 0, "lm_head");
            gpu_.barrier();
            gpu_.copy(logits_.b, logits_.off, hlog_, 0, (VkDeviceSize)c_.vocab * 4 * n_out);
        }
        gpu_.barrier();
        gpu_.submit_wait();
        if (logits) std::memcpy(logits, hlog_->map, (size_t)c_.vocab * 4 * n_out);
        for (int l : deferred)
            qsa_.update(l, (const float*)(hqsa_->map + (size_t)qsa_slot_[l] * NMAX * H * 4), n, pos_);
        pos_ += n;
    }

    // Runs the draft block on n rows at positions p0.. : row i pairs the token toks[i] with the residual streams
    // of the position before it. from_trunk: those are the pending row (hlast_) followed by rows 0..n-2 of the
    // last trunk pass (x_); otherwise (a further draft step, n == 1) the block's own previous output. commit:
    // row n-1 of the trunk pass becomes the new pending row. logits: of the last row (the token after toks[n-1]).
    void mtp_run(const int* toks, int n, int p0, float* logits, bool from_trunk, bool commit) {
        ensure_kv(std::min(max_ctx_, p0 + n + 1));
        const int H = c_.hidden, HC = c_.hc_dim(), l = c_.n_layers;
        const GLayer& g = L_[l];
        float* he = (float*)hme_->map;
        cpu_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) gq::dequant_row(w_.embed.type, w_.embed.row(toks[t]), H, he + (size_t)t * H);
        }, 16);
        gpu_.begin();
        gpu_.copy(hme_, 0, memb_.b, memb_.off, (VkDeviceSize)n * H * 4);
        if (from_trunk) {
            gpu_.copy(hlast_, 0, mh_.b, mh_.off, (VkDeviceSize)HC * 4);
            if (n > 1) gpu_.copy(x_.b, x_.off, mh_.b, mh_.off + (VkDeviceSize)HC * 4, (VkDeviceSize)(n - 1) * HC * 4);
        } else {
            gpu_.copy(mx_.b, mx_.off, mh_.b, mh_.off, (VkDeviceSize)HC * 4);
        }
        gpu_.barrier();
        if (commit) gpu_.copy(x_.b, x_.off + (VkDeviceSize)(n - 1) * HC * 4, hlast_, 0, (VkDeviceSize)HC * 4);
        // stream s of the block input = eh_proj(norm(embedding) | norm(stream s of the residual))
        PcNorm pe{memb_.addr(), 0, mtp_enorm_.addr(), men_.addr(), (uint32_t)H, c_.rms_eps};
        gpu_.dispatch("add_rmsnorm", pe, n);
        PcHc ph{mh_.addr(), mtp_hnorm_.addr(), xnorm_.addr(), (uint32_t)H, (uint32_t)c_.hc, c_.rms_eps};
        gpu_.dispatch("q4_hc_norm", ph, n, c_.hc);
        gpu_.barrier();
        PcHc pcat{men_.addr(), xnorm_.addr(), mcat_.addr(), (uint32_t)H, (uint32_t)c_.hc, c_.rms_eps};
        gpu_.dispatch("q4_hc_cat", pcat, groups(2 * H, 256), n * c_.hc);
        gpu_.barrier();
        mat(mtp_eh_, mcat_.addr(), mx_.addr(), n * c_.hc, 2 * H, 0, 0, "mtp_eh");
        gpu_.barrier();
        std::vector<int> deferred;
        hc_mix(g.hc_attn, mx_.addr(), n);
        attention(l, n, p0, deferred);
        hc_combine(mx_.addr(), n);
        hc_mix(g.hc_ffn, mx_.addr(), n);
        moe(l, n);
        hc_combine(mx_.addr(), n);
        if (logits) {
            hc_mix(mtp_head_, mx_.addr() + (uint64_t)(n - 1) * HC * 4, 1);
            mat(output_, xn_.addr(), logits_.addr(), 1, H, 0, 0, "mtp_head");
            gpu_.barrier();
            gpu_.copy(logits_.b, logits_.off, hlog_, 0, (VkDeviceSize)c_.vocab * 4);
        }
        gpu_.barrier();
        gpu_.submit_wait();
        if (logits) std::memcpy(logits, hlog_->map, (size_t)c_.vocab * 4);
        for (int dl : deferred)
            qsa_.update(dl, (const float*)(hqsa_->map + (size_t)qsa_slot_[dl] * NMAX * H * 4), n, p0);
    }
};

}  // namespace

std::unique_ptr<Backend> make_qwen4_vulkan_backend(const Q4Config& cfg, const Q4Weights& w, int max_ctx, const VulkanOptions& opt) {
    return std::make_unique<Qwen4Vulkan>(cfg, w, max_ctx, opt);
}
