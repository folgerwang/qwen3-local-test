// Vulkan compute backend.
//
// Memory placement
//   * Dense weights (attention, DeltaNet, shared experts, router, lm_head) live in VRAM.
//   * Every routed expert has a copy in pinned host memory that the GPU can read directly over PCIe,
//     laid out exactly like a VRAM cache slot. A VRAM cache of fixed-size slots holds the hot experts.
//   * A per-(layer, expert) location table on the GPU holds the address the shaders read from: the
//     cache slot when cached, the pinned host copy otherwise.
//
// Execution: routing, grouping and expert dispatch happen on the GPU (indirect dispatch), so a forward
// pass is a single submission. Afterwards the CPU reads the selected expert ids, updates LRU state and
// usage counts, and schedules copies of missed experts into cache slots at the start of the next pass.
// Hybrid mode (models whose experts mostly do not fit in VRAM): each layer syncs once after routing; the
// GPU runs the cached experts while the CPU computes the missed ones straight from RAM (no PCIe weight
// traffic), then the GPU combines both. No pinned copy is needed in this mode.
// If pinned memory is unavailable in non-hybrid mode, the backend falls back to a CPU sync per layer
// that uploads misses before the experts run.
//
// Shaders get every buffer as a device address in push constants; there are no descriptor sets.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "cpu_kernels.h"
#include "vk_common.h"

namespace {

using namespace vkc;

constexpr int NMAX = 1024;            // max tokens per forward chunk (big chunks reuse expert weights more)
constexpr int NT = 8;                 // tokens per matvec pass (matches matmul_fp8 / matmul_bf16)
constexpr int GEMM_MIN = 9;           // dense matrices use the tiled GEMM from this many tokens
constexpr int EXPERT_GEMM_MIN = 33;   // grouped experts use the tiled GEMM from this many tokens
constexpr int ATT_PART_MAX = 2048;    // max (tokens x key splits) for split-K attention scratch
constexpr int MAX_UPLOADS = 32;       // cache fills per forward pass (bounds added latency)
constexpr int FILL_MB = 32;           // cache-fill byte budget per pass

constexpr int MAX_UPLOADS_HYBRID = 16; // same in hybrid mode, where a fill is a CPU memcpy + upload

// ---------------------------------------------------------------------------------------------
// Push-constant layouts (must match the shaders, std430)
// ---------------------------------------------------------------------------------------------
struct PcMatFp8 { uint64_t w1, s1, w2, s2, x, y; uint32_t rows, cols, n, x_stride, y_stride, accumulate; };
struct PcGemm { uint64_t w1, s1, w2, s2, x, y; uint32_t rows, cols, n, x_stride, y_stride; };
struct PcExpert { uint64_t table, toks, x, y; uint32_t woff1, soff1, woff2, soff2, rows, cols, x_stride, y_stride, x_mode; };
struct PcMatBf16 { uint64_t w, x, y; uint32_t rows, cols, n, x_stride, y_stride; };
struct PcNorm { uint64_t x, delta, w, out; uint32_t dim; float eps; };
struct PcConv { uint64_t data, w, state; uint32_t n, channels; };
struct PcDelta { uint64_t qkv, a, b, neg_a, dt_bias, state, out; uint32_t n, conv_dim, key_dim, n_v_heads, ratio; };
struct PcGatedNorm { uint64_t o, z, w; uint32_t n_heads; float eps; };
struct PcQkRope { uint64_t qg, k, v, q_out, kcache, vcache, qn, kn, rope; uint32_t pos, n_heads, n_kv, rope_dim; float eps; };
struct PcAttn { uint64_t q, kcache, vcache, part; uint32_t pos, n_heads, n_kv, chunk, n_split; float scale; };
struct PcAttnCombine { uint64_t part, qg, out; uint32_t n_heads, n_split; };
struct PcRoute { uint64_t logits, sel, selw; uint32_t n_experts, top_k; };
struct PcGroup { uint64_t sel, selw, loc, entries, toks, comb, args; uint32_t n_sel, top_k, n_experts, chunk, gx_up, gx_down; };
struct PcCombine { uint64_t x, sgate, sy, ey, table; uint32_t dim, top_k; };
struct PcCombW { uint64_t comb, selw; uint32_t n; };

struct ExpertEntry { uint64_t base; uint32_t start, count; };  // matches `Entry` in the expert shaders
static_assert(sizeof(ExpertEntry) == 16, "ExpertEntry layout");

// GPU-side views of weights.
struct GQ { Ptr w, s; int rows = 0, cols = 0; };
struct GB { Ptr w; int rows = 0, cols = 0; };

struct GLayer {
    Ptr in_norm, post_norm;                 // f32
    GQ q, k, v, o; Ptr qn, kn;              // attention
    GQ qkv, z, out; GB a, b; Ptr conv, neg_a, dt_bias, lnorm;  // deltanet
    GB router, sgate; GQ sh_gate, sh_up, sh_down;
};

uint32_t groups(uint32_t n, uint32_t per) { return (n + per - 1) / per; }

// ---------------------------------------------------------------------------------------------
class VulkanBackend final : public Backend {
public:
    VulkanBackend(const Config& c, const ModelWeights& w, int max_ctx, const VulkanOptions& opt)
        : c_(c), w_(w), max_ctx_(max_ctx), opt_(opt), arena_(gpu_), up_(gpu_) {
        upload_weights();
        alloc_state();
        setup_experts();
        reset();
    }

    ~VulkanBackend() override {
        std::fprintf(stderr, "[vulkan] expert lookups: %llu cached, %llu %s; %llu cache fills\n", (unsigned long long)hits_,
                     (unsigned long long)misses_, hybrid_ ? "computed on the CPU" : "read from host memory",
                     (unsigned long long)fills_);
        save_stats();
    }

    void reset() override {
        pos_ = 0;
        gpu_.begin();
        gpu_.fill_zero(conv_state_);
        gpu_.fill_zero(rec_state_);
        gpu_.submit_wait();
    }
    int pos() const override { return pos_; }
    int max_ctx() const override { return max_ctx_; }
    const char* name() const override { return "vulkan"; }

    void save_state() override {
        gpu_.begin();
        gpu_.copy(conv_state_, 0, conv_saved_, 0, conv_state_->size);
        gpu_.copy(rec_state_, 0, rec_saved_, 0, rec_state_->size);
        gpu_.submit_wait();
        saved_pos_ = pos_;
    }
    void restore_state() override {
        gpu_.begin();
        gpu_.copy(conv_saved_, 0, conv_state_, 0, conv_state_->size);
        gpu_.copy(rec_saved_, 0, rec_state_, 0, rec_state_->size);
        gpu_.submit_wait();
        pos_ = saved_pos_;
    }

    void forward(const int* tokens, int n, float* logits) override {
        if (pos_ + n > max_ctx_) throw std::runtime_error("context overflow");
        for (int done = 0; done < n;) {
            const int m = std::min(NMAX, n - done);
            forward_chunk(tokens + done, m, done + m == n ? logits : nullptr);
            done += m;
        }
    }

private:
    const Config& c_;
    const ModelWeights& w_;
    int max_ctx_;
    VulkanOptions opt_;
    int pos_ = 0, saved_pos_ = 0;
    Gpu gpu_;
    Arena arena_;
    Uploader up_;

    std::vector<GLayer> L_;
    GB lm_head_;
    Ptr final_norm_, rope_;
    std::vector<int> slot_;  // per layer: index into full-attn or linear-attn state arrays

    // activations (device local)
    Ptr x_, xn_, mix_, qkv_, z_, a_, b_, core_, qg_, kb_, vb_, qo_, att_, part_, sact_, sy_, sgate_, act_, ey_;
    Ptr rlogits_, selw_, toks_, comb_, entries_, args_, sel_, logits_;
    Buf* kcache_ = nullptr;
    Buf* vcache_ = nullptr;
    Buf* conv_state_ = nullptr;
    Buf* rec_state_ = nullptr;
    Buf* conv_saved_ = nullptr;
    Buf* rec_saved_ = nullptr;
    // host-visible
    Buf* hx_ = nullptr;     // embeddings of the current chunk
    Buf* hsel_ = nullptr;   // selected expert ids, [layer][NMAX * top_k]
    Buf* hlog_ = nullptr;   // final logits

    // experts
    struct Slot { Buf* buf = nullptr; VkDeviceSize off = 0; int key = -1; uint64_t tick = 0; };
    std::vector<Slot> slots_;               // VRAM cache
    std::vector<int> where_;                // key (layer * n_experts + expert) -> slot or -1
    std::vector<Buf*> host_bufs_;           // pinned copies of all experts
    VkDeviceSize host_per_buf_ = 0;
    bool pinned_ = false;
    bool hybrid_ = false;                   // CPU computes cache misses (see header comment)
    std::unique_ptr<ThreadPool> cpu_;
    Buf* hxn_ = nullptr;                    // hybrid: normalized activations of the layer (GPU -> CPU)
    Buf* htab_ = nullptr;                   // hybrid: entries | toks | comb tables written by the CPU
    Buf* hcy_ = nullptr;                    // hybrid: expert outputs computed on the CPU (CPU -> GPU)
    VkDeviceSize htab_toks_ = 0, htab_comb_ = 0;
    uint64_t cpu_rows_ = 0;
    Ptr loc_;                               // [layers * n_experts] uint64 address the shaders read from
    std::vector<int> pending_;              // keys to bring into the cache at the start of the next pass
    std::vector<uint64_t> usage_;           // selection counts per key (persisted)
    VkDeviceSize slot_size_ = 0, off_up_ = 0, off_down_ = 0, off_sg_ = 0, off_su_ = 0, off_sd_ = 0;
    uint64_t tick_ = 0, hits_ = 0, misses_ = 0, fills_ = 0;

    // ---- setup ----------------------------------------------------------------------------
    GQ up_q(const QMat& m) {
        GQ g;
        g.rows = m.rows;
        g.cols = m.cols;
        g.w = arena_.alloc(m.weight_bytes());
        g.s = arena_.alloc(m.scale_count() * 2);
        up_.put(g.w, m.w, m.weight_bytes());
        up_.put(g.s, m.scale, m.scale_count() * 2);
        return g;
    }
    GB up_b(const BMat& m) {
        GB g;
        g.rows = m.rows;
        g.cols = m.cols;
        g.w = arena_.alloc((size_t)m.rows * m.cols * 2);
        up_.put(g.w, m.w, (size_t)m.rows * m.cols * 2);
        return g;
    }
    Ptr up_f32(const std::vector<float>& v) {
        Ptr p = arena_.alloc(v.size() * 4);
        up_.put(p, v.data(), v.size() * 4);
        return p;
    }
    static std::vector<float> to_f32(const BMat& m) {
        std::vector<float> v((size_t)m.rows * m.cols);
        for (size_t i = 0; i < v.size(); i++) v[i] = m.at(i);
        return v;
    }

    void upload_weights() {
        lm_head_ = up_b(w_.lm_head);
        final_norm_ = up_f32(to_f32(w_.final_norm));
        int nf = 0, nl = 0;
        for (int l = 0; l < c_.n_layers; l++) {
            const LayerWeights& W = w_.layers[l];
            GLayer g;
            g.in_norm = up_f32(to_f32(W.input_norm));
            g.post_norm = up_f32(to_f32(W.post_norm));
            if (W.full_attn) {
                slot_.push_back(nf++);
                g.q = up_q(W.q_proj); g.k = up_q(W.k_proj); g.v = up_q(W.v_proj); g.o = up_q(W.o_proj);
                g.qn = up_f32(to_f32(W.q_norm));
                g.kn = up_f32(to_f32(W.k_norm));
            } else {
                slot_.push_back(nl++);
                g.qkv = up_q(W.in_qkv); g.z = up_q(W.in_z); g.out = up_q(W.out_proj);
                g.a = up_b(W.in_a); g.b = up_b(W.in_b);
                g.conv = up_f32(to_f32(W.conv1d));
                std::vector<float> na = to_f32(W.a_log);
                for (float& v : na) v = -std::exp(v);
                g.neg_a = up_f32(na);
                g.dt_bias = up_f32(to_f32(W.dt_bias));
                g.lnorm = up_f32(to_f32(W.lin_norm));
            }
            g.router = up_b(W.router);
            g.sgate = up_b(W.shared_gate);
            g.sh_gate = up_q(W.shared.gate); g.sh_up = up_q(W.shared.up); g.sh_down = up_q(W.shared.down);
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
    }

    void alloc_state() {
        const int H = c_.hidden, TK = c_.top_k;
        const int n_full = (int)std::count(c_.is_full_attn.begin(), c_.is_full_attn.end(), true);
        const int n_lin = c_.n_layers - n_full;
        auto act = [&](size_t floats) { return arena_.alloc(floats * 4); };
        x_ = act((size_t)NMAX * H);
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
        toks_ = act((size_t)NMAX * TK);
        comb_ = act((size_t)NMAX * TK * 2);
        entries_ = arena_.alloc((VkDeviceSize)NMAX * TK * sizeof(ExpertEntry));
        args_ = arena_.alloc(64);
        sel_ = arena_.alloc((VkDeviceSize)c_.n_layers * NMAX * TK * 4);
        logits_ = arena_.alloc((VkDeviceSize)c_.vocab * 4);
        const VkDeviceSize kv = (VkDeviceSize)n_full * max_ctx_ * c_.n_kv_heads * c_.head_dim * 4;
        kcache_ = gpu_.create(std::max<VkDeviceSize>(kv, 256), Mem::Device);
        vcache_ = gpu_.create(std::max<VkDeviceSize>(kv, 256), Mem::Device);
        const VkDeviceSize cs = std::max<VkDeviceSize>((VkDeviceSize)n_lin * (c_.conv_kernel - 1) * c_.conv_dim() * 4, 256);
        const VkDeviceSize rs = std::max<VkDeviceSize>((VkDeviceSize)n_lin * c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4, 256);
        conv_state_ = gpu_.create(cs, Mem::Device);
        rec_state_ = gpu_.create(rs, Mem::Device);
        conv_saved_ = gpu_.create(cs, Mem::Device);
        rec_saved_ = gpu_.create(rs, Mem::Device);
        hx_ = gpu_.create((VkDeviceSize)NMAX * H * 4, Mem::HostUpload);
        hsel_ = gpu_.create((VkDeviceSize)c_.n_layers * NMAX * TK * 4, Mem::HostRead);
        hlog_ = gpu_.create((VkDeviceSize)c_.vocab * 4, Mem::HostRead);
    }

    // ---- experts ------------------------------------------------------------------------------
    size_t n_keys() const { return (size_t)c_.n_layers * c_.n_experts; }
    const Expert& expert_of(int key) const { return w_.layers[key / c_.n_experts].experts[key % c_.n_experts]; }
    Ptr host_ptr(int key) const {
        return {host_bufs_[key / host_per_buf_], (VkDeviceSize)(key % host_per_buf_) * slot_size_};
    }
    uint64_t home_addr(int key) const { return pinned_ ? host_ptr(key).addr() : 0; }

    void write_expert(const Expert& ex, uint8_t* d) const {
        std::memcpy(d, ex.gate.w, ex.gate.weight_bytes());
        std::memcpy(d + off_up_, ex.up.w, ex.up.weight_bytes());
        std::memcpy(d + off_down_, ex.down.w, ex.down.weight_bytes());
        std::memcpy(d + off_sg_, ex.gate.scale, ex.gate.scale_count() * 2);
        std::memcpy(d + off_su_, ex.up.scale, ex.up.scale_count() * 2);
        std::memcpy(d + off_sd_, ex.down.scale, ex.down.scale_count() * 2);
    }

    void setup_experts() {
        const Expert& e0 = w_.layers[0].experts[0];
        auto align = [](VkDeviceSize v) { return (v + 255) & ~VkDeviceSize(255); };
        VkDeviceSize o = align(e0.gate.weight_bytes());
        off_up_ = o; o = align(o + e0.up.weight_bytes());
        off_down_ = o; o = align(o + e0.down.weight_bytes());
        off_sg_ = o; o = align(o + e0.gate.scale_count() * 2);
        off_su_ = o; o = align(o + e0.up.scale_count() * 2);
        off_sd_ = o; o = align(o + e0.down.scale_count() * 2);
        slot_size_ = o;
        usage_.assign(n_keys(), 0);
        load_stats();

        // VRAM cache slots
        VkDeviceSize budget;
        if (opt_.expert_cache_mb) budget = (VkDeviceSize)opt_.expert_cache_mb << 20;
        else {
            const VkDeviceSize free = gpu_.device_free(), margin = 1536ull << 20;  // driver / display headroom
            budget = free > margin ? free - margin : 0;
        }
        size_t want = std::min<size_t>(n_keys(), budget / slot_size_);
        want = std::max<size_t>(want, std::min<size_t>(n_keys(), c_.n_experts));  // at least one layer's worth
        const size_t per_chunk = CHUNK / slot_size_;
        while (slots_.size() < want) {
            const size_t k = std::min(per_chunk, want - slots_.size());
            Buf* b = gpu_.create(k * slot_size_, Mem::Device, true);
            if (!b) break;
            for (size_t i = 0; i < k; i++) slots_.push_back({b, i * slot_size_, -1, 0});
        }
        where_.assign(n_keys(), -1);
        // Misses are read by the GPU from pinned host memory when it fits (fast on PCIe 4/5 machines: measured
        // faster than computing them on the CPU here). If pinning fails, or with --hybrid, the CPU computes them.
        hybrid_ = opt_.hybrid > 0;
        if (!hybrid_) {
            // pinned host copies of every expert
            host_per_buf_ = CHUNK / slot_size_;
            if (opt_.pin_experts) {
                const size_t nbuf = (n_keys() + host_per_buf_ - 1) / host_per_buf_;
                pinned_ = true;
                for (size_t i = 0; i < nbuf; i++) {
                    const size_t cnt = std::min<size_t>(host_per_buf_, n_keys() - i * host_per_buf_);
                    Buf* b = gpu_.create(cnt * slot_size_, Mem::HostUpload, true);
                    if (!b) {  // not enough pinnable memory: release what we got
                        pinned_ = false;
                        for (Buf* x : host_bufs_) gpu_.destroy(x);
                        host_bufs_.clear();
                        break;
                    }
                    host_bufs_.push_back(b);
                }
                if (pinned_) {
                    auto t0 = std::chrono::steady_clock::now();
                    const int nth = std::max(1, std::min(16, (int)std::thread::hardware_concurrency()));
                    std::vector<std::thread> th;
                    for (int t = 0; t < nth; t++)
                        th.emplace_back([&, t] {
                            for (size_t k = t; k < n_keys(); k += nth) write_expert(expert_of((int)k), host_bufs_[k / host_per_buf_]->map + (k % host_per_buf_) * slot_size_);
                        });
                    for (auto& x : th) x.join();
                    std::fprintf(stderr, "[vulkan] pinned %zu experts (%.1f GB) in host memory in %.1fs\n", n_keys(),
                                 n_keys() * slot_size_ / 1073741824.0,
                                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                } else {
                    std::fprintf(stderr, "[vulkan] could not pin %.1f GB of experts in host memory\n", n_keys() * slot_size_ / 1073741824.0);
                }
            }
            if (!pinned_ && opt_.hybrid < 0) hybrid_ = true;
            if (!pinned_ && !hybrid_ && slots_.size() < (size_t)c_.n_experts) throw std::runtime_error("not enough VRAM for the expert cache");
        }
        if (hybrid_) {
            const int H = c_.hidden, TK = c_.top_k;
            cpu_ = std::make_unique<ThreadPool>(0);
            hxn_ = gpu_.create((VkDeviceSize)NMAX * H * 4, Mem::HostRead);
            hcy_ = gpu_.create((VkDeviceSize)NMAX * TK * H * 4, Mem::HostUpload);
            const VkDeviceSize ne = (VkDeviceSize)NMAX * TK;
            htab_toks_ = (ne * sizeof(ExpertEntry) + 255) & ~VkDeviceSize(255);
            htab_comb_ = (htab_toks_ + ne * 4 + 255) & ~VkDeviceSize(255);
            htab_ = gpu_.create(htab_comb_ + ne * 8, Mem::HostUpload);
            std::fprintf(stderr, "[vulkan] hybrid mode: GPU runs cached experts, %d CPU threads compute the rest\n", cpu_->size());
        }
        std::fprintf(stderr, "[vulkan] expert cache: %zu slots x %.2f MB = %.1f GB (%.0f%% of %zu experts)\n",
                     slots_.size(), slot_size_ / 1048576.0, slots_.size() * slot_size_ / 1073741824.0,
                     100.0 * slots_.size() / n_keys(), n_keys());

        // location table: everything starts at its pinned home, then preload overwrites cached entries
        loc_ = arena_.alloc(n_keys() * 8);
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
        const size_t n = std::min(slots_.size(), order.size());
        auto t0 = std::chrono::steady_clock::now();
        if (pinned_) {
            for (size_t i = 0; i < n; i += 256) {  // batched GPU copies from the pinned copies
                gpu_.begin();
                for (size_t k = i; k < std::min(n, i + 256); k++) {
                    const Ptr h = host_ptr(order[k]);
                    gpu_.copy(h.b, h.off, slots_[k].buf, slots_[k].off, slot_size_);
                }
                gpu_.submit_wait();
            }
        } else {
            for (size_t k = 0; k < n; k++)
                up_.put_fn({slots_[k].buf, slots_[k].off}, slot_size_, [&](void* d) { write_expert(expert_of(order[k]), (uint8_t*)d); });
            up_.flush();
        }
        for (size_t k = 0; k < n; k++) {
            slots_[k].key = order[k];
            where_[order[k]] = (int)k;
        }
        std::vector<uint64_t> loc(n_keys());
        for (size_t k = 0; k < n_keys(); k++) loc[k] = where_[k] >= 0 ? slot_addr(where_[k]) : home_addr((int)k);
        up_.put(loc_, loc.data(), loc.size() * 8);
        up_.flush();
        std::fprintf(stderr, "[vulkan] preloaded %zu experts (%s) in %.1fs\n", n,
                     std::any_of(usage_.begin(), usage_.end(), [](uint64_t u) { return u > 0; }) ? "by usage stats" : "uniform",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // cache fills per pass, by bytes: big experts would otherwise eat the PCIe bandwidth the misses need
    int max_fills() const { return std::max(1, std::min(MAX_UPLOADS, (int)((size_t)FILL_MB << 20) / (int)slot_size_)); }

    uint64_t slot_addr(int s) const { return slots_[s].buf->addr + slots_[s].off; }

    // Picks the least recently used slot not used in the current pass.
    int victim() {
        int v = -1;
        for (int i = 0; i < (int)slots_.size(); i++) {
            if (slots_[i].key < 0) return i;
            if (slots_[i].tick == tick_) continue;
            if (v < 0 || slots_[i].tick < slots_[v].tick) v = i;
        }
        return v;
    }

    // Records (into the open command buffer) moving `key` into a cache slot; returns false if none free.
    // With pinned memory the copy is GPU-side; otherwise data goes through the staging uploader.
    bool fill_slot(int key) {
        const int s = victim();
        if (s < 0) return false;
        Slot& sl = slots_[s];
        if (sl.key >= 0) {
            where_[sl.key] = -1;
            const uint64_t home = home_addr(sl.key);
            gpu_.update({loc_.b, loc_.off + (VkDeviceSize)sl.key * 8}, &home, 8);
        }
        if (pinned_) {
            const Ptr h = host_ptr(key);
            gpu_.copy(h.b, h.off, sl.buf, sl.off, slot_size_);
        } else {
            up_.put_fn({sl.buf, sl.off}, slot_size_, [&](void* d) { write_expert(expert_of(key), (uint8_t*)d); });
        }
        const uint64_t a = slot_addr(s);
        gpu_.update({loc_.b, loc_.off + (VkDeviceSize)key * 8}, &a, 8);
        sl.key = key;
        sl.tick = tick_;
        where_[key] = s;
        fills_++;
        return true;
    }

    // After a pass: count usage, refresh LRU ticks, queue misses for the next pass.
    void account(int l, int n) {
        const uint32_t* sel = (const uint32_t*)(hsel_->map + (size_t)l * NMAX * c_.top_k * 4);
        for (int i = 0; i < n * c_.top_k; i++) {
            const int key = l * c_.n_experts + (int)sel[i];
            usage_[key]++;
            if (where_[key] >= 0) {
                hits_++;
                slots_[where_[key]].tick = tick_;
            } else {
                misses_++;
                if (pinned_ && (int)pending_.size() < max_fills() &&
                    std::find(pending_.begin(), pending_.end(), key) == pending_.end())
                    pending_.push_back(key);
            }
        }
    }

    void load_stats() {
        if (opt_.stats_path.empty()) return;
        std::ifstream f(opt_.stats_path, std::ios::binary);
        if (!f) return;
        std::vector<uint64_t> v(n_keys());
        if (f.read((char*)v.data(), v.size() * 8)) usage_ = v;
    }
    void save_stats() {
        if (opt_.stats_path.empty()) return;
        std::ofstream f(opt_.stats_path, std::ios::binary);
        f.write((const char*)usage_.data(), usage_.size() * 8);
    }

    // ---- recording helpers --------------------------------------------------------------------
    void mat_q(const GQ& m, Ptr x, Ptr y, int n) {
        if (n >= GEMM_MIN) {
            PcGemm pc{m.w.addr(), m.s.addr(), 0, 0, x.addr(), y.addr(), (uint32_t)m.rows, (uint32_t)m.cols, (uint32_t)n,
                      (uint32_t)m.cols, (uint32_t)m.rows};
            gpu_.dispatch(gpu_.coop() ? "gemm_coop" : "gemm_fp8", pc, groups(m.rows, 64), groups(n, 64));
            return;
        }
        PcMatFp8 pc{m.w.addr(), m.s.addr(), 0, 0, x.addr(), y.addr(), (uint32_t)m.rows, (uint32_t)m.cols, (uint32_t)n,
                    (uint32_t)m.cols, (uint32_t)m.rows, 0};
        gpu_.dispatch("matmul_fp8", pc, groups(m.rows, 8), groups(n, NT));
    }
    void mat_swiglu(const GQ& g, const GQ& u, Ptr x, Ptr y, int n) {
        if (n >= GEMM_MIN) {
            PcGemm pc{g.w.addr(), g.s.addr(), u.w.addr(), u.s.addr(), x.addr(), y.addr(), (uint32_t)g.rows,
                      (uint32_t)g.cols, (uint32_t)n, (uint32_t)g.cols, (uint32_t)g.rows};
            gpu_.dispatch(gpu_.coop() ? "gemm_coop_swiglu" : "gemm_fp8_swiglu", pc, groups(g.rows, 64), groups(n, 64));
            return;
        }
        PcMatFp8 pc{g.w.addr(), g.s.addr(), u.w.addr(), u.s.addr(), x.addr(), y.addr(), (uint32_t)g.rows,
                    (uint32_t)g.cols, (uint32_t)n, (uint32_t)g.cols, (uint32_t)g.rows, 0};
        gpu_.dispatch("matmul_fp8_swiglu", pc, groups(g.rows, 8), groups(n, NT));
    }
    void mat_b(const GB& m, uint64_t x, uint64_t y, int n, int x_stride, const char* label = nullptr) {
        PcMatBf16 pc{m.w.addr(), x, y, (uint32_t)m.rows, (uint32_t)m.cols, (uint32_t)n, (uint32_t)x_stride, (uint32_t)m.rows};
        gpu_.dispatch("matmul_bf16", pc, groups(m.rows, 8), groups(n, NT), 1, label);
    }
    void norm(Ptr x, uint64_t delta, Ptr w, Ptr out, int n) {
        PcNorm pc{x.addr(), delta, w.addr(), out.addr(), (uint32_t)c_.hidden, c_.rms_eps};
        gpu_.dispatch("add_rmsnorm", pc, n);
    }

    void deltanet(int l, int n) {
        const GLayer& g = L_[l];
        const int s = slot_[l];
        mat_q(g.qkv, xn_, qkv_, n);
        mat_q(g.z, xn_, z_, n);
        mat_b(g.a, xn_.addr(), a_.addr(), n, c_.hidden);
        mat_b(g.b, xn_.addr(), b_.addr(), n, c_.hidden);
        gpu_.barrier();
        const VkDeviceSize cs = (VkDeviceSize)(c_.conv_kernel - 1) * c_.conv_dim() * 4;
        PcConv pcc{qkv_.addr(), g.conv.addr(), conv_state_->addr + s * cs, (uint32_t)n, (uint32_t)c_.conv_dim()};
        gpu_.dispatch("conv1d", pcc, groups(c_.conv_dim(), 256));
        gpu_.barrier();
        const VkDeviceSize rs = (VkDeviceSize)c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4;
        PcDelta pcd{qkv_.addr(), a_.addr(), b_.addr(), g.neg_a.addr(), g.dt_bias.addr(), rec_state_->addr + s * rs,
                    core_.addr(), (uint32_t)n, (uint32_t)c_.conv_dim(), (uint32_t)c_.lin_key_dim(),
                    (uint32_t)c_.lin_v_heads, (uint32_t)(c_.lin_v_heads / c_.lin_k_heads)};
        gpu_.dispatch("deltanet", pcd, c_.lin_v_heads, c_.lin_v_dim / 16);
        gpu_.barrier();
        PcGatedNorm pcg{core_.addr(), z_.addr(), g.lnorm.addr(), (uint32_t)c_.lin_v_heads, c_.rms_eps};
        gpu_.dispatch("gated_norm", pcg, n, c_.lin_v_heads);
        gpu_.barrier();
        mat_q(g.out, core_, mix_, n);
    }

    void attention(int l, int n) {
        const GLayer& g = L_[l];
        const VkDeviceSize kv = (VkDeviceSize)slot_[l] * max_ctx_ * c_.n_kv_heads * c_.head_dim * 4;
        mat_q(g.q, xn_, qg_, n);
        mat_q(g.k, xn_, kb_, n);
        mat_q(g.v, xn_, vb_, n);
        gpu_.barrier();
        PcQkRope pr{qg_.addr(), kb_.addr(), vb_.addr(), qo_.addr(), kcache_->addr + kv, vcache_->addr + kv,
                    g.qn.addr(), g.kn.addr(), rope_.addr(), (uint32_t)pos_, (uint32_t)c_.n_heads,
                    (uint32_t)c_.n_kv_heads, (uint32_t)c_.rope_dim, c_.rms_eps};
        gpu_.dispatch("qk_norm_rope", pr, n, c_.n_heads + c_.n_kv_heads);
        gpu_.barrier();
        // split-K over keys: many splits when few tokens (decode), few when the batch is already wide
        const int maxlen = pos_ + n;
        int splits = n <= NT ? (maxlen + 255) / 256 : (maxlen + 1023) / 1024;
        splits = std::max(1, std::min(splits, std::min(64, ATT_PART_MAX / n)));
        const int chunk = (maxlen + splits - 1) / splits;
        PcAttn pa{qo_.addr(), kcache_->addr + kv, vcache_->addr + kv, part_.addr(), (uint32_t)pos_,
                  (uint32_t)c_.n_heads, (uint32_t)c_.n_kv_heads, (uint32_t)chunk, (uint32_t)splits,
                  1.0f / std::sqrt((float)c_.head_dim)};
        gpu_.dispatch("attention", pa, n, c_.n_heads, splits);
        gpu_.barrier();
        PcAttnCombine pc{part_.addr(), qg_.addr(), att_.addr(), (uint32_t)c_.n_heads, (uint32_t)splits};
        gpu_.dispatch("attn_combine", pc, n, c_.n_heads);
        gpu_.barrier();
        mat_q(g.o, att_, mix_, n);
    }

    void moe(int l, int n) {
        const GLayer& g = L_[l];
        const int H = c_.hidden, TK = c_.top_k, E = c_.n_experts, I = c_.moe_inter;
        const uint64_t sel = sel_.addr() + (uint64_t)l * NMAX * TK * 4;
        mat_b(g.router, xn_.addr(), rlogits_.addr(), n, H, "router");
        mat_b(g.sgate, xn_.addr(), sgate_.addr(), n, H, "shared_gate");
        mat_swiglu(g.sh_gate, g.sh_up, xn_, sact_, n);
        gpu_.barrier();
        mat_q(g.sh_down, sact_, sy_, n);
        PcRoute pr{rlogits_.addr(), sel, selw_.addr(), (uint32_t)E, (uint32_t)TK};
        gpu_.dispatch("moe_route", pr, n);
        gpu_.barrier();
        if (hybrid_) {
            moe_hybrid(l, n);
            return;
        }

        if (!pinned_) {
            // Fallback: experts must be resident before they run, so stop and upload the misses now.
            const VkDeviceSize so = (VkDeviceSize)l * NMAX * TK * 4;
            gpu_.copy(sel_.b, sel_.off + so, hsel_, so, (VkDeviceSize)n * TK * 4);
            gpu_.barrier();
            gpu_.submit_wait();
            tick_++;
            const uint32_t* s = (const uint32_t*)(hsel_->map + (size_t)l * NMAX * TK * 4);
            gpu_.begin();
            for (int i = 0; i < n * TK; i++) {
                const int key = l * E + (int)s[i];
                usage_[key]++;
                if (where_[key] >= 0) { hits_++; slots_[where_[key]].tick = tick_; continue; }
                misses_++;
                if (!fill_slot(key)) throw std::runtime_error("expert cache too small for this batch");
            }
            up_.flush();  // staged expert data (separate submission, before the loc updates execute)
            gpu_.barrier();
        }

        const bool gemm = n >= EXPERT_GEMM_MIN;
        const uint32_t chunk = gemm ? 64 : NT;
        const uint32_t rpw = gemm ? 64 : 8;  // rows per workgroup
        PcGroup pg{sel, selw_.addr(), loc_.addr() + (uint64_t)l * E * 8, entries_.addr(), toks_.addr(), comb_.addr(),
                   args_.addr(), (uint32_t)(n * TK), (uint32_t)TK, (uint32_t)E, chunk, groups(I, rpw), groups(H, rpw)};
        gpu_.dispatch("moe_group", pg, 1);
        gpu_.barrier();
        PcExpert pu{entries_.addr(), toks_.addr(), xn_.addr(), act_.addr(), 0, (uint32_t)off_sg_, (uint32_t)off_up_,
                    (uint32_t)off_su_, (uint32_t)I, (uint32_t)H, (uint32_t)H, (uint32_t)I, 0};
        const bool tc = gpu_.coop();
        gpu_.dispatch_indirect(gemm ? (tc ? "gemm_coop_expert_swiglu" : "gemm_fp8_expert_swiglu") : "matmul_fp8_expert_swiglu", pu, args_);
        gpu_.barrier();
        PcExpert pd{entries_.addr(), toks_.addr(), act_.addr(), ey_.addr(), (uint32_t)off_down_, (uint32_t)off_sd_, 0, 0,
                    (uint32_t)H, (uint32_t)I, (uint32_t)I, (uint32_t)H, 1};
        gpu_.dispatch_indirect(gemm ? (tc ? "gemm_coop_expert" : "gemm_fp8_expert") : "matmul_fp8_expert", pd, {args_.b, args_.off + 12});
        gpu_.barrier();
        PcCombine pc{x_.addr(), sgate_.addr(), sy_.addr(), ey_.addr(), comb_.addr(), (uint32_t)H, (uint32_t)TK};
        gpu_.dispatch("moe_combine", pc, groups(H, 256), n);
        gpu_.barrier();
    }

    // Hybrid MoE for layer l (command buffer open, routing recorded). Leaves a command buffer open.
    void moe_hybrid(int l, int n) {
        const int H = c_.hidden, TK = c_.top_k, E = c_.n_experts, I = c_.moe_inter;
        const VkDeviceSize so = (VkDeviceSize)l * NMAX * TK * 4;
        gpu_.copy(sel_.b, sel_.off + so, hsel_, so, (VkDeviceSize)n * TK * 4);
        gpu_.copy(xn_.b, xn_.off, hxn_, 0, (VkDeviceSize)n * H * 4);
        gpu_.barrier();
        gpu_.submit_wait();

        // CPU: split selections into cached (GPU) and missed (CPU) experts, build the GPU tables
        const uint32_t* sel = (const uint32_t*)(hsel_->map + so);
        std::vector<std::vector<int>> per(E);  // expert -> selection indices (t * TK + k)
        for (int i = 0; i < n * TK; i++) per[sel[i]].push_back(i);
        const bool gemm = n >= EXPERT_GEMM_MIN;
        const uint32_t chunk = gemm ? 64 : NT;
        ExpertEntry* ent = (ExpertEntry*)htab_->map;
        uint32_t* toks = (uint32_t*)(htab_->map + htab_toks_);
        uint32_t* comb = (uint32_t*)(htab_->map + htab_comb_);
        uint32_t ne = 0, row = 0;
        std::vector<int> cpu_experts;
        for (int e = 0; e < E; e++) {
            if (per[e].empty()) continue;
            const int key = l * E + e;
            usage_[key]++;
            if (where_[key] < 0) {
                misses_ += per[e].size();
                cpu_experts.push_back(e);
                if ((int)pending_.size() < MAX_UPLOADS_HYBRID && std::find(pending_.begin(), pending_.end(), key) == pending_.end())
                    pending_.push_back(key);
                continue;
            }
            hits_ += per[e].size();
            slots_[where_[key]].tick = tick_;
            const uint64_t base = slot_addr(where_[key]);
            for (size_t i0 = 0; i0 < per[e].size(); i0 += chunk) {
                const uint32_t cnt = (uint32_t)std::min<size_t>(chunk, per[e].size() - i0);
                ent[ne++] = {base, row, cnt};
                for (uint32_t j = 0; j < cnt; j++) {
                    const int i = per[e][i0 + j];
                    toks[row + j] = (uint32_t)(i / TK);
                    comb[2 * i] = row + j;
                }
                row += cnt;
            }
        }
        const uint32_t gpu_rows = row;

        // GPU: cached experts (runs while the CPU computes the misses)
        gpu_.begin();
        if (ne) {
            const uint64_t tab = htab_->addr, tk = htab_->addr + htab_toks_;
            PcExpert pu{tab, tk, xn_.addr(), act_.addr(), 0, (uint32_t)off_sg_, (uint32_t)off_up_, (uint32_t)off_su_,
                        (uint32_t)I, (uint32_t)H, (uint32_t)H, (uint32_t)I, 0};
            const bool tc = gpu_.coop();
            gpu_.dispatch(gemm ? (tc ? "gemm_coop_expert_swiglu" : "gemm_fp8_expert_swiglu") : "matmul_fp8_expert_swiglu", pu,
                          groups(I, gemm ? 64 : 8), ne);
            gpu_.barrier();
            PcExpert pd{tab, tk, act_.addr(), ey_.addr(), (uint32_t)off_down_, (uint32_t)off_sd_, 0, 0, (uint32_t)H,
                        (uint32_t)I, (uint32_t)I, (uint32_t)H, 1};
            gpu_.dispatch(gemm ? (tc ? "gemm_coop_expert" : "gemm_fp8_expert") : "matmul_fp8_expert", pd, groups(H, gemm ? 64 : 8), ne);
        }
        gpu_.submit();

        // CPU: missed experts, straight from the mmapped weights
        const float* xn = (const float*)hxn_->map;
        float* cy = (float*)hcy_->map;
        size_t rows_cpu = 0;
        for (int e : cpu_experts) rows_cpu += per[e].size();
        if (rows_cpu) {
            std::vector<float> xin(rows_cpu * H), g(rows_cpu * I), u(rows_cpu * I);
            std::vector<ck::MatJob> jobs;
            size_t r = 0;
            for (int e : cpu_experts) {
                const Expert& ex = w_.layers[l].experts[e];
                for (size_t j = 0; j < per[e].size(); j++) {
                    const int i = per[e][j];
                    std::memcpy(&xin[(r + j) * H], xn + (size_t)(i / TK) * H, (size_t)H * 4);
                    comb[2 * i] = gpu_rows + (uint32_t)(r + j);
                }
                const int cnt = (int)per[e].size();
                jobs.push_back({&ex.gate, nullptr, &xin[r * H], &g[r * I], cnt});
                jobs.push_back({&ex.up, nullptr, &xin[r * H], &u[r * I], cnt});
                r += cnt;
            }
            ck::run_jobs(*cpu_, jobs);
            cpu_->parallel_for((int64_t)g.size(), [&](int64_t b, int64_t e2) {
                for (int64_t i = b; i < e2; i++) g[i] = ck::silu(g[i]) * u[i];
            }, 4096);
            jobs.clear();
            r = 0;
            for (int e : cpu_experts) {
                const int cnt = (int)per[e].size();
                jobs.push_back({&w_.layers[l].experts[e].down, nullptr, &g[r * I], cy + r * H, cnt});
                r += cnt;
            }
            ck::run_jobs(*cpu_, jobs);
            cpu_rows_ += rows_cpu;
        }
        gpu_.wait();

        // GPU: bring CPU rows next to the GPU rows, then combine (weights from moe_route via selw_)
        gpu_.begin();
        if (rows_cpu) {
            gpu_.copy(hcy_, 0, ey_.b, ey_.off + (VkDeviceSize)gpu_rows * H * 4, (VkDeviceSize)rows_cpu * H * 4);
        }
        // comb rows come from the host table; weights are filled on the GPU from selw_
        gpu_.copy(htab_, htab_comb_, comb_.b, comb_.off, (VkDeviceSize)n * TK * 8);
        gpu_.barrier();
        PcCombW pw{comb_.addr(), selw_.addr(), (uint32_t)(n * TK)};
        gpu_.dispatch("comb_weights", pw, groups(n * TK, 256));
        gpu_.barrier();
        PcCombine pc{x_.addr(), sgate_.addr(), sy_.addr(), ey_.addr(), comb_.addr(), (uint32_t)H, (uint32_t)TK};
        gpu_.dispatch("moe_combine", pc, groups(H, 256), n);
        gpu_.barrier();
    }

    void forward_chunk(const int* tokens, int n, float* logits) {
        const int H = c_.hidden;
        float* hx = (float*)hx_->map;
        for (int t = 0; t < n; t++) {
            const uint16_t* e = w_.embed.w + (size_t)tokens[t] * H;
            for (int i = 0; i < H; i++) hx[(size_t)t * H + i] = bf16_to_f32(e[i]);
        }
        tick_++;
        gpu_.begin();
        // cache fills queued by the previous pass
        for (int key : pending_)
            if (where_[key] < 0 && !fill_slot(key)) break;
        up_.flush();  // non-pinned fills go through the staging buffer
        pending_.clear();
        gpu_.copy(hx_, 0, x_.b, x_.off, (VkDeviceSize)n * H * 4);
        gpu_.barrier();
        for (int l = 0; l < c_.n_layers; l++) {
            const GLayer& g = L_[l];
            norm(x_, 0, g.in_norm, xn_, n);
            gpu_.barrier();
            if (w_.layers[l].full_attn) attention(l, n);
            else deltanet(l, n);
            gpu_.barrier();
            norm(x_, mix_.addr(), g.post_norm, xn_, n);
            gpu_.barrier();
            moe(l, n);
        }
        if (logits) {
            const Ptr xl{x_.b, x_.off + (VkDeviceSize)(n - 1) * H * 4};
            norm(xl, 0, final_norm_, xn_, 1);
            gpu_.barrier();
            mat_b(lm_head_, xn_.addr(), logits_.addr(), 1, H, "lm_head");
            gpu_.barrier();
            gpu_.copy(logits_.b, logits_.off, hlog_, 0, (VkDeviceSize)c_.vocab * 4);
        }
        if (pinned_)  // selected expert ids of all layers, for cache bookkeeping
            for (int l = 0; l < c_.n_layers; l++) {
                const VkDeviceSize so = (VkDeviceSize)l * NMAX * c_.top_k * 4;
                gpu_.copy(sel_.b, sel_.off + so, hsel_, so, (VkDeviceSize)n * c_.top_k * 4);
            }
        gpu_.barrier();
        gpu_.submit_wait();
        if (logits) std::memcpy(logits, hlog_->map, (size_t)c_.vocab * 4);
        if (pinned_)
            for (int l = 0; l < c_.n_layers; l++) account(l, n);
        pos_ += n;
    }
};

}  // namespace

std::unique_ptr<Backend> make_vulkan_backend(const Config& cfg, const ModelWeights& w, int max_ctx, const VulkanOptions& opt) {
    return std::make_unique<VulkanBackend>(cfg, w, max_ctx, opt);
}
