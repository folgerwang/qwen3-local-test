// Inference backend interface: CPU and Vulkan implement the same forward pass.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "model.h"

class Backend {
public:
    virtual ~Backend() = default;
    // Clears KV cache / recurrent state.
    virtual void reset() = 0;
    // Appends `n` tokens at the current position; writes logits of the last token into `logits` (vocab floats).
    virtual void forward(const int* tokens, int n, float* logits) = 0;
    virtual int pos() const = 0;
    // Single-slot snapshot of the recurrent state + position (KV entries below pos are never rewritten),
    // used to re-feed the canonical form of the last assistant turn.
    virtual void save_state() = 0;
    virtual void restore_state() = 0;
    virtual int max_ctx() const = 0;
    virtual const char* name() const = 0;

    // Speculative decoding with a draft head (optional). The caller samples the next token, asks for a draft of
    // the tokens after it, feeds all of them in one forward_all() and keeps the prefix the model agrees with:
    //   n = draft(next, k, d);  forward_all({next, d[0..n)}, n + 1, logits);  accept(kept);
    // max_draft(): how many tokens draft() can propose (0 = no draft head).
    virtual int max_draft() const { return 0; }
    // Proposes up to n tokens following `token`, which is about to be fed at pos(). Stops when unsure.
    virtual int draft(int token, int n, int* out) { (void)token; (void)n; (void)out; return 0; }
    // Like forward(), for up to max_draft() + 1 tokens, returning the logits of every position (n * vocab).
    // Must be followed by accept().
    virtual void forward_all(const int* tokens, int n, float* logits) { (void)tokens; (void)n; (void)logits; }
    // Keeps the first `keep` (>= 1) tokens of the last forward_all() and drops the rest.
    virtual void accept(int keep) { (void)keep; }

    // Expert cache counters since the start (backends that keep experts in a VRAM cache): lookups served from
    // the cache and by the CPU, experts cached now and experts in the model. False when there is no such cache.
    struct CacheStats { unsigned long long hits = 0, misses = 0, cached = 0, experts = 0; };
    virtual bool cache_stats(CacheStats& s) const { (void)s; return false; }
};

std::unique_ptr<Backend> make_cpu_backend(const Config& cfg, const ModelWeights& w, int max_ctx, int threads);
struct VulkanOptions {
    size_t expert_cache_mb = 0;  // 0 = auto (all free VRAM minus a margin)
    int threads = 0;             // GGUF backend: CPU threads for the experts computed from RAM (0 = auto)
    int draft = 0;               // GGUF backend: tokens drafted per pass with the MTP head (0 = off)
    bool expert_cache_set = false;  // GGUF backend: expert_cache_mb was given (0 = no VRAM cache)
    std::string stats_path;      // expert usage counts, loaded for preload and updated on exit ("" = off)
    bool pin_experts = true;     // keep all experts in pinned host memory the GPU reads directly on a miss
    int hybrid = -1;             // CPU computes cache misses: -1 auto (only if pinning fails), 0 off, 1 on
};
std::unique_ptr<Backend> make_vulkan_backend(const Config& cfg, const ModelWeights& w, int max_ctx, const VulkanOptions& opt);
