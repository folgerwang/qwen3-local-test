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
};

std::unique_ptr<Backend> make_cpu_backend(const Config& cfg, const ModelWeights& w, int max_ctx, int threads);
struct VulkanOptions {
    size_t expert_cache_mb = 0;  // 0 = auto (all free VRAM minus a margin)
    int threads = 0;             // GGUF backend: CPU threads for the experts computed from RAM (0 = auto)
    bool expert_cache_set = false;  // GGUF backend: expert_cache_mb was given (0 = no VRAM cache)
    std::string stats_path;      // expert usage counts, loaded for preload and updated on exit ("" = off)
    bool pin_experts = true;     // keep all experts in pinned host memory the GPU reads directly on a miss
    int hybrid = -1;             // CPU computes cache misses: -1 auto (only if pinning fails), 0 off, 1 on
};
std::unique_ptr<Backend> make_vulkan_backend(const Config& cfg, const ModelWeights& w, int max_ctx, const VulkanOptions& opt);
