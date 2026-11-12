// Inference backend interface: CPU and Vulkan implement the same forward pass.
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"

struct ImageInput;

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

    // Vision: encode_image runs the vision tower and returns the merged embeddings [tokens][hidden];
    // forward_image appends them as image tokens (a lgh x lgw grid) at the current position, with 2D
    // rotary positions. logits may be null.
    // One line about memory use (VRAM, expert cache, pinned RAM), or "".
    virtual std::string memory_report() const { return ""; }

    virtual bool has_vision() const { return false; }
    virtual std::vector<float> encode_image(const ImageInput&) { throw std::runtime_error("no vision support"); }
    virtual void forward_image(const float*, int, int, float*) { throw std::runtime_error("no vision support"); }
};

std::unique_ptr<Backend> make_cpu_backend(const Config& cfg, const ModelWeights& w, int max_ctx, int threads);
struct VulkanOptions {
    size_t expert_cache_mb = 0;  // 0 = auto (all free VRAM minus a margin)
    std::string stats_path;      // expert usage counts, loaded for preload and updated on exit ("" = off)
    bool pin_experts = true;     // keep all experts in pinned host memory the GPU reads directly on a miss
    int hybrid = -1;             // CPU computes cache misses: -1 auto (only if pinning fails), 0 off, 1 on
    bool vision = true;          // load the vision tower (when the model has one)
    int64_t image_max_pixels = 1 << 20;  // largest image the vision buffers are sized for
    size_t vram_reserve_mb = 5120;       // VRAM left free for other programs (auto expert cache size)
    size_t ram_reserve_mb = 16384;       // system RAM left free: experts are not pinned if that would cut into it
};
std::unique_ptr<Backend> make_vulkan_backend(const Config& cfg, const ModelWeights& w, int max_ctx, const VulkanOptions& opt);
