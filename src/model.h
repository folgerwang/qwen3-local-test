// Qwen3.5/3.6 MoE text model: config and weight views (pointers into the mmapped safetensors).
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "safetensors.h"

struct Config {
    int hidden = 0;            // 2048
    int n_layers = 0;          // 40
    int vocab = 0;             // 248320
    float rms_eps = 1e-6f;
    // full attention
    int n_heads = 0;           // 16
    int n_kv_heads = 0;        // 2
    int head_dim = 0;          // 256
    int rope_dim = 0;          // 64 (partial rotary)
    float rope_theta = 0;
    // gated deltanet (linear attention)
    int lin_k_heads = 0;       // 16
    int lin_v_heads = 0;       // 32
    int lin_k_dim = 0;         // 128
    int lin_v_dim = 0;         // 128
    int conv_kernel = 0;       // 4
    // moe
    int n_experts = 0;         // 256
    int top_k = 0;             // 8
    int moe_inter = 0;         // 512
    int shared_inter = 0;      // 512
    std::vector<bool> is_full_attn;  // per layer
    std::vector<int> eos_ids;

    int lin_key_dim() const { return lin_k_heads * lin_k_dim; }
    int lin_value_dim() const { return lin_v_heads * lin_v_dim; }
    int conv_dim() const { return 2 * lin_key_dim() + lin_value_dim(); }

    static Config load(const std::string& model_dir);
};

// FP8 e4m3 weight [rows, cols] with one bf16 scale per 128x128 block ([rows/128, cols/128]).
struct QMat {
    const uint8_t* w = nullptr;
    const uint16_t* scale = nullptr;
    int rows = 0, cols = 0;
    size_t weight_bytes() const { return (size_t)rows * cols; }
    size_t scale_count() const { return (size_t)(rows / 128) * (cols / 128); }
};

// BF16 weight [rows, cols] (or a vector when rows == 1). Small tensors stored as F32 in the checkpoint
// are kept as floats in `f32` instead (w == nullptr); use at() for element access.
struct BMat {
    const uint16_t* w = nullptr;
    int rows = 0, cols = 0;
    std::vector<float> f32;
    float at(size_t i) const;
};

struct Expert {
    QMat gate, up, down;
};

struct LayerWeights {
    bool full_attn = false;
    BMat input_norm, post_norm;
    // full attention
    QMat q_proj, k_proj, v_proj, o_proj;
    BMat q_norm, k_norm;
    // gated deltanet
    QMat in_qkv, in_z, out_proj;
    BMat in_a, in_b, conv1d, a_log, dt_bias, lin_norm;
    // moe
    BMat router, shared_gate;
    Expert shared;
    std::vector<Expert> experts;
};

struct ModelWeights {
    BMat embed, lm_head, final_norm;
    std::vector<LayerWeights> layers;
    void load(const SafeTensors& st, const Config& cfg);
};

// MoE routing: top-k of the router logits; weights = softmax over the selected logits
// (equal to softmax over all experts followed by renormalization over the top-k).
void route_topk(const float* logits, int n_experts, int k, int* ids, float* weights);

inline float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline float BMat::at(size_t i) const { return w ? bf16_to_f32(w[i]) : f32[i]; }
