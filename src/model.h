// Qwen3.5/3.6 MoE and Qwen3.8-Flash-Next (qwen4_exp) text models: config and weight views (pointers into the
// mmapped safetensors).
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
    // qwen4_exp (Qwen3.8-Flash-Next): hc parallel residual streams mixed around every block, BF16 dense
    // weights, sigmoid DeltaNet output gate, hashed n-gram embedding (PLE) at one layer, sparse attention indexer
    bool qwen4 = false;
    int hc = 1;                // 4 residual streams
    int hc_rank = 0;           // 320
    int ple_layer = -1;        // 0-based layer index that adds the PLE features (-1 = none)
    int ngram = 0;             // 3
    int ngram_heads = 0;       // (ngram - 1) * heads_per_ngram = 16
    int ple_dim = 0;           // 2560
    int ple_conv = 0;          // 4 (dilation = ngram)
    int ple_eos = 0;           // segment separator for the n-gram history
    int idx_heads = 0;         // 4
    int idx_dim = 0;           // 128
    int idx_budget = 0;        // 2048 tokens
    int idx_ratio = 0;         // 4 tokens per pooled key block

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

// Dense projection: FP8 (Qwen3.5/3.6) or BF16 (qwen4_exp).
struct Lin {
    QMat q;
    BMat b;
    bool fp8() const { return q.w != nullptr; }
    int rows() const { return fp8() ? q.rows : b.rows; }
    int cols() const { return fp8() ? q.cols : b.cols; }
};

struct Expert {
    QMat gate, up, down;
};

// Gated residual over hc streams: norm (grouped per stream), low-rank input mix, per-stream output injection.
struct HyperConn {
    BMat norm, down, up, inject;  // inject is empty for the final mixer
};

// Hashed n-gram embedding ("PLE"): FP8 table split into row shards, one per-tensor scale.
struct Ple {
    BMat key_proj, value_proj, norm_key, norm_query, norm_conv, conv;
    std::vector<const uint8_t*> shards;
    int64_t shard_rows = 0;
    int dim = 0;  // per-head embedding width (160)
    float scale = 1.0f;
    int64_t mult[4] = {};
    std::vector<int64_t> offsets, sizes;  // per n-gram head
};

struct LayerWeights {
    bool full_attn = false;
    BMat input_norm, post_norm;
    // full attention
    Lin q_proj, k_proj, v_proj, o_proj;
    BMat q_norm, k_norm;
    BMat idx_qk, idx_qn, idx_kn;  // qwen4 sparse attention indexer
    // gated deltanet
    Lin in_qkv, in_z, out_proj;
    BMat in_a, in_b, conv1d, a_log, dt_bias, lin_norm;
    // moe
    BMat router, shared_gate;
    Lin sh_gate, sh_up, sh_down;
    std::vector<Expert> experts;
    HyperConn attn_hc, mlp_hc;  // qwen4
};

struct ModelWeights {
    BMat embed, lm_head, final_norm;
    std::vector<LayerWeights> layers;
    HyperConn mixer;  // qwen4: folds the streams back into one before lm_head (replaces final_norm)
    Ple ple;
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
