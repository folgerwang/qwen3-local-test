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
    bool expert_q4 = false;    // routed experts are 4-bit GPTQ (the other weights BF16)
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
    // vision tower (Qwen3.5/3.6 checkpoints): ViT over 16x16 patches, 2x2 patch merge into the text model
    struct Vision {
        int depth = 0;             // 27 blocks (0 = no vision tower)
        int hidden = 0;            // 1152
        int heads = 0;             // 16 (head_dim 72)
        int inter = 0;             // 4304
        int patch = 0;             // 16
        int temporal = 0;          // 2 (an image is two identical frames)
        int merge = 0;             // 2
        int n_pos = 0;             // 2304 learned positions (48 x 48 grid)
        int out = 0;               // = text hidden size
        float rope_theta = 10000.0f;
        float mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};
        int64_t min_pixels = 65536;
        int image_token = -1, vision_start = -1, vision_end = -1;
        int patch_dim() const { return 3 * temporal * patch * patch; }
    } vis;

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

// 4-bit GPTQ weight [rows, cols] (symmetric, group 128, no act-order), as stored: qweight int32 [cols/8, rows]
// (8 consecutive inputs per word, low nibble first), fp16 scales [cols/128, rows]; value = (q - 8) * scale.
struct Q4Mat {
    const uint32_t* qweight = nullptr;
    const uint16_t* scales = nullptr;
    int rows = 0, cols = 0;
    size_t weight_bytes() const { return (size_t)rows * cols / 2; }
    size_t scale_count() const { return (size_t)rows * (cols / 128); }
};

struct Expert {
    QMat gate, up, down;  // FP8 checkpoints
    Q4Mat g4, u4, d4;     // 4-bit GPTQ checkpoints (cfg.expert_q4)
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

// Vision tower: all BF16, linear layers have biases, norms are LayerNorm (weight + bias).
struct VisionBlock {
    BMat norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b, norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b;
};
struct VisionWeights {
    BMat patch_w, patch_b;  // Conv3d [hidden, 3, temporal, patch, patch] seen as [hidden, patch_dim]
    BMat pos_embed;         // [n_pos, hidden]
    std::vector<VisionBlock> blocks;
    BMat merge_norm_w, merge_norm_b, merge_fc1_w, merge_fc1_b, merge_fc2_w, merge_fc2_b;
};

struct ModelWeights {
    BMat embed, lm_head, final_norm;
    std::vector<LayerWeights> layers;
    HyperConn mixer;  // qwen4: folds the streams back into one before lm_head (replaces final_norm)
    Ple ple;
    VisionWeights vis;  // empty unless cfg.vis.depth
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
