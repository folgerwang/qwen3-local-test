// Qwen3.8-Flash-Next ("qwen4exp" GGUF architecture): config and weight views into the mmapped GGUF.
//
// Differences from the Qwen3.5/3.6 MoE models (model.h):
//  - hyper-connections: `hc` parallel residual streams, mixed down to one block input by a low-rank gate
//    and written back with per-stream injection weights; they replace the layer norms and the final norm
//  - QSA: full-attention layers attend only to the top blocks of `compress` tokens picked by an indexer
//    (plus the incomplete tail block) once the context exceeds the token budget
//  - PLE: hashed n-gram embeddings injected into the residual streams in front of one layer
//  - weights are ggml-quantized (Q4_K / Q5_K / Q5_1 / Q8_0 / IQ4_NL ...) instead of FP8
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf.h"
#include "quant.h"

struct Q4Config {
    int hidden = 0, n_layers = 0, vocab = 0;
    float rms_eps = 1e-6f;
    // full attention
    int n_heads = 0, n_kv_heads = 0, head_dim = 0, rope_dim = 0;
    float rope_theta = 0;
    // QSA indexer
    int idx_heads = 0, idx_dim = 0, idx_budget = 0;
    std::vector<int> compress;  // per layer, 0 = dense attention
    // gated deltanet
    int lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, conv_kernel = 0;
    // moe
    int n_experts = 0, top_k = 0, moe_inter = 0, shared_inter = 0;
    // hyper-connections
    int hc = 0, hc_rank = 0;
    // PLE n-gram embeddings
    int ple_ngram = 0, ple_per_gram = 0, ple_kernel = 0, ple_dim = 0, ple_eos = 0;
    std::vector<uint64_t> ple_mult, ple_offsets, ple_sizes;
    std::vector<bool> is_ple;        // per layer
    std::vector<bool> is_full_attn;  // per layer
    std::vector<int> eos_ids;

    int ple_heads() const { return (ple_ngram - 1) * ple_per_gram; }
    int lin_key_dim() const { return lin_k_heads * lin_k_dim; }
    int lin_value_dim() const { return lin_v_heads * lin_v_dim; }
    int conv_dim() const { return 2 * lin_key_dim() + lin_value_dim(); }
    int hc_dim() const { return hc * hidden; }

    static Q4Config load(const Gguf& g);
};

// One hyper-connection mixer. All norm gammas in the GGUF are already folded to (1 + w).
struct Q4Hc {
    std::vector<float> norm;  // [hc * hidden]
    gq::GMat down;            // [rank, hc * hidden]
    gq::GMat up;              // [hc * hidden, rank]
    gq::GMat inject;          // [hc, hc * hidden]; absent (data == nullptr) for the output head
};

struct Q4Layer {
    bool full_attn = false, ple = false;
    int compress = 0;
    Q4Hc hc_attn, hc_ffn;
    // full attention (+ QSA indexer)
    gq::GMat q, k, v, o, idx_q, idx_k;
    std::vector<float> q_norm, k_norm, idx_q_norm, idx_k_norm;
    // gated deltanet
    gq::GMat qkv, z, alpha, beta, out;
    std::vector<float> conv1d, dt_bias, ssm_a, ssm_norm;
    // PLE
    gq::GMat ple_key, ple_value;
    std::vector<float> ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
    // moe
    gq::GMat router, sh_gate, sh_up, sh_down;
    std::vector<float> shared_gate;
    gq::GMat ex_gate, ex_up, ex_down;  // stacks of n_experts matrices, see GMat::slice
};

struct Q4Weights {
    gq::GMat embed, output;
    gq::GMat ple_table;  // [rows, ple_dim]
    Q4Hc hc_head;
    std::vector<Q4Layer> layers;
    void load(const Gguf& g, const Q4Config& cfg);
};

// PLE embedding of position p: the rows of the n-gram table picked by hashing toks[p] with its predecessors
// (an EOS or the sequence start cuts the window), concatenated over the heads into out[hidden].
void q4_ple_embed(const Q4Config& c, const Q4Weights& w, const int* toks, int p, float* out);

std::unique_ptr<Backend> make_qwen4_cpu_backend(const Q4Config& cfg, const Q4Weights& w, int max_ctx, int threads);
std::unique_ptr<Backend> make_qwen4_vulkan_backend(const Q4Config& cfg, const Q4Weights& w, int max_ctx, const VulkanOptions& opt);
