#include "qwen4.h"

#include <algorithm>
#include <stdexcept>

Q4Config Q4Config::load(const Gguf& g) {
    const std::string arch = g.kv_str("general.architecture");
    if (arch != "qwen4exp") throw std::runtime_error("unsupported GGUF architecture: " + arch);
    const std::string p = arch + ".";
    auto u = [&](const std::string& k) { return (int)g.kv_u64(p + k); };

    Q4Config c;
    c.hidden = u("embedding_length");
    c.n_layers = u("block_count");
    c.vocab = (int)g.kv("tokenizer.ggml.tokens").strs.size();
    c.rms_eps = (float)g.kv_num(p + "attention.layer_norm_rms_epsilon", 1e-6);
    c.n_heads = u("attention.head_count");
    c.n_kv_heads = u("attention.head_count_kv");
    c.head_dim = u("attention.key_length");
    c.rope_dim = u("rope.dimension_count");
    c.rope_theta = (float)g.kv_num(p + "rope.freq_base", 1e7);
    c.idx_heads = u("attention.indexer.head_count");
    c.idx_dim = u("attention.indexer.key_length");
    c.idx_budget = u("attention.indexer.top_k");
    c.lin_k_heads = u("ssm.group_count");
    c.lin_v_heads = u("ssm.time_step_rank");
    c.lin_k_dim = c.lin_v_dim = u("ssm.state_size");
    c.conv_kernel = u("ssm.conv_kernel");
    c.n_experts = u("expert_count");
    c.top_k = u("expert_used_count");
    c.moe_inter = u("expert_feed_forward_length");
    c.shared_inter = u("expert_shared_feed_forward_length");
    c.hc = u("hyper_connection.count");
    c.hc_rank = u("hyper_connection.low_rank");
    if (c.n_layers > 256 || c.n_experts > 1024 || c.top_k > 16 || c.hc < 2 || c.hc > 8 || c.conv_kernel > 8)
        throw std::runtime_error("unsupported model dimensions");
    if (c.rope_dim > c.head_dim || c.rope_dim > c.idx_dim || c.rope_dim % 2) throw std::runtime_error("unsupported rope dimension");
    if (c.lin_v_heads % c.lin_k_heads) throw std::runtime_error("unsupported deltanet head counts");
    if (c.n_heads % c.n_kv_heads) throw std::runtime_error("unsupported attention head counts");

    const int interval = (int)g.kv_num(p + "full_attention_interval", 4);
    for (int l = 0; l < c.n_layers; l++) c.is_full_attn.push_back((l + 1) % interval == 0);

    c.compress.assign(c.n_layers, 0);
    if (const GValue* cr = g.find_kv(p + "attention.compress_ratios")) {
        if (cr->kind == GValue::NumArr) {
            for (int l = 0; l < c.n_layers && l < (int)cr->ints.size(); l++) c.compress[l] = (int)cr->ints[l];
        } else {
            c.compress.assign(c.n_layers, (int)cr->u64);
        }
    }
    for (int l = 0; l < c.n_layers; l++) {
        if (!c.is_full_attn[l]) c.compress[l] = 0;
        if (c.compress[l] == 1 || (c.compress[l] > 1 && c.idx_budget % c.compress[l]))
            throw std::runtime_error("unsupported QSA compress ratio");
    }

    c.is_ple.assign(c.n_layers, false);
    if (const GValue* pl = g.find_kv(p + "ple.layers"); pl && !pl->ints.empty()) {
        if (pl->ints.size() != 1) throw std::runtime_error("only one PLE layer is supported");
        if (pl->ints[0] >= (uint64_t)c.n_layers) throw std::runtime_error("PLE layer out of range");
        c.is_ple[(size_t)pl->ints[0]] = true;
        c.ple_ngram = u("ple.ngram_size");
        c.ple_per_gram = u("ple.heads_per_ngram");
        c.ple_kernel = u("ple.conv_kernel");
        c.ple_eos = u("ple.eos_token_id");
        c.ple_dim = u("embedding_length_per_layer_input");
        c.ple_mult = g.kv(p + "ple.layer_multipliers").ints;
        c.ple_offsets = g.kv(p + "ple.head_offsets").ints;
        c.ple_sizes = g.kv(p + "ple.head_vocab_sizes").ints;
        const size_t nh = (size_t)c.ple_heads();
        if (c.ple_ngram < 2 || c.ple_ngram > 8 || c.ple_kernel < 1 || c.ple_kernel > 8 || (int)c.ple_mult.size() < c.ple_ngram ||
            c.ple_offsets.size() < nh || c.ple_sizes.size() < nh || c.ple_dim * (int)nh != c.hidden)
            throw std::runtime_error("unsupported PLE configuration");
        for (size_t h = 0; h < nh; h++)
            if (c.ple_sizes[h] == 0) throw std::runtime_error("bad PLE head size");
    }

    // <|im_end|> ends a turn; <|endoftext|> (the pad/bos id) ends the sequence.
    c.eos_ids.push_back((int)g.kv_u64("tokenizer.ggml.eos_token_id"));
    for (const char* k : {"tokenizer.ggml.padding_token_id", "tokenizer.ggml.bos_token_id"})
        if (const GValue* v = g.find_kv(k); v && (int)v->u64 != c.eos_ids[0] && (c.eos_ids.size() < 2 || c.eos_ids[1] != (int)v->u64))
            c.eos_ids.push_back((int)v->u64);
    return c;
}

namespace {

struct Loader {
    const Gguf& g;
    const Q4Config& c;
    gq::GMat mat(const std::string& n, int rows, int cols, int count = 1) const { return gq::make_mat(g.get(n), n, rows, cols, count); }
    std::vector<float> vec(const std::string& n, size_t len) const { return gq::to_floats(g.get(n), n, len); }
    Q4Hc hc(const std::string& pre, bool inject) const {
        Q4Hc h;
        h.norm = vec(pre + "norm.weight", c.hc_dim());
        h.down = mat(pre + "down.weight", c.hc_rank, c.hc_dim());
        h.up = mat(pre + "up.weight", c.hc_dim(), c.hc_rank);
        if (inject) h.inject = mat(pre + "inject.weight", c.hc, c.hc_dim());
        return h;
    }
    Q4Layer layer(int l, bool full_attn, int compress, bool ple) const {
        const int H = c.hidden, HC = c.hc_dim();
        Q4Layer L;
        const std::string b = "blk." + std::to_string(l) + ".";
        L.full_attn = full_attn;
        L.compress = compress;
        L.hc_attn = hc(b + "hc_attn_", true);
        L.hc_ffn = hc(b + "hc_ffn_", true);
        if (L.full_attn) {
            const int AD = c.n_heads * c.head_dim, KV = c.n_kv_heads * c.head_dim;
            L.q = mat(b + "attn_q.weight", AD * 2, H);
            L.k = mat(b + "attn_k.weight", KV, H);
            L.v = mat(b + "attn_v.weight", KV, H);
            L.o = mat(b + "attn_output.weight", H, AD);
            L.q_norm = vec(b + "attn_q_norm.weight", c.head_dim);
            L.k_norm = vec(b + "attn_k_norm.weight", c.head_dim);
            if (L.compress > 0) {
                L.idx_q = mat(b + "indexer.q_proj.weight", c.idx_heads * c.idx_dim, H);
                L.idx_k = mat(b + "indexer.k_proj.weight", c.idx_dim, H);
                L.idx_q_norm = vec(b + "indexer.q_norm.weight", c.idx_dim);
                L.idx_k_norm = vec(b + "indexer.k_norm.weight", c.idx_dim);
            }
        } else {
            const int VD = c.lin_value_dim(), NV = c.lin_v_heads;
            L.qkv = mat(b + "attn_qkv.weight", c.conv_dim(), H);
            L.z = mat(b + "attn_gate.weight", VD, H);
            L.alpha = mat(b + "ssm_alpha.weight", NV, H);
            L.beta = mat(b + "ssm_beta.weight", NV, H);
            L.out = mat(b + "ssm_out.weight", H, VD);
            L.conv1d = vec(b + "ssm_conv1d.weight", (size_t)c.conv_dim() * c.conv_kernel);
            L.dt_bias = vec(b + "ssm_dt.bias", NV);
            L.ssm_a = vec(b + "ssm_a", NV);
            L.ssm_norm = vec(b + "ssm_norm.weight", c.lin_v_dim);
        }
        if (ple) {
            L.ple = true;
            L.ple_key = mat(b + "ple_key.weight", HC, H);
            L.ple_value = mat(b + "ple_value.weight", H, H);
            L.ple_norm_key = vec(b + "ple_norm_key.weight", HC);
            L.ple_norm_query = vec(b + "ple_norm_query.weight", HC);
            L.ple_norm_conv = vec(b + "ple_norm_conv.weight", HC);
            L.ple_conv1d = vec(b + "ple_conv1d.weight", (size_t)HC * c.ple_kernel);
        }
        L.router = mat(b + "ffn_gate_inp.weight", c.n_experts, H);
        L.shared_gate = vec(b + "ffn_gate_inp_shexp.weight", H);
        L.sh_gate = mat(b + "ffn_gate_shexp.weight", c.shared_inter, H);
        L.sh_up = mat(b + "ffn_up_shexp.weight", c.shared_inter, H);
        L.sh_down = mat(b + "ffn_down_shexp.weight", H, c.shared_inter);
        L.ex_gate = mat(b + "ffn_gate_exps.weight", c.moe_inter, H, c.n_experts);
        L.ex_up = mat(b + "ffn_up_exps.weight", c.moe_inter, H, c.n_experts);
        L.ex_down = mat(b + "ffn_down_exps.weight", H, c.moe_inter, c.n_experts);
        return L;
    }
};

}  // namespace

void Q4Weights::load(const Gguf& g, const Q4Config& c) {
    const Loader ld{g, c};
    const int H = c.hidden;
    embed = ld.mat("token_embd.weight", c.vocab, H);
    output = g.find("output.weight") ? ld.mat("output.weight", c.vocab, H) : embed;
    hc_head = ld.hc("output_hc_", false);

    bool any_ple = false;
    for (int l = 0; l < c.n_layers; l++) {
        layers.push_back(ld.layer(l, c.is_full_attn[l], c.compress[l], c.is_ple[l]));
        any_ple |= c.is_ple[l];
    }

    if (any_ple) {
        const std::string n = "per_layer_token_embd.weight";
        const GTensor& t = g.get(n);
        uint64_t need = 0;
        for (int h = 0; h < c.ple_heads(); h++) need = std::max(need, c.ple_offsets[h] + c.ple_sizes[h]);
        if (t.dims.size() != 2 || t.dims[0] != c.ple_dim || (uint64_t)t.dims[1] < need || gtype_block(t.type) == 0)
            throw std::runtime_error("bad PLE table: " + n);
        ple_table.type = t.type;
        ple_table.data = t.data;
        ple_table.cols = c.ple_dim;
        ple_table.rows = 0;  // too many rows for an int; index with row()
        ple_table.row_bytes = (size_t)(c.ple_dim / gtype_block(t.type)) * gtype_block_bytes(t.type);
    }
}

void Q4Weights::load_mtp(const Gguf& g, const Q4Config& c) {
    const std::string arch = g.kv_str("general.architecture");
    if (arch != "qwen4exp" || (int)g.kv_u64(arch + ".embedding_length") != c.hidden ||
        (int)g.kv_u64(arch + ".expert_count") != c.n_experts)
        throw std::runtime_error("the MTP draft head does not belong to this model");
    const Loader ld{g, c};
    const int l = c.n_layers, H = c.hidden, HC = c.hc_dim();
    int compress = 0;  // the MTP block is one more QSA layer with the trunk's block size
    for (int v : c.compress) compress = std::max(compress, v);
    layers.push_back(ld.layer(l, true, compress, false));
    const std::string b = "blk." + std::to_string(l) + ".nextn.";
    mtp_eh = ld.mat(b + "eh_proj.weight", H, 2 * H);
    mtp_enorm = ld.vec(b + "enorm.weight", H);
    mtp_hnorm = ld.vec(b + "hnorm.weight", HC);
    mtp_head = ld.hc(b + "hc_head_", false);
    has_mtp = true;
}

void q4_ple_embed(const Q4Config& c, const Q4Weights& w, const int* toks, int p, float* out) {
    // mixed_n = (t[p] * m[0]) ^ ... ^ (t[p-n+1] * m[n-1]) in wrapping 64-bit; row = mixed_n % size[h] + offset[h]
    uint64_t mixed = (uint64_t)toks[p] * c.ple_mult[0];
    bool cut = false;
    for (int g = 2; g <= c.ple_ngram; g++) {
        const int s = g - 1;
        cut = cut || p - s < 0 || toks[p - s] == c.ple_eos;
        mixed ^= (uint64_t)(cut ? c.ple_eos : toks[p - s]) * c.ple_mult[s];
        for (int k = 0; k < c.ple_per_gram; k++) {
            const int h = (g - 2) * c.ple_per_gram + k;
            const uint64_t row = mixed % c.ple_sizes[h] + c.ple_offsets[h];
            gq::dequant_row(w.ple_table.type, w.ple_table.row((int64_t)row), c.ple_dim, out + (size_t)h * c.ple_dim);
        }
    }
}
