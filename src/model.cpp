#include "model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.h"

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

Config Config::load(const std::string& dir) {
    json::Value root = json::parse(read_file(dir + "/config.json"));
    const json::Value& t = root.find("text_config") ? root["text_config"] : root;
    Config c;
    c.hidden = (int)t["hidden_size"].as_int();
    c.n_layers = (int)t["num_hidden_layers"].as_int();
    c.vocab = (int)t["vocab_size"].as_int();
    c.rms_eps = (float)t.get_num("rms_norm_eps", 1e-6);
    c.n_heads = (int)t["num_attention_heads"].as_int();
    c.n_kv_heads = (int)t["num_key_value_heads"].as_int();
    c.head_dim = (int)t["head_dim"].as_int();
    const json::Value* rp = t.find("rope_parameters");
    double partial = rp ? rp->get_num("partial_rotary_factor", 1.0) : t.get_num("partial_rotary_factor", 1.0);
    c.rope_theta = (float)(rp ? rp->get_num("rope_theta", 1e7) : t.get_num("rope_theta", 1e7));
    c.rope_dim = (int)(c.head_dim * partial);
    c.lin_k_heads = (int)t["linear_num_key_heads"].as_int();
    c.lin_v_heads = (int)t["linear_num_value_heads"].as_int();
    c.lin_k_dim = (int)t["linear_key_head_dim"].as_int();
    c.lin_v_dim = (int)t["linear_value_head_dim"].as_int();
    c.conv_kernel = (int)t["linear_conv_kernel_dim"].as_int();
    c.n_experts = (int)t["num_experts"].as_int();
    c.top_k = (int)t["num_experts_per_tok"].as_int();
    if (c.n_experts > 1024 || c.top_k > 16) throw std::runtime_error("unsupported expert count");
    c.moe_inter = (int)t["moe_intermediate_size"].as_int();
    c.shared_inter = (int)t["shared_expert_intermediate_size"].as_int();
    if (const json::Value* og = t.find("output_gate_type"); og && og->type == json::Value::String && og->as_str() != "sigmoid")
        throw std::runtime_error("unsupported output_gate_type: " + og->as_str());
    for (auto& lt : t["layer_types"].arr) c.is_full_attn.push_back(lt.as_str() == "full_attention");
    if ((int)c.is_full_attn.size() != c.n_layers) throw std::runtime_error("layer_types size mismatch");

    if (const json::Value* mt = t.find("model_type"); mt && mt->as_str() == "qwen4_exp_text") {
        c.qwen4 = true;
        c.hc = (int)t["hc_count"].as_int();
        c.hc_rank = (int)t["hc_lowrank"].as_int();
        if (const json::Value* pl = t.find("ple_layer_ids"); pl && pl->type == json::Value::Array && !pl->arr.empty()) {
            if (pl->arr.size() != 1) throw std::runtime_error("only one PLE layer is supported");
            c.ple_layer = (int)pl->arr[0].as_int() - 1;  // config ids are 1-based
            c.ngram = (int)t["ngram_size"].as_int();
            c.ngram_heads = (c.ngram - 1) * (int)t["heads_per_ngram"].as_int();
            c.ple_dim = (int)t["ple_embed_dim"].as_int();
            c.ple_conv = (int)t["ple_conv_kernel_size"].as_int();
            const json::Value& e = t["eos_token_id"];
            c.ple_eos = (int)(e.type == json::Value::Array ? e.arr[0].as_int() : e.as_int());
            if (c.ngram > 4 || c.ple_dim % c.ngram_heads) throw std::runtime_error("unsupported PLE config");
        }
        c.idx_heads = (int)t["indexer_n_heads"].as_int();
        c.idx_dim = (int)t["indexer_head_dim"].as_int();
        c.idx_budget = (int)t["indexer_budget"].as_int();
        c.idx_ratio = (int)t["indexer_compress_ratio"].as_int();
        if (t["indexer_kv_heads"].as_int() != 1) throw std::runtime_error("unsupported indexer_kv_heads");
    }

    // vision tower (not wired up for qwen4_exp yet)
    if (const json::Value* v = root.find("vision_config"); v && !c.qwen4) {
        Config::Vision& vc = c.vis;
        vc.depth = (int)v->get_int("depth", 0);
        vc.hidden = (int)(*v)["hidden_size"].as_int();
        vc.heads = (int)(*v)["num_heads"].as_int();
        vc.inter = (int)(*v)["intermediate_size"].as_int();
        vc.patch = (int)(*v)["patch_size"].as_int();
        vc.temporal = (int)v->get_int("temporal_patch_size", 2);
        vc.merge = (int)v->get_int("spatial_merge_size", 2);
        vc.n_pos = (int)(*v)["num_position_embeddings"].as_int();
        vc.out = (int)(*v)["out_hidden_size"].as_int();
        if (const json::Value* rp = v->find("rope_parameters")) vc.rope_theta = (float)rp->get_num("rope_theta", 10000.0);
        vc.image_token = (int)root.get_int("image_token_id", -1);
        vc.vision_start = (int)root.get_int("vision_start_token_id", -1);
        vc.vision_end = (int)root.get_int("vision_end_token_id", -1);
        const json::Value* ds = v->find("deepstack_visual_indexes");
        const int hd = vc.heads ? vc.hidden / vc.heads : 0, side = (int)std::lround(std::sqrt((double)vc.n_pos));
        // image tokens use interleaved M-RoPE; the shader maps rotary frequency i to position component i % 3,
        // which is exactly the [11, 11, 10] interleaved split of the 32 frequencies
        bool mrope_ok = false;
        if (rp && rp->find("mrope_section")) {
            const auto& s = (*rp)["mrope_section"].arr;
            const json::Value* il = rp->find("mrope_interleaved");
            mrope_ok = il && il->b && s.size() == 3 && s[0].as_int() == 11 && s[1].as_int() == 11 && s[2].as_int() == 10 &&
                       c.rope_dim == 64;
        }
        if (vc.out != c.hidden || (ds && !ds->arr.empty()) || vc.image_token < 0 || hd != 72 || vc.hidden % 64 ||
            vc.inter % 16 || side * side != vc.n_pos || vc.merge != 2 || !mrope_ok)
            vc.depth = 0;  // a layout this engine does not implement: text only
        try {
            json::Value pp = json::parse(read_file(dir + "/preprocessor_config.json"));
            if (const json::Value* m = pp.find("image_mean"))
                for (int i = 0; i < 3; i++) vc.mean[i] = (float)m->arr.at(i).as_num();
            if (const json::Value* s = pp.find("image_std"))
                for (int i = 0; i < 3; i++) vc.std[i] = (float)s->arr.at(i).as_num();
            if (const json::Value* sz = pp.find("size")) vc.min_pixels = sz->get_int("shortest_edge", vc.min_pixels);
        } catch (...) {
        }
    }

    // eos ids: generation_config.json has the full list (<|im_end|>, <|endoftext|>).
    try {
        json::Value g = json::parse(read_file(dir + "/generation_config.json"));
        const json::Value& e = g["eos_token_id"];
        if (e.type == json::Value::Array) for (auto& v : e.arr) c.eos_ids.push_back((int)v.as_int());
        else c.eos_ids.push_back((int)e.as_int());
    } catch (...) {
        c.eos_ids.push_back((int)t["eos_token_id"].as_int());
    }
    return c;
}

static void check_shape(const TensorView& t, const std::string& name, std::vector<int64_t> shape) {
    if (t.shape != shape) {
        std::string s = "shape mismatch for " + name + ": got [";
        for (auto d : t.shape) s += std::to_string(d) + ",";
        throw std::runtime_error(s + "]");
    }
}

// BF16 tensor, or a small F32 one (converted to an owned float copy).
static BMat bmat(const SafeTensors& st, const std::string& name, int rows, int cols) {
    const TensorView& t = st.get(name);
    if ((int64_t)rows * cols != t.numel()) throw std::runtime_error("size mismatch: " + name);
    BMat m;
    m.rows = rows;
    m.cols = cols;
    if (t.dtype == DType::BF16) {
        m.w = (const uint16_t*)t.data;
    } else if (t.dtype == DType::F32 && t.numel() <= (1 << 20)) {
        m.f32.resize(t.numel());
        std::memcpy(m.f32.data(), t.data, t.numel() * 4);
    } else {
        throw std::runtime_error("expected BF16: " + name);
    }
    return m;
}

static QMat qmat(const SafeTensors& st, const std::string& name, int rows, int cols) {
    const TensorView& t = st.get(name + ".weight");
    const TensorView& s = st.get(name + ".weight_scale_inv");
    if (t.dtype != DType::F8_E4M3) throw std::runtime_error("expected F8_E4M3: " + name);
    if (s.dtype != DType::BF16) throw std::runtime_error("expected BF16 scale: " + name);
    if (rows % 128 || cols % 128) throw std::runtime_error("dims not multiple of 128: " + name);
    check_shape(t, name, {rows, cols});
    check_shape(s, name + ".weight_scale_inv", {rows / 128, cols / 128});
    return QMat{t.data, (const uint16_t*)s.data, rows, cols};
}

// FP8 block-scaled or BF16 projection, whichever the checkpoint has.
static Lin lin(const SafeTensors& st, const std::string& name, int rows, int cols) {
    Lin m;
    if (st.find(name + ".weight_scale_inv")) m.q = qmat(st, name, rows, cols);
    else m.b = bmat(st, name + ".weight", rows, cols);
    return m;
}

static HyperConn hyper(const SafeTensors& st, const std::string& p, const Config& c, bool inject) {
    const int HH = c.hc * c.hidden;
    HyperConn h;
    h.norm = bmat(st, p + "hc_norm.weight", 1, HH);
    h.down = bmat(st, p + "input_mix_weight_down.weight", c.hc_rank, HH);
    h.up = bmat(st, p + "input_mix_weight_up.weight", HH, c.hc_rank);
    if (inject) h.inject = bmat(st, p + "block_inject_weight.weight", c.hc, HH);
    return h;
}

static std::vector<int64_t> i64s(const SafeTensors& st, const std::string& name, size_t n) {
    const TensorView& t = st.get(name);
    if (t.dtype != DType::I64 || (size_t)t.numel() != n)
        throw std::runtime_error("expected I64[" + std::to_string(n) + "]: " + name);
    std::vector<int64_t> v(n);
    std::memcpy(v.data(), t.data, n * 8);
    return v;
}

static Ple load_ple(const SafeTensors& st, const std::string& p, const Config& c) {
    const int HH = c.hc * c.hidden;
    Ple e;
    e.dim = c.ple_dim / c.ngram_heads;
    e.key_proj = bmat(st, p + "key_proj.weight", HH, c.ple_dim);
    e.value_proj = bmat(st, p + "value_proj.weight", c.hidden, c.ple_dim);
    e.norm_key = bmat(st, p + "norm_key.weight", 1, HH);
    e.norm_query = bmat(st, p + "norm_query.weight", 1, HH);
    e.norm_conv = bmat(st, p + "norm_conv.weight", 1, HH);
    e.conv = bmat(st, p + "conv1d.weight", HH, c.ple_conv);
    const std::string ep = p + "ple_embedding.";
    std::vector<int64_t> m = i64s(st, ep + "layer_multipliers", c.ngram);
    for (int i = 0; i < c.ngram; i++) e.mult[i] = m[i];
    e.offsets = i64s(st, ep + "ngram_heads_offsets", c.ngram_heads);
    e.sizes = i64s(st, ep + "ngram_heads_vocab_sizes", c.ngram_heads);
    const TensorView& sc = st.get(ep + "ngram_embedding.weight_scale");
    if (sc.dtype == DType::BF16) e.scale = bf16_to_f32(*(const uint16_t*)sc.data);
    else if (sc.dtype == DType::F32) std::memcpy(&e.scale, sc.data, 4);
    else throw std::runtime_error("unexpected ngram weight_scale dtype");
    for (int i = 0;; i++) {  // shards are concatenated along rows
        const TensorView* t = st.find(ep + "ngram_embedding.shard_" + std::to_string(i) + ".weight");
        if (!t) break;
        if (t->dtype != DType::F8_E4M3 || t->shape.size() != 2 || t->shape[1] != e.dim)
            throw std::runtime_error("unexpected ngram embedding shard " + std::to_string(i));
        if (i == 0) e.shard_rows = t->shape[0];
        else if (t->shape[0] != e.shard_rows) throw std::runtime_error("ngram embedding shards differ in size");
        e.shards.push_back(t->data);
    }
    if (e.shards.empty()) throw std::runtime_error("ngram embedding not found");
    if (e.offsets.back() + e.sizes.back() > e.shard_rows * (int64_t)e.shards.size())
        throw std::runtime_error("ngram embedding too small");
    return e;
}

void ModelWeights::load(const SafeTensors& st, const Config& c) {
    const std::string p = "model.language_model.";
    const int H = c.hidden;
    embed = bmat(st, p + "embed_tokens.weight", c.vocab, H);
    lm_head = bmat(st, "lm_head.weight", c.vocab, H);
    if (c.qwen4) {
        mixer = hyper(st, p + "hyper_connection_mixer.", c, false);
        if (c.ple_layer >= 0) ple = load_ple(st, p + "layers." + std::to_string(c.ple_layer) + ".ple.", c);
    } else {
        final_norm = bmat(st, p + "norm.weight", 1, H);
    }
    layers.resize(c.n_layers);
    for (int l = 0; l < c.n_layers; l++) {
        LayerWeights& L = layers[l];
        const std::string lp = p + "layers." + std::to_string(l) + ".";
        L.full_attn = c.is_full_attn[l];
        if (c.qwen4) {
            L.attn_hc = hyper(st, lp + "attn_hyper_connection.", c, true);
            L.mlp_hc = hyper(st, lp + "mlp_hyper_connection.", c, true);
        } else {
            L.input_norm = bmat(st, lp + "input_layernorm.weight", 1, H);
            L.post_norm = bmat(st, lp + "post_attention_layernorm.weight", 1, H);
        }
        if (L.full_attn) {
            const std::string a = lp + "self_attn.";
            L.q_proj = lin(st, a + "q_proj", c.n_heads * c.head_dim * 2, H);
            L.k_proj = lin(st, a + "k_proj", c.n_kv_heads * c.head_dim, H);
            L.v_proj = lin(st, a + "v_proj", c.n_kv_heads * c.head_dim, H);
            L.o_proj = lin(st, a + "o_proj", H, c.n_heads * c.head_dim);
            L.q_norm = bmat(st, a + "q_norm.weight", 1, c.head_dim);
            L.k_norm = bmat(st, a + "k_norm.weight", 1, c.head_dim);
            if (c.qwen4) {
                L.idx_qk = bmat(st, a + "indexer.index_qk_proj.weight", (c.idx_heads + 1) * c.idx_dim, H);
                L.idx_qn = bmat(st, a + "indexer.q_layernorm.weight", 1, c.idx_dim);
                L.idx_kn = bmat(st, a + "indexer.k_layernorm.weight", 1, c.idx_dim);
            }
        } else {
            const std::string a = lp + "linear_attn.";
            L.in_qkv = lin(st, a + "in_proj_qkv", c.conv_dim(), H);
            L.in_z = lin(st, a + "in_proj_z", c.lin_value_dim(), H);
            L.out_proj = lin(st, a + "out_proj", H, c.lin_value_dim());
            L.in_a = bmat(st, a + "in_proj_a.weight", c.lin_v_heads, H);
            L.in_b = bmat(st, a + "in_proj_b.weight", c.lin_v_heads, H);
            L.conv1d = bmat(st, a + "conv1d.weight", c.conv_dim(), c.conv_kernel);
            L.a_log = bmat(st, a + "A_log", 1, c.lin_v_heads);
            L.dt_bias = bmat(st, a + "dt_bias", 1, c.lin_v_heads);
            L.lin_norm = bmat(st, a + "norm.weight", 1, c.lin_v_dim);
        }
        const std::string m = lp + "mlp.";
        L.router = bmat(st, m + "gate.weight", c.n_experts, H);
        L.shared_gate = bmat(st, m + "shared_expert_gate.weight", 1, H);
        L.sh_gate = lin(st, m + "shared_expert.gate_proj", c.shared_inter, H);
        L.sh_up = lin(st, m + "shared_expert.up_proj", c.shared_inter, H);
        L.sh_down = lin(st, m + "shared_expert.down_proj", H, c.shared_inter);
        L.experts.resize(c.n_experts);
        for (int e = 0; e < c.n_experts; e++) {
            const std::string ep = m + "experts." + std::to_string(e) + ".";
            L.experts[e] = {qmat(st, ep + "gate_proj", c.moe_inter, H), qmat(st, ep + "up_proj", c.moe_inter, H),
                            qmat(st, ep + "down_proj", H, c.moe_inter)};
        }
    }

    // vision tower, when the config describes one and the checkpoint has it
    const Config::Vision& vc = c.vis;
    const std::string v = "model.visual.";
    if (vc.depth && st.find(v + "patch_embed.proj.weight")) {
        const int VH = vc.hidden, M4 = VH * vc.merge * vc.merge;
        vis.patch_w = bmat(st, v + "patch_embed.proj.weight", VH, vc.patch_dim());
        vis.patch_b = bmat(st, v + "patch_embed.proj.bias", 1, VH);
        vis.pos_embed = bmat(st, v + "pos_embed.weight", vc.n_pos, VH);
        vis.blocks.resize(vc.depth);
        for (int b = 0; b < vc.depth; b++) {
            const std::string bp = v + "blocks." + std::to_string(b) + ".";
            VisionBlock& B = vis.blocks[b];
            B.norm1_w = bmat(st, bp + "norm1.weight", 1, VH);
            B.norm1_b = bmat(st, bp + "norm1.bias", 1, VH);
            B.qkv_w = bmat(st, bp + "attn.qkv.weight", 3 * VH, VH);
            B.qkv_b = bmat(st, bp + "attn.qkv.bias", 1, 3 * VH);
            B.proj_w = bmat(st, bp + "attn.proj.weight", VH, VH);
            B.proj_b = bmat(st, bp + "attn.proj.bias", 1, VH);
            B.norm2_w = bmat(st, bp + "norm2.weight", 1, VH);
            B.norm2_b = bmat(st, bp + "norm2.bias", 1, VH);
            B.fc1_w = bmat(st, bp + "mlp.linear_fc1.weight", vc.inter, VH);
            B.fc1_b = bmat(st, bp + "mlp.linear_fc1.bias", 1, vc.inter);
            B.fc2_w = bmat(st, bp + "mlp.linear_fc2.weight", VH, vc.inter);
            B.fc2_b = bmat(st, bp + "mlp.linear_fc2.bias", 1, VH);
        }
        vis.merge_norm_w = bmat(st, v + "merger.norm.weight", 1, VH);
        vis.merge_norm_b = bmat(st, v + "merger.norm.bias", 1, VH);
        vis.merge_fc1_w = bmat(st, v + "merger.linear_fc1.weight", M4, M4);
        vis.merge_fc1_b = bmat(st, v + "merger.linear_fc1.bias", 1, M4);
        vis.merge_fc2_w = bmat(st, v + "merger.linear_fc2.weight", vc.out, M4);
        vis.merge_fc2_b = bmat(st, v + "merger.linear_fc2.bias", 1, vc.out);
    }
}

void route_topk(const float* logits, int n_experts, int k, int* ids, float* weights) {
    // Mirrors the reference: softmax in fp32, top-k on probabilities (ties -> lower index), renormalize.
    float mx = logits[0];
    for (int e = 1; e < n_experts; e++) mx = std::max(mx, logits[e]);
    float p[1024];
    float sum = 0;
    for (int e = 0; e < n_experts; e++) { p[e] = std::exp(logits[e] - mx); sum += p[e]; }
    for (int e = 0; e < n_experts; e++) p[e] /= sum;
    int cnt = 0;
    for (int e = 0; e < n_experts; e++) {
        const float v = p[e];
        if (cnt == k && v <= p[ids[k - 1]]) continue;
        int pos = cnt < k ? cnt++ : k - 1;
        while (pos > 0 && p[ids[pos - 1]] < v) { ids[pos] = ids[pos - 1]; pos--; }
        ids[pos] = e;
    }
    float ts = 0;
    for (int i = 0; i < k; i++) ts += p[ids[i]];
    for (int i = 0; i < k; i++) weights[i] = p[ids[i]] / ts;
}
