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

void ModelWeights::load(const SafeTensors& st, const Config& c) {
    const std::string p = "model.language_model.";
    const int H = c.hidden;
    embed = bmat(st, p + "embed_tokens.weight", c.vocab, H);
    lm_head = bmat(st, "lm_head.weight", c.vocab, H);
    final_norm = bmat(st, p + "norm.weight", 1, H);
    layers.resize(c.n_layers);
    for (int l = 0; l < c.n_layers; l++) {
        LayerWeights& L = layers[l];
        const std::string lp = p + "layers." + std::to_string(l) + ".";
        L.full_attn = c.is_full_attn[l];
        L.input_norm = bmat(st, lp + "input_layernorm.weight", 1, H);
        L.post_norm = bmat(st, lp + "post_attention_layernorm.weight", 1, H);
        if (L.full_attn) {
            const std::string a = lp + "self_attn.";
            L.q_proj = qmat(st, a + "q_proj", c.n_heads * c.head_dim * 2, H);
            L.k_proj = qmat(st, a + "k_proj", c.n_kv_heads * c.head_dim, H);
            L.v_proj = qmat(st, a + "v_proj", c.n_kv_heads * c.head_dim, H);
            L.o_proj = qmat(st, a + "o_proj", H, c.n_heads * c.head_dim);
            L.q_norm = bmat(st, a + "q_norm.weight", 1, c.head_dim);
            L.k_norm = bmat(st, a + "k_norm.weight", 1, c.head_dim);
        } else {
            const std::string a = lp + "linear_attn.";
            L.in_qkv = qmat(st, a + "in_proj_qkv", c.conv_dim(), H);
            L.in_z = qmat(st, a + "in_proj_z", c.lin_value_dim(), H);
            L.out_proj = qmat(st, a + "out_proj", H, c.lin_value_dim());
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
        L.shared = {qmat(st, m + "shared_expert.gate_proj", c.shared_inter, H),
                    qmat(st, m + "shared_expert.up_proj", c.shared_inter, H),
                    qmat(st, m + "shared_expert.down_proj", H, c.shared_inter)};
        L.experts.resize(c.n_experts);
        for (int e = 0; e < c.n_experts; e++) {
            const std::string ep = m + "experts." + std::to_string(e) + ".";
            L.experts[e] = {qmat(st, ep + "gate_proj", c.moe_inter, H), qmat(st, ep + "up_proj", c.moe_inter, H),
                            qmat(st, ep + "down_proj", H, c.moe_inter)};
        }
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
