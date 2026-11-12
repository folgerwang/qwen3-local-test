# Reference logits for a (small) qwen4exp GGUF from HF transformers (fp32, CPU), with the GGUF tensors dequantized
# by the numpy code below and mapped back to the reference parameters. Test oracle only; meant for the model
# written by make_tiny_gguf.py (the real one does not fit in memory as fp32).
# usage: python ref_hf_gguf.py MODEL.gguf id,id,id,...
import struct, sys
import numpy as np, torch
from transformers.models.qwen4_exp import modeling_qwen4_exp as M, configuration_qwen4_exp as C

# ---- GGUF reader -------------------------------------------------------------------------------------------
def read_gguf(path):
    d = open(path, "rb").read()
    p = [4]
    def rd(fmt):
        v = struct.unpack_from("<" + fmt, d, p[0]); p[0] += struct.calcsize("<" + fmt); return v[0]
    def rs():
        n = rd("Q"); s = d[p[0]:p[0] + n].decode("utf-8"); p[0] += n; return s
    SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def rv(t):
        if t == 8: return rs()
        if t == 9:
            et, n = rd("I"), rd("Q"); return [rv(et) for _ in range(n)]
        return rd(SC[t])
    assert d[:4] == b"GGUF"
    rd("I"); nt, nkv = rd("Q"), rd("Q")
    kv = {}
    for _ in range(nkv):
        k = rs(); kv[k] = rv(rd("I"))
    infos = []
    for _ in range(nt):
        name = rs(); dims = [rd("Q") for _ in range(rd("I"))]; infos.append((name, dims, rd("I"), rd("Q")))
    base = (p[0] + 31) // 32 * 32
    return kv, {name: (dims, ty, base + off) for name, dims, ty, off in infos}, d

# ---- dequantization (after ggml-quants.c) --------------------------------------------------------------------
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
def h(a): return a.copy().view(np.float16).astype(np.float32)

def scale_min_k4(sc):  # [n, 12] bytes -> 8 scales, 8 mins
    s, m = np.zeros((len(sc), 8), np.float32), np.zeros((len(sc), 8), np.float32)
    for j in range(8):
        if j < 4:
            s[:, j] = sc[:, j] & 63; m[:, j] = sc[:, j + 4] & 63
        else:
            s[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            m[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return s, m

def dequant(raw, ty, n):
    if ty == 0: return np.frombuffer(raw, np.float32, n).copy()
    if ty == 30: return (np.frombuffer(raw, np.uint16, n).astype(np.uint32) << 16).view(np.float32)
    if ty == 8:
        b = np.frombuffer(raw, np.uint8, n // 32 * 34).reshape(-1, 34)
        return (h(b[:, :2]) * b[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)
    if ty == 20:
        b = np.frombuffer(raw, np.uint8, n // 32 * 18).reshape(-1, 18)
        q = b[:, 2:]
        return (h(b[:, :2]) * np.concatenate([KV_IQ4NL[q & 0xF], KV_IQ4NL[q >> 4]], 1)).reshape(-1)
    if ty == 7:
        b = np.frombuffer(raw, np.uint8, n // 32 * 24).reshape(-1, 24)
        d, m = h(b[:, 0:2]), h(b[:, 2:4])
        qh = b[:, 4:8].copy().view(np.uint32)                       # [nb, 1]
        j = np.arange(16, dtype=np.uint32)
        x0 = (b[:, 8:] & 0xF) | (((qh >> j) & 1) << 4)
        x1 = (b[:, 8:] >> 4) | (((qh >> (j + 16)) & 1) << 4)
        return (np.concatenate([x0, x1], 1).astype(np.float32) * d + m).reshape(-1)
    if ty in (12, 13):
        size = 144 if ty == 12 else 176
        b = np.frombuffer(raw, np.uint8, n // 256 * size).reshape(-1, size)
        d, dmin = h(b[:, 0:2]), h(b[:, 2:4])
        s, m = scale_min_k4(b[:, 4:16])
        qh = b[:, 16:48] if ty == 13 else None
        qs = b[:, (48 if ty == 13 else 16):].reshape(-1, 4, 32)
        out = np.zeros((len(b), 8, 32), np.float32)
        for g in range(4):
            lo, hi = (qs[:, g] & 0xF).astype(np.float32), (qs[:, g] >> 4).astype(np.float32)
            if qh is not None:
                lo += ((qh >> (2 * g)) & 1) * 16.0
                hi += ((qh >> (2 * g + 1)) & 1) * 16.0
            out[:, 2 * g] = d * s[:, 2 * g:2 * g + 1] * lo - dmin * m[:, 2 * g:2 * g + 1]
            out[:, 2 * g + 1] = d * s[:, 2 * g + 1:2 * g + 2] * hi - dmin * m[:, 2 * g + 1:2 * g + 2]
        return out.reshape(-1)
    raise ValueError(f"unsupported ggml type {ty}")

BPW = {0: 32, 30: 16, 8: 34 * 8 / 32, 20: 18 * 8 / 32, 7: 24 * 8 / 32, 12: 144 * 8 / 256, 13: 176 * 8 / 256}

def main():
    path, ids = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    kv, tens, data = read_gguf(path)
    A = "qwen4exp."
    def get(name):  # torch tensor in the reference's (reversed) dim order
        dims, ty, off = tens[name]
        n = int(np.prod(dims))
        return torch.from_numpy(dequant(data[off:off + int(n * BPW[ty] / 8)], ty, n).reshape(dims[::-1]).copy())

    L, H = kv[A + "block_count"], kv[A + "embedding_length"]
    interval = kv[A + "full_attention_interval"]
    types = ["full_attention" if (l + 1) % interval == 0 else "linear_attention" for l in range(L)]
    cfg = C.Qwen4ExpTextConfig(
        vocab_size=len(kv["tokenizer.ggml.tokens"]), hidden_size=H, num_hidden_layers=L,
        num_attention_heads=kv[A + "attention.head_count"], num_key_value_heads=kv[A + "attention.head_count_kv"],
        head_dim=kv[A + "attention.key_length"], hc_count=kv[A + "hyper_connection.count"],
        hc_lowrank=kv[A + "hyper_connection.low_rank"], indexer_budget=kv[A + "attention.indexer.top_k"],
        indexer_compress_ratio=max(kv[A + "attention.compress_ratios"]), indexer_head_dim=kv[A + "attention.indexer.key_length"],
        indexer_kv_heads=1, indexer_n_heads=kv[A + "attention.indexer.head_count"],
        linear_conv_kernel_dim=kv[A + "ssm.conv_kernel"], linear_key_head_dim=kv[A + "ssm.state_size"],
        linear_value_head_dim=kv[A + "ssm.state_size"], linear_num_key_heads=kv[A + "ssm.group_count"],
        linear_num_value_heads=kv[A + "ssm.time_step_rank"], moe_intermediate_size=kv[A + "expert_feed_forward_length"],
        shared_expert_intermediate_size=kv[A + "expert_shared_feed_forward_length"], num_experts=kv[A + "expert_count"],
        num_experts_per_tok=kv[A + "expert_used_count"], ngram_size=kv[A + "ple.ngram_size"],
        heads_per_ngram=kv[A + "ple.heads_per_ngram"], ngram_vocab_size_base=kv[A + "ple.head_vocab_sizes"][0] - 9,
        ple_conv_kernel_size=kv[A + "ple.conv_kernel"], ple_embed_dim=H, ple_layer_ids=[x + 1 for x in kv[A + "ple.layers"]],
        eos_token_id=kv[A + "ple.eos_token_id"], bos_token_id=kv[A + "ple.eos_token_id"], pad_token_id=kv[A + "ple.eos_token_id"],
        output_gate_type="sigmoid", rms_norm_eps=kv[A + "attention.layer_norm_rms_epsilon"],
        rope_parameters={"rope_type": "default", "rope_theta": kv[A + "rope.freq_base"],
                         "partial_rotary_factor": kv[A + "rope.dimension_count"] / kv[A + "attention.key_length"],
                         "mrope_section": kv[A + "rope.dimension_sections"][:3], "mrope_interleaved": True},
        layer_types=types)
    cfg._attn_implementation = "eager"
    m = M.Qwen4ExpForCausalLM(cfg).float().eval()
    params = dict(m.named_parameters())
    done = set()
    def put(name, t):
        assert params[name].shape == t.shape, (name, tuple(params[name].shape), tuple(t.shape))
        params[name].data.copy_(t); done.add(name)
    def hc(dst, src, inject=True):  # the GGUF norm gammas are folded to (1 + w)
        put(dst + "hc_norm.weight", get(src + "norm.weight") - 1)
        put(dst + "input_mix_weight_down.weight", get(src + "down.weight"))
        put(dst + "input_mix_weight_up.weight", get(src + "up.weight"))
        if inject: put(dst + "block_inject_weight.weight", get(src + "inject.weight"))

    put("model.embed_tokens.weight", get("token_embd.weight"))
    put("lm_head.weight", get("output.weight"))
    hc("model.hyper_connection_mixer.", "output_hc_", inject=False)
    for l in range(L):
        b, p = f"blk.{l}.", f"model.layers.{l}."
        hc(p + "attn_hyper_connection.", b + "hc_attn_")
        hc(p + "mlp_hyper_connection.", b + "hc_ffn_")
        if types[l] == "full_attention":
            a = p + "self_attn."
            for d_, s_ in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                put(a + d_ + ".weight", get(b + s_ + ".weight"))
            put(a + "q_norm.weight", get(b + "attn_q_norm.weight") - 1)
            put(a + "k_norm.weight", get(b + "attn_k_norm.weight") - 1)
            put(a + "indexer.index_qk_proj.weight", torch.cat([get(b + "indexer.q_proj.weight"), get(b + "indexer.k_proj.weight")]))
            put(a + "indexer.q_layernorm.weight", get(b + "indexer.q_norm.weight") - 1)
            put(a + "indexer.k_layernorm.weight", get(b + "indexer.k_norm.weight") - 1)
        else:
            # The GGUF converter reorders the value heads from grouped (head j pairs with key head j // r) to
            # tiled (head g pairs with key head g % n_k): reference head j is GGUF head (j % r) * n_k + j // r.
            a = p + "linear_attn."
            NK, NV, SD = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim
            hp = torch.tensor([(j % (NV // NK)) * NK + j // (NV // NK) for j in range(NV)])
            ep = (hp[:, None] * SD + torch.arange(SD)[None, :]).reshape(-1)  # the same permutation per element
            kd = 2 * NK * cfg.linear_key_head_dim
            qkv, conv = get(b + "attn_qkv.weight"), get(b + "ssm_conv1d.weight")
            put(a + "in_proj_qkv.weight", torch.cat([qkv[:kd], qkv[kd:][ep]]))
            put(a + "conv1d.weight", torch.cat([conv[:kd], conv[kd:][ep]]).unsqueeze(1))
            put(a + "in_proj_z.weight", get(b + "attn_gate.weight")[ep])
            put(a + "in_proj_a.weight", get(b + "ssm_alpha.weight")[hp])
            put(a + "in_proj_b.weight", get(b + "ssm_beta.weight")[hp])
            put(a + "dt_bias", get(b + "ssm_dt.bias")[hp])
            put(a + "A_log", torch.log(-get(b + "ssm_a"))[hp])  # the GGUF stores -exp(A_log)
            put(a + "norm.weight", get(b + "ssm_norm.weight"))
            put(a + "out_proj.weight", get(b + "ssm_out.weight")[:, ep])
        if (b + "ple_key.weight") in tens:
            a = p + "ple."
            put(a + "key_proj.weight", get(b + "ple_key.weight"))
            put(a + "value_proj.weight", get(b + "ple_value.weight"))
            for n_ in ("key", "query", "conv"):
                put(a + f"norm_{n_}.weight", get(b + f"ple_norm_{n_}.weight") - 1)
            put(a + "conv1d.weight", get(b + "ple_conv1d.weight").unsqueeze(1))
            put(a + "ple_embedding.ngram_embedding.weight", get("per_layer_token_embd.weight"))
            emb = m.model.layers[l].ple.ple_embedding
            emb.layer_multipliers.copy_(torch.tensor(kv[A + "ple.layer_multipliers"], dtype=torch.long))
            assert emb.ngram_heads_vocab_sizes.tolist() == kv[A + "ple.head_vocab_sizes"]
            assert emb.ngram_heads_offsets.tolist() == kv[A + "ple.head_offsets"]
        a = p + "mlp."
        put(a + "gate.weight", get(b + "ffn_gate_inp.weight"))
        put(a + "shared_expert_gate.weight", get(b + "ffn_gate_inp_shexp.weight").unsqueeze(0))
        put(a + "shared_expert.gate_proj.weight", get(b + "ffn_gate_shexp.weight"))
        put(a + "shared_expert.up_proj.weight", get(b + "ffn_up_shexp.weight"))
        put(a + "shared_expert.down_proj.weight", get(b + "ffn_down_shexp.weight"))
        put(a + "experts.gate_up_proj", torch.cat([get(b + "ffn_gate_exps.weight"), get(b + "ffn_up_exps.weight")], 1))
        put(a + "experts.down_proj", get(b + "ffn_down_exps.weight"))
    missing = [n for n in params if n not in done]
    assert not missing, missing[:8]
    with torch.no_grad():
        logits = m(torch.tensor([ids])).logits[0, -1]
    top = torch.topk(logits, 10)
    for i, v in zip(top.indices.tolist(), top.values.tolist()): print(i, f"{v:.4f}")

main()
