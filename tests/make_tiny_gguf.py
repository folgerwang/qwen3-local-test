# Writes a tiny random qwen4exp GGUF (same layout and quantization mix as Qwen3.8-Flash-Next UD-Q4_K_XL) for
# fast backend tests: CPU vs Vulkan vs llama.cpp without the 111 GB model. The quantized payloads are random
# bytes with sane scales, so every codec path gets exercised; the "model" is of course gibberish.
# usage: python make_tiny_gguf.py OUT.gguf
import struct, sys
import numpy as np

rng = np.random.default_rng(1234)
H, LAYERS, VOCAB = 512, 4, 512
HEADS, KV, HD = 2, 1, 256
IDX_H, IDX_D, BUDGET, CR = 2, 128, 16, 4          # QSA selects once more than 4 blocks (16 tokens) exist
KH, VH, SD, CK = 2, 4, 128, 4                    # deltanet: key heads, value heads, state size, conv kernel
E, TOPK, FF = 16, 4, 128
HC, RANK = 4, 64
NGRAM, PER_GRAM, PLE_K, PLE_DIM = 3, 2, 4, 128
PLE_SIZES = [1009, 1013, 1019, 1021]
EOS, IM_END = 256, 258
KEY, VAL = KH * SD, VH * SD
CONV = 2 * KEY + VAL
# expert quantization per layer (gate/up, down): covers every codec and several slot sizes
EXPERT_Q = [("Q4_K", "Q5_1"), ("Q5_K", "Q8_0"), ("Q4_K", "Q8_0"), ("Q4_K", "Q5_1")]

TYPES = {"F32": 0, "Q5_1": 7, "Q8_0": 8, "Q4_K": 12, "Q5_K": 13, "IQ4_NL": 20, "BF16": 30}

def f16(a):
    return np.asarray(a, dtype=np.float16).view(np.uint8)

def blocks(kind, n):
    """n weights of random quantized payload."""
    if kind == "F32":
        return rng.normal(0, 0.05, n).astype(np.float32).tobytes()
    if kind == "BF16":
        return (rng.normal(0, 0.05, n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
    nb = n // 32
    d = rng.uniform(0.5, 1.5, nb)
    if kind == "Q8_0":
        out = np.zeros((nb, 34), np.uint8)
        out[:, :2] = f16(d * 0.05 / 40).reshape(nb, 2)
        out[:, 2:] = np.clip(rng.normal(0, 40, (nb, 32)), -127, 127).astype(np.int8).view(np.uint8)
        return out.tobytes()
    if kind == "IQ4_NL":
        out = rng.integers(0, 256, (nb, 18), dtype=np.uint8)
        out[:, :2] = f16(d * 0.05 / 60).reshape(nb, 2)
        return out.tobytes()
    if kind == "Q5_1":
        out = rng.integers(0, 256, (nb, 24), dtype=np.uint8)
        s = d * 0.05 / 9
        out[:, 0:2] = f16(s).reshape(nb, 2)
        out[:, 2:4] = f16(-15.5 * s).reshape(nb, 2)
        return out.tobytes()
    ns = n // 256
    d = rng.uniform(0.5, 1.5, ns)
    if kind == "Q4_K":
        out = rng.integers(0, 256, (ns, 144), dtype=np.uint8)
        s = d * 0.05 / (4.5 * 31.5)
        out[:, 0:2] = f16(s).reshape(ns, 2)
        out[:, 2:4] = f16(7.5 * s).reshape(ns, 2)
        return out.tobytes()
    if kind == "Q5_K":
        out = rng.integers(0, 256, (ns, 176), dtype=np.uint8)
        s = d * 0.05 / (9 * 31.5)
        out[:, 0:2] = f16(s).reshape(ns, 2)
        out[:, 2:4] = f16(15.5 * s).reshape(ns, 2)
        return out.tobytes()
    raise ValueError(kind)

tensors = []  # (name, dims in ggml order, type, bytes)
def T(name, dims, kind, data=None):
    n = int(np.prod(dims))
    tensors.append((name, list(dims), kind, data if data is not None else blocks(kind, n)))
def gamma(name, n):
    T(name, [n], "F32", rng.uniform(0.7, 1.3, n).astype(np.float32).tobytes())
def hc(pre, inject=True):
    gamma(pre + "norm.weight", HC * H)
    T(pre + "down.weight", [HC * H, RANK], "Q8_0")
    T(pre + "up.weight", [RANK, HC * H], "Q8_0")
    if inject:
        T(pre + "inject.weight", [HC * H, HC], "F32")

T("token_embd.weight", [H, VOCAB], "Q8_0")
T("output.weight", [H, VOCAB], "Q8_0")
hc("output_hc_", inject=False)
rows = (sum(PLE_SIZES) + 127) // 128 * 128
T("per_layer_token_embd.weight", [PLE_DIM, rows], "IQ4_NL")
for l in range(LAYERS):
    b = f"blk.{l}."
    hc(b + "hc_attn_")
    hc(b + "hc_ffn_")
    if (l + 1) % 4 == 0:
        T(b + "attn_q.weight", [H, HEADS * HD * 2], "Q8_0")
        T(b + "attn_k.weight", [H, KV * HD], "Q8_0")
        T(b + "attn_v.weight", [H, KV * HD], "Q8_0")
        T(b + "attn_output.weight", [HEADS * HD, H], "Q8_0")
        gamma(b + "attn_q_norm.weight", HD)
        gamma(b + "attn_k_norm.weight", HD)
        T(b + "indexer.q_proj.weight", [H, IDX_H * IDX_D], "BF16")
        T(b + "indexer.k_proj.weight", [H, IDX_D], "BF16")
        gamma(b + "indexer.q_norm.weight", IDX_D)
        gamma(b + "indexer.k_norm.weight", IDX_D)
    else:
        T(b + "attn_qkv.weight", [H, CONV], "Q8_0")
        T(b + "attn_gate.weight", [H, VAL], "Q8_0")
        T(b + "ssm_alpha.weight", [H, VH], "F32")
        T(b + "ssm_beta.weight", [H, VH], "F32")
        T(b + "ssm_conv1d.weight", [CK, CONV], "F32", rng.normal(0, 0.5, CK * CONV).astype(np.float32).tobytes())
        T(b + "ssm_dt.bias", [VH], "F32", rng.uniform(-1, 1, VH).astype(np.float32).tobytes())
        T(b + "ssm_a", [VH], "F32", (-rng.uniform(0.1, 2.0, VH)).astype(np.float32).tobytes())
        gamma(b + "ssm_norm.weight", SD)
        T(b + "ssm_out.weight", [VAL, H], "Q8_0")
    if l == 1:
        T(b + "ple_key.weight", [H, HC * H], "Q8_0")
        T(b + "ple_value.weight", [H, H], "Q8_0")
        gamma(b + "ple_norm_key.weight", HC * H)
        gamma(b + "ple_norm_query.weight", HC * H)
        gamma(b + "ple_norm_conv.weight", HC * H)
        T(b + "ple_conv1d.weight", [PLE_K, HC * H], "F32", rng.normal(0, 0.5, PLE_K * HC * H).astype(np.float32).tobytes())
    gu, dn = EXPERT_Q[l]
    T(b + "ffn_gate_inp.weight", [H, E], "F32")
    T(b + "ffn_gate_inp_shexp.weight", [H], "F32")
    T(b + "ffn_gate_shexp.weight", [H, FF], "Q8_0")
    T(b + "ffn_up_shexp.weight", [H, FF], "Q8_0")
    T(b + "ffn_down_shexp.weight", [FF, H], "Q8_0")
    T(b + "ffn_gate_exps.weight", [H, FF, E], gu)
    T(b + "ffn_up_exps.weight", [H, FF, E], gu)
    T(b + "ffn_down_exps.weight", [FF, H, E], dn)

# byte-level vocabulary (GPT-2 byte <-> unicode mapping) plus control tokens
tokens, extra = [], 0
for b in range(256):
    printable = (33 <= b <= 126) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF)
    tokens.append(chr(b) if printable else chr(256 + extra))
    extra += 0 if printable else 1
special = {EOS: "<|endoftext|>", 257: "<|im_start|>", IM_END: "<|im_end|>", 259: "<think>", 260: "</think>"}
for i in range(256, VOCAB):
    tokens.append(special.get(i, f"<|extra_{i}|>"))
token_types = [1] * 256 + [3] * (VOCAB - 256)

A = "qwen4exp."
U32, F32_, BOOL, STR, ARR, U64 = 4, 6, 7, 8, 9, 10
kv = [
    ("general.architecture", STR, "qwen4exp"), ("general.name", STR, "tiny qwen4exp test model"),
    (A + "block_count", U32, LAYERS), (A + "context_length", U32, 4096), (A + "embedding_length", U32, H),
    (A + "attention.head_count", U32, HEADS), (A + "attention.head_count_kv", U32, KV),
    (A + "rope.dimension_sections", (ARR, U32), [11, 11, 10, 0]), (A + "rope.freq_base", F32_, 1e7),
    (A + "attention.layer_norm_rms_epsilon", F32_, 1e-6),
    (A + "expert_count", U32, E), (A + "expert_used_count", U32, TOPK),
    (A + "attention.key_length", U32, HD), (A + "attention.value_length", U32, HD),
    (A + "expert_feed_forward_length", U32, FF), (A + "expert_shared_feed_forward_length", U32, FF),
    (A + "ssm.conv_kernel", U32, CK), (A + "ssm.state_size", U32, SD), (A + "ssm.group_count", U32, KH),
    (A + "ssm.time_step_rank", U32, VH), (A + "ssm.inner_size", U32, VAL),
    (A + "full_attention_interval", U32, 4), (A + "rope.dimension_count", U32, 64),
    (A + "hyper_connection.count", U32, HC), (A + "hyper_connection.low_rank", U32, RANK),
    (A + "attention.indexer.head_count", U32, IDX_H), (A + "attention.indexer.key_length", U32, IDX_D),
    (A + "attention.indexer.top_k", U32, BUDGET),
    (A + "attention.compress_ratios", (ARR, U32), [CR if (l + 1) % 4 == 0 else 0 for l in range(LAYERS)]),
    (A + "ple.layers", (ARR, U32), [1]), (A + "ple.ngram_size", U32, NGRAM), (A + "ple.heads_per_ngram", U32, PER_GRAM),
    (A + "ple.conv_kernel", U32, PLE_K), (A + "ple.eos_token_id", U32, EOS),
    (A + "embedding_length_per_layer_input", U32, PLE_DIM),
    (A + "ple.layer_multipliers", (ARR, U64), [23703573157769, 20109073645365, 8052911324071]),
    (A + "ple.head_offsets", (ARR, U64), [int(x) for x in np.cumsum([0] + PLE_SIZES[:-1])]),
    (A + "ple.head_vocab_sizes", (ARR, U64), PLE_SIZES),
    ("tokenizer.ggml.model", STR, "gpt2"), ("tokenizer.ggml.pre", STR, "qwen35"),
    ("tokenizer.ggml.tokens", (ARR, STR), tokens), ("tokenizer.ggml.token_type", (ARR, 5), token_types),
    ("tokenizer.ggml.merges", (ARR, STR), ["a b"]),
    ("tokenizer.ggml.eos_token_id", U32, IM_END), ("tokenizer.ggml.padding_token_id", U32, EOS),
    ("tokenizer.ggml.bos_token_id", U32, EOS), ("tokenizer.ggml.add_bos_token", BOOL, False),
    ("general.quantization_version", U32, 2), ("general.file_type", U32, 15),
]

def s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b
FMT = {4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q"}
def val(t, v):
    if t == STR: return s(v)
    return struct.pack(FMT[t], v)

out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
for k, t, v in kv:
    out += s(k)
    if isinstance(t, tuple):
        out += struct.pack("<IIQ", ARR, t[1], len(v)) + b"".join(val(t[1], x) for x in v)
    else:
        out += struct.pack("<I", t) + val(t, v)
off = 0
for name, dims, kind, data in tensors:
    out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
    out += struct.pack("<IQ", TYPES[kind], off)
    off += (len(data) + 31) // 32 * 32
out += b"\0" * (-len(out) % 32)
for name, dims, kind, data in tensors:
    out += data + b"\0" * (-len(data) % 32)
open(sys.argv[1], "wb").write(out)
print(f"wrote {sys.argv[1]}: {len(tensors)} tensors, {len(out) / 1e6:.1f} MB")
