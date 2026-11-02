# Reference logits for Qwen3.8-Flash-Next (qwen4_exp) from HF transformers in fp32 on the CPU. Test oracle only.
# The model is too big to materialize in fp32, so decoder layers are built, loaded and run one at a time, and the
# 51 GB n-gram embedding table is read row by row from its FP8 shards.
# usage: python ref_logits_flash.py MODEL_DIR id,id,id,... [OUT.pt]
import json, os, sys, torch
from safetensors import safe_open
from transformers.models.qwen4_exp import modeling_qwen4_exp as M, configuration_qwen4_exp as C

model_dir, ids = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
out_path = sys.argv[3] if len(sys.argv) > 3 else None
cfg = C.Qwen4ExpTextConfig(**json.load(open(f"{model_dir}/config.json"))["text_config"])
cfg._attn_implementation = "eager"
wmap = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
files = {}

def tensor(name):
    fn = wmap[name]
    if fn not in files: files[fn] = safe_open(f"{model_dir}/{fn}", "pt")
    return files[fn].get_tensor(name)

def weight(name):
    t = tensor(name)
    if t.dtype == torch.float8_e4m3fn:
        s = tensor(name + "_scale_inv").float()
        t = t.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: t.shape[0], : t.shape[1]]
    return t.float()

def build(cls, prefix, *args):
    with torch.device("meta"):
        mod = cls(*args)
    mod = mod.to_empty(device="cpu").float().eval()
    for n, p in mod.named_parameters():
        if ".ngram_embedding." in n: continue
        if n.endswith("experts.gate_up_proj") or n.endswith("experts.down_proj"):
            base = prefix + n.rsplit(".", 1)[0] + "."
            I = cfg.moe_intermediate_size
            for e in range(cfg.num_experts):
                if n.endswith("down_proj"): p.data[e].copy_(weight(f"{base}{e}.down_proj.weight"))
                else:
                    p.data[e, :I].copy_(weight(f"{base}{e}.gate_proj.weight"))
                    p.data[e, I:].copy_(weight(f"{base}{e}.up_proj.weight"))
            continue
        p.data.copy_(weight(prefix + n).reshape(p.shape))
    for n, b in mod.named_buffers():  # n-gram hashing tables are stored in the checkpoint
        if prefix + n in wmap: b.data.copy_(tensor(prefix + n))
    return mod

class ShardedFp8Embedding(torch.nn.Module):
    """Row lookup in the FP8 n-gram table (shards concatenated along rows, one per-tensor scale)."""
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self.scale = tensor(prefix + "weight_scale").float().item()
        self.register_buffer("weight", torch.empty(0))  # the model only reads weight.device

    def forward(self, idx):
        shard0 = self.prefix + "shard_0.weight"
        fn = wmap[shard0]
        if fn not in files: files[fn] = safe_open(f"{model_dir}/{fn}", "pt")
        R = files[fn].get_slice(shard0).get_shape()[0]
        flat = idx.reshape(-1).tolist()
        out = []
        for r in flat:
            name = f"{self.prefix}shard_{r // R}.weight"
            if wmap[name] not in files: files[wmap[name]] = safe_open(f"{model_dir}/{wmap[name]}", "pt")
            out.append(files[wmap[name]].get_slice(name)[r % R : r % R + 1].float()[0] * self.scale)
        return torch.stack(out).reshape(*idx.shape, -1)

P = "model.language_model."
if os.environ.get("QW_DENSE_ATTN"):  # debug: every visible position attended (indexer off)
    M.Qwen4ExpTextQSAIndexer.forward = lambda self, h, pe, mask, cache: torch.zeros_like(mask)
x_ids = torch.tensor([ids])
n = len(ids)
with torch.no_grad():
    h = tensor(P + "embed_tokens.weight")[x_ids].float()
    pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
    rot = M.Qwen4ExpTextRotaryEmbedding(cfg)
    pe = rot(h, pos)
    mask = torch.full((n, n), torch.finfo(torch.float32).min).triu(1).view(1, 1, n, n)
    h = h.repeat(1, 1, cfg.hc_count)
    for l in range(cfg.num_hidden_layers):
        layer = build(M.Qwen4ExpTextDecoderLayer, f"{P}layers.{l}.", cfg, l)
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = ShardedFp8Embedding(f"{P}layers.{l}.ple.ple_embedding.ngram_embedding.")
        h = layer(h, position_embeddings=pe, attention_mask=mask, conv_mask=None, past_key_values=None, ple_input_ids=x_ids)
        del layer
        print(f"layer {l} done", file=sys.stderr, flush=True)
    mixer = build(M.Qwen4ExpTextGatedResidual, P + "hyper_connection_mixer.", cfg, False)
    last = mixer(h)[0, -1]
    logits = weight("lm_head.weight") @ last
top = torch.topk(logits, 10)
for i, v in zip(top.indices.tolist(), top.values.tolist()): print(i, f"{v:.4f}")
if out_path: torch.save(logits, out_path)
