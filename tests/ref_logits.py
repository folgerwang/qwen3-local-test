# Reference logits from HF transformers (fp32, CPU) with FP8 weights dequantized. Test oracle only.
# usage: python ref_logits.py MODEL_DIR id,id,id,...
import json, sys, glob, torch
from safetensors import safe_open
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M, configuration_qwen3_5_moe as C

model_dir, ids = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
cfg = C.Qwen3_5MoeTextConfig(**json.load(open(f"{model_dir}/config.json"))["text_config"])
cfg._attn_implementation = "eager"
with torch.device("meta"):
    m = M.Qwen3_5MoeForCausalLM(cfg)
m = m.to_empty(device="cpu").float().eval()
m.model.rotary_emb = M.Qwen3_5MoeTextRotaryEmbedding(cfg)
params = dict(m.named_parameters())

def load(f, name):
    t = f.get_tensor(name)
    if t.dtype == torch.float8_e4m3fn:
        s = f.get_tensor(name + "_scale_inv").float()
        t = t.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: t.shape[0], : t.shape[1]]
    return t.float()

done = set()
for path in glob.glob(f"{model_dir}/*.safetensors"):
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.endswith("_scale_inv") or k.startswith("mtp.") or "visual" in k: continue
            name = k.replace("model.language_model.", "model.")
            if ".experts." in name:
                pre, rest = name.split(".experts.")
                e, proj = rest.split(".")[0:2]
                e = int(e)
                t = load(f, k)
                if proj == "down_proj": params[pre + ".experts.down_proj"].data[e].copy_(t)
                else:
                    off = 0 if proj == "gate_proj" else cfg.moe_intermediate_size
                    params[pre + ".experts.gate_up_proj"].data[e, off: off + cfg.moe_intermediate_size].copy_(t)
                done.add(pre + ".experts")
            else:
                params[name].data.copy_(load(f, k).reshape(params[name].shape))
                done.add(name)
missing = [n for n in params if n not in done and n.rsplit(".", 1)[0] not in done]
assert not missing, missing[:5]
with torch.no_grad():
    logits = m(torch.tensor([ids])).logits[0, -1]
top = torch.topk(logits, 10)
for i, v in zip(top.indices.tolist(), top.values.tolist()): print(i, f"{v:.4f}")
torch.save(logits, "tests/ref_last_logits.pt")
