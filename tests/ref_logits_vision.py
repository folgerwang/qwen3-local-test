# Reference logits for a prompt with an image, from HF transformers (fp32, CPU) with FP8 weights dequantized.
# Test oracle only (needs ~150 GB of RAM for the 35B).
# usage: python tests/ref_logits_vision.py MODEL_DIR IMAGE id,id,...  (the image token id marks the image)
import json, sys, glob, torch
from PIL import Image, ImageOps
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M, configuration_qwen3_5_moe as C

model_dir, image, ids = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
raw = json.load(open(f"{model_dir}/config.json"))
cfg = C.Qwen3_5MoeConfig(**raw)
for c in (cfg, cfg.text_config, cfg.vision_config): c._attn_implementation = "eager"
with torch.device("meta"):
    m = M.Qwen3_5MoeForConditionalGeneration(cfg)
m = m.to_empty(device="cpu").float().eval()
# non-persistent buffers are left uninitialized by to_empty
m.model.language_model.rotary_emb = M.Qwen3_5MoeTextRotaryEmbedding(cfg.text_config)
m.model.visual.rotary_pos_emb = M.Qwen3_5MoeVisionRotaryEmbedding(cfg.vision_config)
params = dict(m.named_parameters())
tc = cfg.text_config

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
            if k.endswith("_scale_inv") or k.startswith("mtp."): continue
            if ".experts." in k:
                pre, rest = k.split(".experts.")
                e, proj = rest.split(".")[0:2]
                t = load(f, k)
                if proj == "down_proj": params[pre + ".experts.down_proj"].data[int(e)].copy_(t)
                else:
                    off = 0 if proj == "gate_proj" else tc.moe_intermediate_size
                    params[pre + ".experts.gate_up_proj"].data[int(e), off: off + tc.moe_intermediate_size].copy_(t)
                done.add(pre + ".experts")
            else:
                params[k].data.copy_(load(f, k).reshape(params[k].shape))
                done.add(k)
missing = [n for n in params if n not in done and n.rsplit(".", 1)[0] not in done]
assert not missing, missing[:5]

proc = AutoImageProcessor.from_pretrained(model_dir)
pp = json.load(open(f"{model_dir}/preprocessor_config.json"))
img = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
out = proc(images=img, size={"shortest_edge": pp["size"]["shortest_edge"], "longest_edge": 1 << 20}, return_tensors="pt")
grid = out["image_grid_thw"]
n_img = int(grid.prod()) // 4
at = ids.index(raw["image_token_id"])
full = ids[:at] + [raw["image_token_id"]] * n_img + ids[at + 1:]
mm = [1 if t == raw["image_token_id"] else 0 for t in full]
with torch.no_grad():
    logits = m(input_ids=torch.tensor([full]), pixel_values=out["pixel_values"].float(), image_grid_thw=grid,
               mm_token_type_ids=torch.tensor([mm])).logits[0, -1]
top = torch.topk(logits, 10)
for i, v in zip(top.indices.tolist(), top.values.tolist()): print(i, f"{v:.4f}")
