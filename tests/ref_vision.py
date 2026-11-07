# Checks the engine's image preprocessing and vision tower against HF transformers (fp32, CPU).
# usage: python tests/ref_vision.py MODEL_DIR IMAGE DUMP_DIR [max_pixels]
#   DUMP_DIR is written by: set QW_VISION_DUMP=DUMP_DIR & qwen.exe --backend vulkan --image IMAGE --eval IDS
#   (IDS containing the image token); max_pixels = --image-tokens * 1024 (default 1048576).
import json, sys, numpy as np, torch
from PIL import Image, ImageOps
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M, configuration_qwen3_5_moe as C

model_dir, image, dump = sys.argv[1:4]
max_pixels = int(sys.argv[4]) if len(sys.argv) > 4 else 1 << 20
cfg = json.load(open(f"{model_dir}/config.json"))
gh, gw, rh, rw = map(int, open(f"{dump}/grid.txt").read().split())
ours_px = np.fromfile(f"{dump}/patches.f32", dtype=np.float32)
ours_emb = np.fromfile(f"{dump}/embeds.f32", dtype=np.float32)

# preprocessing
proc = AutoImageProcessor.from_pretrained(model_dir)
img = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
pp = json.load(open(f"{model_dir}/preprocessor_config.json"))
out = proc(images=img, size={"shortest_edge": pp["size"]["shortest_edge"], "longest_edge": max_pixels}, return_tensors="pt")
ref_px, grid = out["pixel_values"].float(), out["image_grid_thw"]
print(f"grid: ours {gh}x{gw}, HF {grid[0, 1].item()}x{grid[0, 2].item()}")
assert (grid[0, 1].item(), grid[0, 2].item()) == (gh, gw), "grid mismatch"
ours_px = torch.from_numpy(ours_px.reshape(ref_px.shape))
d = (ours_px - ref_px).abs()
level = 2.0 / 255  # one 8-bit level after (x/255 - 0.5) / 0.5
print(f"pixels: max |diff| {d.max():.5f} ({d.max() / level:.1f} levels), "
      f"{(d > level / 2).float().mean() * 100:.3f}% of values differ")

# vision tower
vcfg = C.Qwen3_5MoeVisionConfig(**cfg["vision_config"])
vcfg._attn_implementation = "eager"
vm = M.Qwen3_5MoeVisionModel(vcfg).float().eval()
sd = {}
import glob
for path in glob.glob(f"{model_dir}/*.safetensors"):
    with safe_open(path, "pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."): sd[k[len("model.visual."):]] = f.get_tensor(k).float()
missing, unexpected = vm.load_state_dict(sd, strict=False)
assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
thw = torch.tensor([[1, gh, gw]])
with torch.no_grad():
    ref_on_ours = vm(ours_px, grid_thw=thw).pooler_output
    ref_on_ref = vm(ref_px, grid_thw=thw).pooler_output
ours_emb = torch.from_numpy(ours_emb.reshape(ref_on_ours.shape))

def compare(name, a, b):
    rel = (a - b).norm() / b.norm()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"{name}: rel err {rel:.2e}, min token cosine {cos.min():.6f}, max |diff| {(a - b).abs().max():.4f}")
    return rel.item(), cos.min().item()

rel, cmin = compare("embeddings (same pixels: our tower vs HF tower)", ours_emb, ref_on_ours)
rel2, cmin2 = compare("embeddings (end to end: ours vs HF preprocessing + tower)", ours_emb, ref_on_ref)
ok = rel < 1e-3 and cmin > 0.9999 and rel2 < 1e-3 and cmin2 > 0.9999
print("VISION OK" if ok else "VISION MISMATCH")
sys.exit(0 if ok else 1)
