# One-shot setup: downloads model weights from Hugging Face into models/ and builds qwen.exe.
# usage: python setup.py [--model 35b|122b|flash|27b|all|none] [--no-build] [--cpu-only]
# Weights are not stored in git; re-running skips files that are already complete.
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "35b": "Qwen/Qwen3.6-35B-A3B-FP8",     # default model (chat.bat), ~35 GB
    "122b": "Qwen/Qwen3.5-122B-A10B-FP8",  # chat-122b.bat, ~119 GB
    "122b-int4": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",  # 4-bit experts, ~73 GB: half the RAM, faster
    "flash": "Qwen/Qwen3.8-Flash-Next-FP8",  # 180B MoE, ~186 GB (engine support in progress)
    "27b": "Qwen/Qwen3.8-27B-FP8",         # dense, ~31 GB (not supported by the engine yet)
}

def download(repo):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the Xet backend stalled mid-download; plain HTTP is ~115 MB/s
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    dst = os.path.join(ROOT, "models", repo.split("/")[1])
    print(f"==> {repo} -> {dst}")
    snapshot_download(repo_id=repo, local_dir=dst, max_workers=8)

def build(vulkan):
    bdir = os.path.join(ROOT, "build")
    cfg = ["cmake", "-S", ROOT, "-B", bdir, f"-DQW_VULKAN={'ON' if vulkan else 'OFF'}"]
    if os.name == "nt" and not os.path.exists(os.path.join(bdir, "CMakeCache.txt")):
        cfg += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    subprocess.check_call(cfg)
    subprocess.check_call(["cmake", "--build", bdir, "--config", "Release", "--parallel"])

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="35b", choices=[*MODELS, "all", "none"])
    ap.add_argument("--no-build", action="store_true", help="skip the CMake build")
    ap.add_argument("--cpu-only", action="store_true", help="build without the Vulkan backend")
    a = ap.parse_args()
    for key in (MODELS if a.model == "all" else [] if a.model == "none" else [a.model]):
        download(MODELS[key])
    if not a.no_build:
        build(not a.cpu_only)
    print("Done. Run chat.bat (35B) or chat-122b.bat (122B).")

if __name__ == "__main__":
    main()
