# One-shot setup: downloads model weights from Hugging Face into models/ and builds qwen.exe.
# usage: python setup.py [--model flash|35b|122b|27b|all|none] [--no-build] [--cpu-only]
# Weights are not stored in git; re-running skips files that are already complete.
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# key -> (repo, file patterns or None for the whole repo)
MODELS = {
    # default model (chat.bat): Qwen3.8-Flash-Next 128B as Unsloth's 4-bit UD-Q4_K_XL GGUF, ~111 GB
    # plus its MTP draft head (4 GB) for speculative decoding
    "flash": ("unsloth/Qwen3.8-Flash-Next-GGUF", ["UD-Q4_K_XL/*", "mtp-*.gguf"]),
    "35b": ("Qwen/Qwen3.6-35B-A3B-FP8", None),     # chat-35b.bat, ~35 GB
    "122b": ("Qwen/Qwen3.5-122B-A10B-FP8", None),  # chat-122b.bat, ~119 GB
    "27b": ("Qwen/Qwen3.8-27B-FP8", None),         # ~12 GB
}

def download(repo, patterns):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    dst = os.path.join(ROOT, "models", repo.split("/")[1])
    print(f"==> {repo}{' ' + ' '.join(patterns) if patterns else ''} -> {dst}")
    snapshot_download(repo_id=repo, local_dir=dst, allow_patterns=patterns, max_workers=8)

def build(vulkan):
    bdir = os.path.join(ROOT, "build")
    cfg = ["cmake", "-S", ROOT, "-B", bdir, f"-DQW_VULKAN={'ON' if vulkan else 'OFF'}"]
    if os.name == "nt" and not os.path.exists(os.path.join(bdir, "CMakeCache.txt")):
        cfg += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    subprocess.check_call(cfg)
    subprocess.check_call(["cmake", "--build", bdir, "--config", "Release", "--parallel"])

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="flash", choices=[*MODELS, "all", "none"])
    ap.add_argument("--no-build", action="store_true", help="skip the CMake build")
    ap.add_argument("--cpu-only", action="store_true", help="build without the Vulkan backend")
    a = ap.parse_args()
    for key in (MODELS if a.model == "all" else [] if a.model == "none" else [a.model]):
        download(*MODELS[key])
    if not a.no_build:
        build(not a.cpu_only)
    print("Done. Run chat.bat (Qwen3.8-Flash-Next), chat-35b.bat or chat-122b.bat.")

if __name__ == "__main__":
    main()
