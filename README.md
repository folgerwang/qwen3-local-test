# qwen3-local-test

From-scratch C++ inference engine for Qwen FP8 MoE models (CPU AVX-512 + hand-written Vulkan compute shaders, no ML libraries).

## Setup

Requires Visual Studio 2022, CMake ≥ 3.20, the Vulkan SDK and Python 3.

```
setup.bat                  # download Qwen3.6-35B-A3B-FP8 (~35 GB) and build
setup.bat --model 122b     # Qwen3.5-122B-A10B-FP8 (~119 GB)
setup.bat --model flash    # Qwen3.8-Flash-Next-FP8 (~186 GB, engine support in progress)
setup.bat --model all      # all models (35B, 122B, Flash-Next, 27B)
setup.bat --model none     # build only
setup.bat --cpu-only       # build without the Vulkan backend
```

Weights go to `models/<name>/` (git-ignored). Re-running resumes/skips completed downloads.

## Run

```
chat.bat [cpu|vulkan]      # 35B
chat-122b.bat              # 122B
chat-flash.bat [cpu|vulkan]  # Qwen3.8-Flash-Next
python tests/check.py      # compare logits against the fp32 reference
python tests/check.py cpu flash   # same for Flash-Next (oracle: tests/ref_logits_flash.py)
```
