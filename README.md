# qwen3-local-test

From-scratch C++ inference engine for Qwen MoE models (CPU AVX-512 + hand-written Vulkan compute shaders, no ML libraries).

- **Qwen3.8-Flash-Next 128B** from Unsloth's 4-bit **UD-Q4_K_XL** GGUF (the default model and format). The GGUF is read
  in place: Q4_K / Q5_K / Q5_1 / Q8_0 / IQ4_NL blocks are decoded on the fly by the CPU kernels and the shaders.
- Qwen3.5 / 3.6 MoE models from FP8 safetensors.

## Setup

Requires Visual Studio 2022, CMake ≥ 3.20, the Vulkan SDK and Python 3.

```
setup.bat                  # download Qwen3.8-Flash-Next UD-Q4_K_XL (~111 GB) and build
setup.bat --model 35b      # Qwen3.6-35B-A3B-FP8 (~35 GB)
setup.bat --model 122b     # Qwen3.5-122B-A10B-FP8 (~119 GB)
setup.bat --model all      # all models (flash, 35B, 122B, 27B)
setup.bat --model none     # build only
setup.bat --cpu-only       # build without the Vulkan backend
```

Weights go to `models/<name>/` (git-ignored). Re-running resumes/skips completed downloads.

## Run

```
chat.bat [cpu|vulkan]      # Qwen3.8-Flash-Next UD-Q4_K_XL
chat-35b.bat [cpu|vulkan]  # 35B (FP8)
chat-122b.bat              # 122B (FP8)
```

`qwen.exe --model PATH` takes a directory of `*.gguf` shards (or one `.gguf` file) or an FP8 safetensors model
directory. Any other GGUF quantization of Qwen3.8-Flash-Next works on the CPU backend as long as its tensors are
F32 / F16 / BF16 / Q8_0 / Q4_K / Q5_K / Q6_K / Q5_1 / IQ4_NL; the Vulkan backend needs the routed experts in
Q4_K, Q5_K, Q8_0 (gate/up) and additionally Q5_1 (down).

For the GGUF model the Vulkan backend is built for a machine with much more RAM than VRAM: the dense weights
(~5.5 GB) and a cache of the most used experts (6 GB by default, `--expert-cache MB`, 0 = none) live in VRAM, and the
CPU computes every other expert straight from the memory-mapped GGUF (72 GB of experts, so plan for ~96 GB of RAM or
more). Each layer does one GPU round trip, so other programs using the GPU slow it down noticeably. `--threads N`
sets the CPU threads for the experts (default: 3/8 of the hardware threads).

## Tests

```
python tests/check_gguf.py tiny      # tiny random GGUF: CPU and Vulkan against HF transformers (needs transformers >= 5.8)
python tests/check_gguf.py models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL   # real model: Vulkan against CPU
python tests/check.py                # 35B FP8: compare logits against the fp32 reference
```
