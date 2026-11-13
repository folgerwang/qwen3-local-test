# Project notes for Claude

From-scratch C++ inference engine (CPU AVX-512 + Vulkan compute shaders, no ML libraries) plus SiCode, a terminal
coding assistant built on it. The README describes setup and usage; this file records what is not obvious from
the code: priorities, measured facts, dead ends and working rules.

## Priorities (set by the owner)

1. Generation speed.
2. Then VRAM use as small as possible. The machine has 192 GB of RAM, so RAM is cheap.
3. The default model is Qwen3.8-Flash-Next from Unsloth's UD-Q4_K_XL GGUF, with a 256K default context.

## Machine

- Ryzen 9 9950X3D (16 cores / 32 threads), 4 x 48 GB DDR5-5600 (dual channel), RTX 4090 24 GB (PCIe 4.0), Windows 11.
- Other programs usually hold about 9-11 GB of VRAM (LM Studio, a `llama-server.exe`, a game-preview tool). They
  belong to the owner: never close or kill them. They make benchmarks noisy (often +-15%).

## How the GGUF path works (`src/qwen4_vk.cpp`)

- Dense weights (~5.5 GB) and state live in VRAM. All 25,088 experts (74.6 GB) stay in RAM, memory-mapped from
  the GGUF; a VRAM cache holds the most used ones and the CPU computes the rest with fused AVX-512 dot kernels
  (`src/quant.cpp`). One GPU sync per layer.
- The cache adapts per session: after each pass the most used missed experts (over all layers, up to 64) replace
  the least used cached ones.
- The MTP draft head (`mtp-*.gguf`) drafts up to 3 tokens per pass; a draft is kept only if it equals the token
  sampled from the trunk's logits, so greedy output is identical with and without it.
- GGUF stores DeltaNet value heads in tiled order (key head = h % 16). The shader softplus uses a series for small
  exp(x); `log(1 + exp(x))` loses precision there.

## Measured facts (do not re-derive)

- Speed on the shared 4090: about 24-28 tok/s generation, about 120-137 tok/s prompt reading.
- The CPU expert step is the largest cost (roughly two thirds of generation), the per-layer GPU wait about one third.
- Expert kernel, 16 threads: 33 us per expert when the weights are in CPU cache, 81 us from RAM. One thread: about
  310 us either way. So with 16 threads RAM speed is the limit, with one thread arithmetic is.
- 16 threads (one per core) is the fastest setting; 32 is slower.
- A hindsight-ideal cache would serve about 56-64% of lookups at 5-6 GB, 78% at 10.5 GB, 83% at 13 GB. The
  adaptive cache gets within about 10 points of that.
- Expert weights are incompressible: 3.84-3.99 bits of entropy per 4-bit value, 7.94 per 8-bit value; zstd and
  zlib leave them at 96-100% of their size.
- Vulkan's memory budget ignores what other programs hold. Overcommitting VRAM is far slower than a small cache
  (seen: 28 -> 11 tok/s). Windows also does not count a buffer as used until it is touched.
- Router near-ties (10th vs 11th expert within ~1e-5) make CPU and GPU summation orders pick different experts;
  logits then differ by ~0.1. This is not a bug; `tests/check_gguf.py` allows 0.5 on the real model.
- Strata (github.com/Niko1221/Strata) reports 38 tok/s for this same file on a dedicated RTX 3090. Its ~160 tok/s
  on an RTX 5090 is for a 2-bit IQ2_XS pack, not UD-Q4_K_XL.

## Tried and rejected

- Pinning experts and streaming them to the GPU over PCIe on demand: 5-7 tok/s.
- A fixed 12 GB cache while other programs hold VRAM: overcommits, 14 tok/s.
- Background warm-up threads: slowed everything; warm-up is a blocking page touch at startup.
- Lossless compression of weights (zip-style, texture formats): nothing to gain, see the entropy figures.
- Predicting the next layer's experts and reading them ahead during the GPU wait: the free predictor (next
  layer's router on the current input) finds 67% of the chosen experts in its top 10 and 85% in its top 20, but
  the read-ahead gave no measurable speedup, and a trained (ridge) correction did not improve accuracy.

## Open ideas

- More VRAM for the cache is the main lever (the owner closing other GPU programs, or a card with more VRAM).
- Lossy: re-quantize the Q8_0 down matrices (1.74 MB of an expert, against 0.92 MB each for gate and up), or
  support the 2-bit / 3-bit ISTA packs (needs IQ2 / IQ3 codecs). Quality cost not measured.
- Cut the per-layer GPU wait (about 0.5 ms per layer per pass, mostly small dispatches and sync latency).
- An AVX-512 VNNI kernel would only pay off if weights were in CPU cache or with fewer threads.

## Unverified

- MTP numerics against a reference (only acceptance rate and unchanged greedy output).
- The FP8 backend after the `vk_common.h` refactor (those weights are not on this machine).
- Sparse attention (QSA) on the real model beyond 2,048 tokens; it is exact on the tiny model.
- Intermittent runs at 0.5-3 tok/s with the CPU step on about one core, seen on 2026-11-13; same binary was
  normal before and after. Cause unknown.

## Working rules

- Build: `cmake --build build --config Release --parallel`. Launch `build/Release/qwen.exe` with an absolute
  Windows path from Python.
- Tests: `python tests/check_gguf.py tiny` (CPU and Vulkan against HF transformers),
  `python tests/check_gguf.py models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL` (Vulkan against CPU). `QW_KV_START=8`
  forces K/V cache growth. `tests/test_tokenizer.py` needs a model folder with `tokenizer.json`.
- Benchmark with `QW_PROFILE=1`, `--greedy --no-think --no-stats`, and a prompt not used before: saved usage
  statistics from an earlier run of the same prompt inflate the cache hit rate. Repeat A/B runs; single runs
  are too noisy.
- Do not run downloaded third-party binaries (llama.cpp was denied). Use HF transformers on the tiny synthetic
  model as the reference.
- Bash heredocs turn `\\n` inside C++ string literals into real newlines. Write patch scripts with the Write tool
  or use Edit.
- Commit and push only when asked. `main` is the working branch on GitHub (`folgerwang/qwen3-local-test`).

## State on 2026-11-13

Committed to `main` with this file: the expert cache sized from the VRAM that is really free at startup less a
2 GB margin (NVML, `src/vram_query.cpp`; a fixed 6 GB where NVML is not available; `--expert-cache MB` caps it),
and K/V caches that start at 16,384 tokens and double on demand up to the context limit (taking VRAM back from
the expert cache). At the 256K default context this brought
generation from 11 to about 24 tok/s.
