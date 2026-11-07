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
qwen                       # chat with the 35B (add this folder to PATH to run it anywhere)
qwen 122b                  # 122B;  qwen flash = Flash-Next (CPU backend)
qwen 122b -c               # continue the last conversation
qwen -p "question"         # one-shot prompt
```

In the chat: `/help`, `/new`, `/think` (show/hide reasoning), `/reasoning on|off`, `/system TEXT`,
`/save [NAME]`, `/load NAME`, `/sessions`, `/image FILE`, `/exit`. Ctrl+C stops a reply; pasted multi-line
text is sent as one message (or wrap a block in `"""`). Conversations are saved in `models/<name>/sessions/`.

Tools (on by default; `--no-tools` or `/tools off` to disable): the model can list, read and search files
in the folder qwen was started from, search the web and read pages, and, after you approve each action, write,
edit or delete files there and run PowerShell commands. At an approval prompt answer `y`, `n`, `a` (always allow
that tool this session), or type what it should do instead. File tools cannot leave the working folder
(commands can, which is why they always ask). Web search uses DuckDuckGo and falls back to Bing; both rate
limit keyless use, so for dependable search set `BRAVE_API_KEY` (Brave Search API, free tier) and it is used
first. Try a tool by hand with `qwen --tool web_search query=...` (or `read_file path=...`, `fetch_url url=...`).

Settings in the chat: `/model` lists the downloaded models and `/model 122b` switches (any unique part of the
name; the conversation carries over as a summary), `/permission ask|edits|all|read-only` sets how tool actions
are approved (also `--permission`), `/ctx 64k` reloads with a bigger context, `/settings` shows everything and
`/set temp|top-p|top-k|greedy|max-tokens VALUE` changes sampling. A status line above each prompt shows the
model, backend, context use, permission mode and reasoning setting.

Memory: on the Vulkan backend the expert cache fills the VRAM except `--vram-reserve` (default 3 GB), and
every expert gets a pinned (unpageable) RAM copy the GPU reads when it is not cached: 30 GB for the 35B, 108 GB
for the 122B. If that would leave less than `--ram-reserve` (default 16 GB) free, nothing is pinned and the CPU
computes uncached experts instead (about half the speed on the 122B). The model files are read once while
loading and their pages are handed back to Windows' file cache afterwards. `/info` shows the current use;
`/set vram-reserve GB` and `/set ram-reserve GB` change the reserves (the model reloads).

Project instructions: `AGENTS.md` (or `CLAUDE.md`) in the working folder and its parents up to the git root is
added to the system prompt; `/init` has the model study the project and write one. `/undo` reverts the file
changes of the model's last turn that changed files (repeat for earlier ones; commands are not undone) and
tells the model. Long conversations are summarized automatically when the context fills up (`/compact` does
it by hand), including in the middle of a multi-step tool task, which then continues in the fresh context.

Images (35B and 122B, Vulkan backend): drop an image file into the chat window or use `/image FILE`, then
ask about it; one-shot: `qwen -p "question" --image photo.jpg`. Images are resized to at most
`--image-tokens` (default 1024) tokens of 32x32 pixels. The vision tower takes ~1 GB of VRAM; `--no-vision`
gives it to the expert cache instead. `python tests/ref_vision.py` checks preprocessing and the vision tower
against transformers; `tests/check.py vulkan` includes an image prompt.

Older launchers:

```
chat.bat [cpu|vulkan]      # 35B
chat-122b.bat              # 122B
chat-flash.bat [cpu|vulkan]  # Qwen3.8-Flash-Next
python tests/check.py      # compare logits against the fp32 reference
python tests/check.py cpu flash   # same for Flash-Next (oracle: tests/ref_logits_flash.py)
```
