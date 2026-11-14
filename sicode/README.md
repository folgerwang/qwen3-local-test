# SiCode

A coding assistant for the terminal and the browser, built on two models:

- **The local model: Qwen3.8-Flash-Next, in `qwen.exe` on your GPU.** It reads, edits and runs things with its own
  tool loop. Free to run, slower and less capable than a cloud model.
- **A cloud model: Claude or Codex.** Reached with a Claude API key, or with your Claude or ChatGPT account through
  the vendor's own command-line tool (Claude Code, Codex).

Each request runs in one of three modes: **local** (the local model alone), **cloud** (the cloud model alone) or
**mix** (the cloud model splits the job into steps and gives each to the local model or to itself).

## Install

On 64-bit Windows, from the wheel on the [releases page](https://github.com/folgerwang/qwen3-local-test/releases):

```
pip install sicode-0.2.1-py3-none-win_amd64.whl
sicode --download-model
```

The wheel includes a prebuilt engine. `--download-model` fetches Qwen3.8-Flash-Next UD-Q4_K_XL and its draft head
(about 115 GB) from Hugging Face into `~/.sicode/models`; it resumes if interrupted. The engine needs a CPU with
AVX-512, a Vulkan 1.3 GPU driver (or `--backend cpu`) and the Microsoft Visual C++ runtime.

From a checkout of the repository, after `setup.bat`, `pip install -e .` uses the engine and the model of the
checkout instead. The engine is looked for in this order: `SICODE_ENGINE`, the checkout's `build\Release`, the one in
the wheel. The model: `SICODE_MODEL`, the checkout's `models`, `~/.sicode/models`. To set them explicitly:

```
set SICODE_ENGINE=G:\work\qwen3-local-test\build\Release\qwen.exe
set SICODE_MODEL=G:\work\qwen3-local-test\models\Qwen3.8-Flash-Next-GGUF\UD-Q4_K_XL
```

## Use

```
cd your-project
sicode                                        # interactive
sicode -p "add a --json flag to the cli"      # one request, on the local model
sicode -p "[m] redesign the config loading"   # mix: the cloud model splits the job between the two
sicode -p "[c] redesign the config loading"   # cloud: Claude or Codex does it
python -m sicode                              # the same as sicode
```

Each request says who takes it:

| Request | Who handles it |
|---|---|
| `[l] task` or `task` (no prefix) | local: the local model alone |
| `[m] task` (or `[h]`) | mix: the cloud model plans the job as steps and assigns each one, see below |
| `[c] task` | cloud: the cloud model does the work itself; the local model is not used |

Before 0.2.1, `[c]` meant what `[m]` means now.

In the mix mode the cloud model assigns every step by ability and cost. Steps marked `[l]` go to the local
model, which is free but slow and less capable: mechanical edits, boilerplate, tests for specified behaviour,
exploration, builds and test runs. Steps marked `[c]` it does itself: the design, subtle bugs, changes that must be
right across several files, the final review, and edits too small to be worth a brief. The plan, every hand-off and
every file change carry the mark of who did it.

The cloud model is chosen with `--planner`, `/planner` or Settings in the page, and remembered:

| Planner | Signs in with | Notes |
|---|---|---|
| `claude-code` | your Claude account, through Claude Code | no API key; the default when `claude` is installed |
| `codex` | your ChatGPT account, through Codex | no API key; needs Codex's sandbox to work on your machine |
| `api` | a Claude API key | billed per token; asked for at start, `/key` asks again |

With an account login SiCode runs the vendor's tool in the working folder and never sees the account's credentials.
The tool edits files itself, in the cloud mode and for its own `[c]` steps of the mix mode: SiCode cannot ask
before each of those changes, so it shows them afterwards as diffs and `/undo` puts them back. Claude Code may run
shell commands only when the permission mode is `auto` (otherwise commands are `[l]` steps); Codex runs them inside
its own sandbox. With the API key, the cloud model uses SiCode's own tools and asks like the local model does.

The local and the cloud side keep separate conversations: consecutive local requests continue one conversation,
cloud and mix requests continue the cloud model's, and neither sees the other's. A mix request that hands off
work starts the local conversation afresh.

### In a session

Replies are printed as they are written, each tool call is shown as it runs, and each file change is shown as a
diff. Ctrl+C stops the current reply and keeps the conversation.

| Input | What it does |
|---|---|
| `@path` in a request | attaches that file's content |
| `!command` | runs a shell command yourself, outside any model |
| a line ending in `\` | continues on the next line |
| `/help` | lists everything here |
| `/clear` | starts new conversations (local and Claude) |
| `/compact` | replaces the local conversation by its summary, to free context |
| `/context` | context use, tokens spent, mode, models |
| `/diff` | git diff of the workspace |
| `/undo` | puts back the files changed by the last request |
| `/init` | has the local model write `SICODE.md` |
| `/mode [ask\|edits\|auto]` | shows or sets the permission mode |
| `/model [name]` | Claude model for `[c]`: `opus`, `sonnet`, `haiku`, `fable` or a full id |
| `/think` | shows or hides the local model's reasoning |
| `/key` | sets the Claude API key |
| `/exit`, `exit`, `quit` | leaves SiCode |

### In the browser

```
sicode --web                     # opens http://127.0.0.1:8765/?t=... in the browser
sicode --web --dir D:\project --port 9000 --no-browser
```

The same assistant as a page, organised by task:

- **Tasks** in the left column. "New task" first asks for the working folder: type or paste a path, pick a recent
  folder, walk through the folders (their files are listed too), or press Browse for the Windows folder dialog. Each task has its own folder and conversation and is saved in `~/.sicode/tasks`, so it is
  still there after a restart. Going back to an older task continues it from its transcript.
- **The conversation** in the middle: replies as they are written, each file read, each command with its output,
  each file change as a diff. In the `ask` mode a change or a command waits for Allow, Always allow or Decline.
- **Local, Mix or Claude (Codex)** next to the Send button chooses the mode of the request, like `[l]`, `[m]`
  and `[c]`. Stop interrupts the reply.
- **Progress, changed files, the folder and the context** on the right. A changed file opens with its changes; Undo
  puts back the files of the last request; Compact replaces the local conversation by its summary.
- **Settings** holds the cloud model (account login or API key), the VRAM cap, the Claude API key and model.
- **The bar at the top** shows the RAM and the VRAM in use (all programs, refreshed every two seconds) and the VRAM
  cache rate: the share of expert lookups of the last reply that were served from VRAM instead of computed on the
  CPU. Hovering shows the rate since the model was loaded and how many experts are cached.

The local model holds one conversation at a time, so one request runs at a time. The page is served only on
127.0.0.1 and every request needs the token in the address that `sicode --web` prints (kept in
`~/.sicode/config.json`, so the address can be bookmarked). Anyone who can open that address can run shell commands
as you: do not forward the port.

### Permissions

| Mode | File changes | Shell commands |
|---|---|---|
| `ask` (default) | asks, after showing the diff | asks |
| `edits` | applied | asks |
| `auto` (`--yes`) | applied | run |

Answering `a` to a file change switches to `edits`; answering `a` to a command allows every command that starts
with the same word for the rest of the session. All file tools are confined to the workspace. Shell commands run in
the workspace root but are otherwise not sandboxed. With `-p` and no terminal, nothing can be asked, so use
`--mode auto` (or `edits`) there.

### Project notes

`SICODE.md` at the workspace root (or, failing that, `CLAUDE.md` or `AGENTS.md`) is loaded into every conversation,
local and Claude. `/init` writes it.

### VRAM cap

By default the local model takes the VRAM that is free when it loads, less a 2 GB margin, for its expert cache.
`--vram-cap GB` (or Settings in the page; remembered) sets the most VRAM that may be in use on the card, counting
other programs: the model sizes its cache to stay under it. A lower cap is slower (measured on an RTX 5090: about
1.5 tok/s per GB of cache). The cap is applied when the model loads, from what other programs hold at that moment.

### Options

`--mode`, `--dir PATH` (the workspace), `--model` and `--effort` (the Claude model and its effort, default
`claude-opus-5-5`, `high`), `--worker-effort low|medium|xhigh` (the local model's reasoning effort), `--exe` and
`--worker-model` (the engine and model paths).

## Modules

- `sicode.cli`: the terminal front end and the architect loop.
- `sicode.planners`: the cloud model through an account login (Claude Code, Codex).
- `sicode.sysinfo`: RAM and VRAM figures for the page's status bar.
- `sicode.web`: the browser front end (`--web`): a local HTTP server, with the page in `sicode/webui/index.html`.
- `sicode.worker`: the engine process (`Engine`) and the worker's tool loop (`run_task`), in the tool-call format
  of the model's chat template.
- `sicode.tools`: the file and shell tools both layers use (`Workspace`, `TOOLS`).

## The engine's pipe mode

`qwen.exe --pipe` reads one JSON object per line on stdin and writes one per line on stdout (logs go to stderr).
It prints `{"ready":true,"ctx":N}` once the model is loaded.

- `{"op":"reset"}` clears the conversation. Reply: `{"ok":true}`.
- `{"user":TEXT, "system":TEXT, "max_tokens":N, "think":BOOL, "stream":BOOL, "interrupt_file":PATH}` appends a
  user turn and generates the reply. `system` only counts on the first turn after a reset. With `stream`,
  `{"delta":TEXT}` lines come first, as the reply is written. Generation stops early once the file named by
  `interrupt_file` exists. Reply: `{"text":..., "finished":BOOL, "interrupted":BOOL, "gen_tokens":N,
  "prefill_tokens":N, "gen_tps":X, "prefill_tps":X, "ctx_used":N, "ctx":N}`. `text` includes the reasoning, up to
  `</think>`; `finished` is false when the reply hit `max_tokens` or the context is full.

The engine keeps the conversation, so each request sends only the new turn.
