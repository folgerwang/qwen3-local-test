# SiCode

A terminal coding assistant in two layers:

- **Architect: Claude, over the Claude API.** It talks to you, decides what to do, writes self-contained tasks,
  and reviews the result (the worker's report and the git diff). It can read, search and run commands, but it
  cannot edit files.
- **Worker: Qwen3.8-Flash-Next, in `qwen.exe` on your GPU.** Each task starts a fresh worker that reads, edits
  and runs things with its own tool loop until it reports back.

## Install

From the repository root, after building the engine and downloading the model (`setup.bat --model flash-gguf`):

```
pip install -e .
```

The editable install finds the engine and the model inside the repository. For a regular install
(`pip install .`), point it at them:

```
set SICODE_ENGINE=G:\work\qwen3-local-test\build\Release\qwen.exe
set SICODE_MODEL=G:\work\qwen3-local-test\models\Qwen3.8-Flash-Next-GGUF\UD-Q4_K_XL
```

## Use

```
cd your-project
sicode                                        # interactive
sicode -p "add a --json flag to the cli"      # one request, on the local model
sicode -p "[c] redesign the config loading"   # one request, planned and reviewed by Claude
python -m sicode                              # the same as sicode
```

Each request says who takes it:

| Request | Who handles it |
|---|---|
| `[c] task` | Claude plans, hands the hands-on work to the local model, and reviews the result |
| `[l] task` | the local model alone |
| `task` (no prefix) | the local model alone |

Only `[c]` requests use the Claude API. If no key is set, SiCode asks for one when it starts (typed hidden; Enter
skips it) and saves it in `~/.sicode/config.json` for later runs; `/key` asks again, and `ANTHROPIC_API_KEY` in the
environment takes precedence over the saved key. The two sides keep
separate conversations: consecutive local requests continue one conversation, `[c]` requests continue Claude's,
and neither sees the other's. A `[c]` request that delegates work starts the local conversation afresh.

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

### Options

`--mode`, `--dir PATH` (the workspace), `--model` and `--effort` (the Claude model and its effort, default
`claude-opus-5-5`, `high`), `--worker-effort low|medium|xhigh` (the local model's reasoning effort), `--exe` and
`--worker-model` (the engine and model paths).

## Modules

- `sicode.cli`: the terminal front end and the architect loop.
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
