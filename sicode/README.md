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

Only `[c]` requests use the Claude API, so `ANTHROPIC_API_KEY` is needed for those alone. The two sides keep
separate conversations: consecutive local requests continue one conversation, `[c]` requests continue Claude's,
and neither sees the other's. A `[c]` request that delegates work starts the local conversation afresh.

Options: `--yes` runs shell commands without asking (default: ask each time, `a` = always), `--dir PATH` sets the
workspace, `--model` and `--effort` choose the Claude model and its effort (default `claude-opus-5-5`, `high`),
`--worker-effort low|medium|xhigh` sets the worker's reasoning effort, `--exe` and `--worker-model` override the
engine and model paths.

All file tools are confined to the workspace. Shell commands run in the workspace root but are otherwise not
sandboxed.

## Modules

- `sicode.cli`: the terminal front end and the architect loop.
- `sicode.worker`: the engine process (`Engine`) and the worker's tool loop (`run_task`), in the tool-call format
  of the model's chat template.
- `sicode.tools`: the file and shell tools both layers use (`Workspace`, `TOOLS`).

## The engine's pipe mode

`qwen.exe --pipe` reads one JSON object per line on stdin and writes one per line on stdout (logs go to stderr).
It prints `{"ready":true,"ctx":N}` once the model is loaded.

- `{"op":"reset"}` clears the conversation. Reply: `{"ok":true}`.
- `{"user":TEXT, "system":TEXT, "max_tokens":N, "think":BOOL}` appends a user turn and generates the reply.
  `system` only counts on the first turn after a reset. Reply: `{"text":..., "finished":BOOL, "gen_tokens":N,
  "prefill_tokens":N, "gen_tps":X, "prefill_tps":X, "ctx_used":N, "ctx":N}`. `text` includes the reasoning, up to
  `</think>`; `finished` is false when the reply hit `max_tokens` or the context is full.

The engine keeps the conversation, so each request sends only the new turn.
