"""SiCode: a terminal coding assistant. Claude (the Claude API) is the architect: it talks to you, plans, hands
self-contained tasks to the local worker and reviews the result. Qwen3.8-Flash-Next in qwen.exe is the worker: it
reads, edits and runs things in the current directory with its own tool loop.

A request goes to the local model unless it starts with [c]: "[c] task" is handled by Claude (which hands the
hands-on work to the local model), "[l] task" or a plain "task" by the local model alone.

usage: sicode [--yes] [--dir PATH] [--model claude-opus-5-5] [--effort high] [-p "[c] request"]
"""
import argparse
import os
import subprocess
import sys
import time

from .tools import TOOLS, Workspace, clip
from .worker import DEFAULT_EXE, DEFAULT_MODEL, Engine, run_task

DIM, BOLD, CYAN, YELLOW, RED, GREEN, OFF = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[32m", "\033[0m"

ARCHITECT_PROMPT = """You are the architect in SiCode, a terminal coding assistant. The user works in a repository \
(the workspace) and talks to you. You decide what should be done; a local model does the hands-on work.

The worker is Qwen3.8-Flash-Next (4-bit) running on the user's own GPU. It is free to run but slow (about 35 tokens \
per second) and less capable than you. You reach it with the `delegate` tool: each call starts a fresh worker with \
no memory of earlier tasks or of this conversation, and it works alone with read, write, edit, search and shell \
tools until it reports back. Its report describes what it did; treat that as a claim, not as proof.

How to split the work:
- The user pays for your tokens and not for the worker's, so the worker does the reading, the editing and the \
routine checks. Give it tasks of exploration too ("find where X is handled and report the functions involved").
- Write each task so that someone who has seen nothing can do it: the goal and why it matters, the files or areas \
involved if you know them, the constraints, what done looks like and how to check it, and what to report back. \
Decide the design yourself; do not leave the hard decision to the worker.
- Size tasks so one worker can finish them: one coherent change at a time, in sequence when later tasks depend on \
earlier ones.
- After a task that changes code, look at the diff that `delegate` returns, and read the code or run the tests \
yourself when the change is risky or the report is vague. If the work is wrong, send a follow-up task that says \
exactly what to fix.

Your own tools (`read_file`, `list_files`, `search`, `run_command`) are for deciding and verifying. You cannot edit \
files; every change goes through the worker.

Answer questions that need no work on the repository directly. Tell the user plainly what was done, what was \
verified and how, and what is still open. Keep your replies short: the user reads them in a terminal."""

DELEGATE = {
    "name": "delegate",
    "description": "Hand one self-contained task to the local worker model and wait for it to finish. Returns the "
                   "worker's report, the files it changed and the diff. The worker starts with no context: the task "
                   "text is everything it knows.",
    "input_schema": {"type": "object", "additionalProperties": False, "required": ["task"],
                     "properties": {"task": {"type": "string", "description": "The complete brief for the worker."}}},
}
ARCHITECT_TOOLS = [DELEGATE] + [t for t in TOOLS if t["name"] in ("read_file", "list_files", "search", "run_command")]


def valid_input(tool, args):
    """Checks a tool input against its schema (eager input streaming does not validate on the server)."""
    schema = tool["input_schema"]
    if not isinstance(args, dict) or any(k not in args for k in schema["required"]):
        return False
    types = {"string": str, "integer": int}
    return all(k in schema["properties"] and isinstance(v, types[schema["properties"][k]["type"]])
               and not isinstance(v, bool) for k, v in args.items())


def short(args):
    text = ", ".join(f"{k}={str(v).splitlines()[0][:70] if str(v) else ''!r}" for k, v in args.items()
                     if k not in ("content", "new_text", "old_text"))
    return text[:160]


class App:
    def __init__(self, a):
        self.a = a
        self.ws = Workspace(a.dir, self.approve)
        self.always = a.yes
        self.engine = None
        self.messages = []
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "worker_gen": 0}
        self.client = None      # the Claude API client, created by the first [c] request
        self.anthropic = None
        self.local_open = False  # the engine holds a local conversation that a follow-up can continue

    def approve(self, command):
        if self.always:
            return True
        print(f"{YELLOW}  run: {command}{OFF}")
        answer = input("  allow? [y]es / [a]lways / [n]o: ").strip().lower()
        if answer.startswith("a"):
            self.always = True
        return answer[:1] in ("y", "a")

    def start_engine(self):
        if self.engine is None:
            print(f"{DIM}loading Qwen3.8-Flash-Next ...{OFF}", flush=True)
            t0 = time.time()
            self.engine = Engine(self.a.exe, self.a.worker_model, self.a.backend)
            print(f"{DIM}worker ready in {time.time() - t0:.0f} s (context {self.engine.ctx:,} tokens){OFF}")
        return self.engine

    def git_diff(self):
        try:
            r = subprocess.run(["git", "diff", "--stat", "--patch", "--", *sorted(self.ws.changed)], cwd=self.ws.root,
                               capture_output=True, encoding="utf-8", errors="replace", timeout=60)
            return r.stdout if r.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    def delegate(self, task):
        engine = self.start_engine()
        self.ws.changed.clear()
        print(f"{CYAN}  worker task:{OFF} {DIM}{task.strip().splitlines()[0][:150]}{OFF}")

        def on_event(kind, data):
            if kind == "call":
                print(f"{DIM}    {data[0]}({short(data[1])}){OFF}", flush=True)
            elif kind == "result" and data.startswith("error"):
                print(f"{DIM}      {data.splitlines()[0][:150]}{OFF}")
            elif kind == "stats":
                self.usage["worker_gen"] += data["gen_tokens"]

        t0 = time.time()
        self.local_open = False  # the engine now holds this task, not the user's local conversation
        report, stats = run_task(engine, self.ws, task, on_event, effort=self.a.worker_effort)
        print(f"{DIM}  worker done: {stats['steps']} steps, {stats['gen_tokens']:,} tokens written, "
              f"{stats['ctx_used']:,} tokens of context, {time.time() - t0:.0f} s{OFF}")
        changed = sorted(self.ws.changed)
        out = "WORKER REPORT\n" + report + "\n\nFILES CHANGED\n" + ("\n".join(changed) if changed else "none")
        diff = self.git_diff() if changed else ""
        if diff:
            out += "\n\nDIFF (git, against the last commit; new untracked files are not shown)\n" + clip(diff, 12000)
        return out

    # --- [l]: the user's request goes straight to the local model; follow-ups continue its conversation ---

    def local_turn(self, text):
        engine = self.start_engine()

        def on_event(kind, data):
            if kind == "call":
                print(f"{DIM}  {data[0]}({short(data[1])}){OFF}", flush=True)
            elif kind == "result" and data.startswith("error"):
                print(f"{DIM}    {data.splitlines()[0][:150]}{OFF}")
        t0 = time.time()
        fresh, self.local_open = not self.local_open, False  # stays closed if the turn is interrupted
        report, stats = run_task(engine, self.ws, text, on_event, effort=self.a.worker_effort, fresh=fresh)
        self.local_open = True
        print("\n" + report)
        print(f"{DIM}[{stats['steps']} steps, {stats['gen_tokens']:,} tokens written, {time.time() - t0:.0f} s]{OFF}")

    # --- [c]: Claude is the architect and delegates to the local model ---

    def run_tool(self, block):
        tool = next((t for t in ARCHITECT_TOOLS if t["name"] == block.name), None)
        if tool is None or not valid_input(tool, block.input):
            return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                    "content": "INVALID_JSON: the tool input did not match the schema; send the call again."}
        if block.name == "delegate":
            result = self.delegate(block.input["task"])
        else:
            print(f"{DIM}  {block.name}({short(block.input)}){OFF}", flush=True)
            result = self.ws.call(block.name, block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": result}

    def architect_turn(self, text):
        self.messages.append({"role": "user", "content": text})
        tools = [{**t, "eager_input_streaming": True} for t in ARCHITECT_TOOLS]
        while True:
            with self.client.beta.messages.stream(
                model=self.a.model, max_tokens=32000, system=ARCHITECT_PROMPT, tools=tools, messages=self.messages,
                thinking={"type": "adaptive"}, output_config={"effort": self.a.effort},
                cache_control={"type": "ephemeral"},
                # a request the safety classifiers decline is re-run on a fallback model in the same call
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            ) as stream:
                wrote = False
                for piece in stream.text_stream:
                    print(piece, end="", flush=True)
                    wrote = True
                if wrote:
                    print()
                msg = stream.get_final_message()
            u = msg.usage
            self.usage["in"] += u.input_tokens + (u.cache_creation_input_tokens or 0)
            self.usage["cache_read"] += u.cache_read_input_tokens or 0
            self.usage["out"] += u.output_tokens
            self.messages.append({"role": "assistant", "content": msg.content})
            if msg.stop_reason == "pause_turn":
                continue
            if msg.stop_reason == "refusal":
                print(f"{RED}[Claude declined this request]{OFF}")
                return
            if msg.stop_reason == "max_tokens":
                print(f"{RED}[Claude's reply hit the length limit]{OFF}")
                return
            calls = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason != "tool_use" or not calls:
                return
            self.messages.append({"role": "user", "content": [self.run_tool(b) for b in calls]})

    def claude_ready(self):
        if self.client is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    or os.path.isdir(os.path.expanduser("~/.config/anthropic"))):
                print(f"{RED}No Claude API credentials: set ANTHROPIC_API_KEY to use [c]. "
                      f"Requests without [c] run on the local model.{OFF}")
                return False
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic()
        return True

    def turn(self, text):
        """Routes one request: "[c] ..." to Claude, "[l] ..." or no prefix to the local model."""
        tag, rest = text[:3].lower(), text[3:].strip()
        if tag not in ("[c]", "[l]"):
            tag, rest = "[l]", text
        if not rest:
            return
        if tag == "[l]":
            return self.local_turn(rest)
        if not self.claude_ready():
            return
        text = rest
        mark = len(self.messages)
        an = self.anthropic
        try:
            self.architect_turn(text)
        except KeyboardInterrupt:
            del self.messages[mark:]
            print(f"\n{YELLOW}[interrupted; that request was dropped from the conversation]{OFF}")
        except an.AuthenticationError:
            del self.messages[mark:]
            print(f"{RED}The Claude API rejected the credentials. Check ANTHROPIC_API_KEY.{OFF}")
        except an.RateLimitError:
            del self.messages[mark:]
            print(f"{RED}Rate limited by the Claude API; try again in a minute.{OFF}")
        except an.APIStatusError as e:
            del self.messages[mark:]
            print(f"{RED}Claude API error {e.status_code}: {e.message}{OFF}")
        except an.APIConnectionError:
            del self.messages[mark:]
            print(f"{RED}Could not reach the Claude API (network).{OFF}")
        u = self.usage
        print(f"{DIM}[Claude: {u['in']:,} in, {u['cache_read']:,} cached, {u['out']:,} out | "
              f"worker: {u['worker_gen']:,} tokens written]{OFF}")

    def repl(self):
        print(f"{BOLD}SiCode{OFF}  local Qwen3.8-Flash-Next, Claude {self.a.model} on request\n"
              f"{DIM}workspace {self.ws.root}\n"
              f"[c] task  Claude plans and reviews, the local model does the work\n"
              f"[l] task  the local model alone (also the default without a prefix)\n"
              f"/clear starts new conversations, /exit quits{OFF}")
        while True:
            try:
                text = input(f"\n{GREEN}>{OFF} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break
            if text == "/clear":
                self.messages.clear()
                self.local_open = False
                print(f"{DIM}[conversation cleared]{OFF}")
                continue
            try:
                self.turn(text)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}[interrupted]{OFF}")
            except RuntimeError as e:
                print(f"{RED}{e}{OFF}")
        if self.engine:
            self.engine.close()


def main():
    ap = argparse.ArgumentParser(description="Requests run on the local Qwen3.8-Flash-Next; start one with [c] "
                                             "to have Claude plan and review it.")
    ap.add_argument("--dir", default=os.getcwd(), help="workspace (default: the current directory)")
    ap.add_argument("--yes", action="store_true", help="run shell commands without asking")
    ap.add_argument("--model", default="claude-opus-5-5", help="Claude model for [c] requests")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--worker-effort", default="low", choices=["low", "medium", "xhigh"],
                    help="reasoning effort of the local worker")
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--worker-model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="vulkan")
    ap.add_argument("-p", "--prompt", help="run one request and exit; may start with [c] or [l]")
    a = ap.parse_args()
    os.system("")  # turns on ANSI colours in the Windows console
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    app = App(a)
    if a.prompt:
        try:
            app.turn(a.prompt)
        finally:
            if app.engine:
                app.engine.close()
    else:
        app.repl()


if __name__ == "__main__":
    main()
