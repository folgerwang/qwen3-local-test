"""The local worker: Qwen3.8-Flash-Next in qwen.exe --pipe, running its own tool loop on one task at a time."""
import json
import os
import re
import subprocess
import tempfile

from .tools import TOOLS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXE = os.environ.get("SICODE_ENGINE") or os.path.join(REPO, "build", "Release", "qwen.exe")
DEFAULT_MODEL = os.environ.get("SICODE_MODEL") or os.path.join(REPO, "models", "Qwen3.8-Flash-Next-GGUF", "UD-Q4_K_XL")

# The system block the model's chat template (tokenizer.chat_template in the GGUF) renders for tool use.
EFFORT = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "medium": "",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
           "without unnecessary elaboration.",
}
TOOL_FORMAT = """

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

WORKER_PROMPT = """You are a software engineer working in the repository at the workspace root. You receive one task \
from the lead engineer and carry it out yourself with the tools: read the relevant files before changing them, make \
the changes, and check your work (build or run the tests when the task or the repository gives you a way to).

Stay inside the task: do not refactor or fix things you were not asked about; mention them in the report instead.

When the task is done, or you cannot finish it, reply without a tool call. That reply is your report to the lead \
engineer, who has not seen your work: say what you changed (files and functions), what you ran and its result, and \
anything that is unfinished or that you are unsure about."""


def system_prompt(effort):
    lines = [json.dumps({"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                          "parameters": t["input_schema"]}}) for t in TOOLS]
    head = EFFORT[effort] + "\n\n" if EFFORT[effort] else ""
    return head + "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + "\n".join(lines) + \
        "\n</tools>" + TOOL_FORMAT + "\n\n" + WORKER_PROMPT


CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
SCHEMAS = {t["name"]: t["input_schema"]["properties"] for t in TOOLS}


def parse_reply(text):
    """Splits a raw reply into (reasoning, answer text, [(tool name, args)])."""
    think, sep, rest = text.rpartition("</think>")
    if not sep:
        think, rest = "", text
    calls = []
    for name, body in CALL_RE.findall(rest):
        name = name.strip()
        args = {}
        for key, value in PARAM_RE.findall(body):
            key = key.strip()
            if SCHEMAS.get(name, {}).get(key, {}).get("type") == "integer":
                try:
                    value = int(value.strip())
                except ValueError:
                    pass
            args[key] = value
        calls.append((name, args))
    answer = rest.split("<tool_call>")[0].strip()
    return think.strip(), answer, calls


class Engine:
    """qwen.exe --pipe: one JSON request per line in, one JSON reply per line out. It holds the conversation."""

    def __init__(self, exe=DEFAULT_EXE, model=DEFAULT_MODEL, backend="vulkan", extra=()):
        if not os.path.exists(exe):
            raise RuntimeError(f"{exe} not found: build the engine (setup.bat) or set SICODE_ENGINE")
        self.log_path = os.path.join(tempfile.gettempdir(), "sicode-engine.log")
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        # its own process group: Ctrl+C in the console interrupts the assistant, not the loaded model
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen([exe, "--pipe", "--model", model, "--backend", backend, *extra], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log,
                                     creationflags=flags)
        self.pending = False  # a request was sent and its reply not read yet (an interrupted turn)
        ready = self._read()
        self.ctx = ready["ctx"]

    def _read(self):
        line = self.proc.stdout.readline()  # an interrupt here leaves `pending` set
        self.pending = False
        if not line:
            self.log.flush()
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-1500:]
            raise RuntimeError(f"the engine stopped (exit code {self.proc.wait()}):\n{tail}")
        reply = json.loads(line.decode("utf-8", errors="replace"))
        if "error" in reply:
            raise RuntimeError("engine: " + reply["error"])
        return reply

    def _send(self, obj):
        if self.pending:  # the engine is still finishing an interrupted turn: wait for it and drop the reply
            self.proc.stdout.readline()
        self.pending = True
        self.proc.stdin.write(json.dumps(obj).encode("ascii") + b"\n")
        self.proc.stdin.flush()
        return self._read()

    def reset(self):
        self._send({"op": "reset"})

    def turn(self, user, system=None, max_tokens=8192, think=True):
        req = {"user": user, "max_tokens": max_tokens, "think": think}
        if system:
            req["system"] = system
        return self._send(req)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def run_task(engine, workspace, task, on_event=lambda kind, data: None, effort="low", max_steps=60, fresh=True):
    """Runs one task to completion on the local model. Returns (report, stats).

    fresh=False continues the conversation the engine already holds (a follow-up request) instead of starting
    a new one.

    on_event(kind, data) reports progress: ("say", text), ("call", (name, args)), ("result", text), ("stats", reply).
    """
    if fresh:
        engine.reset()
    stats = {"steps": 0, "gen_tokens": 0, "prefill_tokens": 0, "ctx_used": 0}
    user, system = task, system_prompt(effort)
    for step in range(max_steps):
        reply = engine.turn(user, system=system if step == 0 and fresh else None)
        stats["steps"] += 1
        stats["gen_tokens"] += reply["gen_tokens"]
        stats["prefill_tokens"] += reply["prefill_tokens"]
        stats["ctx_used"] = reply["ctx_used"]
        on_event("stats", reply)
        _, answer, calls = parse_reply(reply["text"])
        if answer:
            on_event("say", answer)
        if reply["ctx_used"] > reply["ctx"] - 12000:
            return (answer + "\n\n[worker stopped: its context is full; the task is not finished]").strip(), stats
        if not calls:
            if not reply["finished"]:  # cut off by the token limit, mid-thought or mid-call
                user = "Your last reply was cut off at the length limit. Continue; write long files in several edits."
                continue
            return answer or "[the worker returned an empty report]", stats
        results = []
        for name, args in calls:
            on_event("call", (name, args))
            # file contents must not be read as chat-template control tokens
            result = workspace.call(name, args).replace("<|im_", "<| im_")
            on_event("result", result)
            results.append(f"<tool_response>\n{result}\n</tool_response>")
        user = "\n".join(results)
    return f"[worker stopped after {max_steps} steps without finishing]\n{answer}", stats
