"""The local model: Qwen3.8-Flash-Next in qwen.exe --pipe, running its own tool loop."""
import json
import os
import re
import subprocess
import tempfile

from . import sysinfo
from .tools import TOOLS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME_MODELS = os.path.join(os.path.expanduser("~"), ".sicode", "models")  # where `sicode --download-model` puts it
MODEL_REPO, MODEL_DIR, MODEL_QUANT = "unsloth/Qwen3.8-Flash-Next-GGUF", "Qwen3.8-Flash-Next-GGUF", "UD-Q4_K_XL"


def _first(paths):
    return next((p for p in paths if os.path.exists(p)), paths[-1])


# The engine: SICODE_ENGINE, a build in a checkout of the repository, or the one shipped inside the wheel.
DEFAULT_EXE = os.environ.get("SICODE_ENGINE") or _first([
    os.path.join(REPO, "build", "Release", "qwen.exe"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "qwen.exe")])
# The model: SICODE_MODEL, the checkout's models folder, or the download folder in the user's home.
DEFAULT_MODEL = os.environ.get("SICODE_MODEL") or _first([
    os.path.join(REPO, "models", MODEL_DIR, MODEL_QUANT),
    os.path.join(HOME_MODELS, MODEL_DIR, MODEL_QUANT)])


def local_models():
    """The model folders found next to the engine's checkout and in ~/.sicode/models: [{path, label}].

    A GGUF quantisation is a folder of .gguf files (models/<name>/<quant>); an FP8 model is a folder named *-FP8
    with safetensors. The 4-bit Flash-Next, the default, comes first.
    """
    found = {}
    for base in (os.path.join(REPO, "models"), HOME_MODELS):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if not os.path.isdir(d):
                continue
            try:
                for sub in sorted(os.listdir(d)):
                    q = os.path.join(d, sub)
                    if os.path.isdir(q) and any(f.endswith(".gguf") for f in os.listdir(q)):
                        found.setdefault(os.path.normcase(q), {"path": q, "label": f"{name.removesuffix('-GGUF')} ({sub}, GGUF)"})
                if name.upper().endswith("-FP8") and any(f.endswith(".safetensors") for f in os.listdir(d)):
                    found.setdefault(os.path.normcase(d), {"path": d, "label": f"{name.removesuffix('-FP8')} (FP8)"})
            except OSError:
                continue
    out = list(found.values())
    key = os.path.normcase(DEFAULT_MODEL)
    if os.path.exists(DEFAULT_MODEL) and key not in found:
        out.append({"path": DEFAULT_MODEL, "label": os.path.basename(os.path.dirname(DEFAULT_MODEL)) or DEFAULT_MODEL})
    for m in out:
        m["default"] = os.path.normcase(m["path"]) == key
        if m["default"]:
            m["label"] = m["label"].replace("UD-Q4_K_XL, GGUF", "4-bit, UD-Q4_K_XL")
    out.sort(key=lambda m: (not m["default"], m["label"].lower()))
    return out


def download_model(dst=None):
    """Downloads Qwen3.8-Flash-Next UD-Q4_K_XL (about 111 GB) and its draft head (4 GB) from Hugging Face.

    Re-running resumes and skips files that are already complete. Returns the model folder.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    dst = dst or os.path.join(HOME_MODELS, MODEL_DIR)
    print(f"{MODEL_REPO} {MODEL_QUANT} -> {dst}")
    snapshot_download(repo_id=MODEL_REPO, local_dir=dst, allow_patterns=[MODEL_QUANT + "/*", "mtp-*.gguf"], max_workers=8)
    return os.path.join(dst, MODEL_QUANT)

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

# For a task delegated by Claude: the reply without a tool call is a report to someone who saw none of the work.
WORKER_PROMPT = """You are a software engineer working in the repository at the workspace root. You receive one task \
from the lead engineer and carry it out yourself with the tools: read the relevant files before changing them, make \
the changes, and check your work (build or run the tests when the task or the repository gives you a way to).

Stay inside the task: do not refactor or fix things you were not asked about; mention them in the report instead.

When the task is done, or you cannot finish it, reply without a tool call. That reply is your report to the lead \
engineer, who has not seen your work: say what you changed (files and functions), what you ran and its result, and \
anything that is unfinished or that you are unsure about."""

# For a request typed by the user: an interactive coding assistant in their terminal.
DIRECT_PROMPT = """You are SiCode, a coding assistant working in the user's terminal, in the repository at the \
workspace root. The user types requests and you carry them out yourself with the tools.

- Answer a question that needs no look at the repository directly. Otherwise read the relevant files before you \
answer or change anything; do not guess at code you have not read.
- Make the changes the request asks for and no others. Check your work when there is a way to (build, run the tests).
- For a job with several steps, call update_plan first and again as steps finish.
- The user sees each tool call and each file change as it happens, and may decline one; if they do, do not retry \
it, ask what they want instead.
- A reply without a tool call ends your turn. Keep it short, since it is read in a terminal: what you did or found, \
what you ran and its result, and anything unfinished. No headings for a short answer."""


def system_prompt(effort, prompt, instructions=""):
    lines = [json.dumps({"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                          "parameters": t["input_schema"]}}) for t in TOOLS]
    head = EFFORT[effort] + "\n\n" if EFFORT[effort] else ""
    text = head + "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + "\n".join(lines) + \
        "\n</tools>" + TOOL_FORMAT + "\n\n" + prompt
    if instructions:
        text += "\n\n# Project instructions\n\nThe repository's own notes for assistants:\n\n" + instructions
    return text


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
    """qwen.exe --pipe: one JSON request per line in, JSON lines out. The engine holds the conversation."""

    def __init__(self, exe=DEFAULT_EXE, model=DEFAULT_MODEL, backend="vulkan", extra=(), vram_cap_gb=0):
        """vram_cap_gb: the most VRAM that may be in use on the card, counting other programs (0: no cap)."""
        if not os.path.exists(exe):
            raise RuntimeError(f"{exe} not found: build the engine (setup.bat) or set SICODE_ENGINE")
        if not os.path.exists(model):
            raise RuntimeError(f"the model is not at {model}: run `sicode --download-model` (about 115 GB) "
                               "or set SICODE_MODEL")
        tmp = tempfile.gettempdir()
        self.log_path = os.path.join(tmp, "sicode-engine.log")
        self.stop_path = os.path.join(tmp, f"sicode-stop-{os.getpid()}")  # its existence interrupts generation
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        # its own process group: Ctrl+C in the console interrupts the assistant, not the loaded model
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        env = dict(os.environ)
        self.vram_limit_mb = None
        if vram_cap_gb and vram_cap_gb > 0:
            # the engine is told how much it may take itself: the cap less what other programs hold right now
            now = sysinfo.vram()
            others = (now[0] >> 20) if now else 0
            self.vram_limit_mb = max(0, int(vram_cap_gb * 1024) - others)
            env["QW_VRAM_LIMIT_MB"] = str(self.vram_limit_mb)
        self.proc = subprocess.Popen([os.path.abspath(exe), "--pipe", "--model", os.path.abspath(model), "--backend", backend, *extra],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     creationflags=flags, env=env)
        self.ctx = self._read()["ctx"]
        self.ctx_used = 0
        self.cache = {"hits": 0, "misses": 0, "last": None, "cached": 0, "experts": 0}

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            self.log.flush()
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-1500:]
            raise RuntimeError(f"the engine stopped (exit code {self.proc.wait()}):\n{tail}")
        reply = json.loads(line.decode("utf-8", errors="replace"))
        if "error" in reply:
            raise RuntimeError("engine: " + reply["error"])
        return reply

    def _write(self, obj):
        self.proc.stdin.write(json.dumps(obj).encode("ascii") + b"\n")
        self.proc.stdin.flush()

    def reset(self):
        self._write({"op": "reset"})
        self._read()
        self.ctx_used = 0

    def turn(self, user, system=None, max_tokens=8192, think=True, on_delta=None):
        """Appends a user turn and returns the engine's reply. on_delta(text) receives the reply as it is written.

        Ctrl+C stops the generation: the engine closes the partial reply, then KeyboardInterrupt is raised.
        """
        req = {"user": user, "max_tokens": max_tokens, "think": think, "stream": True,  # always: the delta lines are also what lets Ctrl+C be noticed
               "interrupt_file": self.stop_path}
        if system:
            req["system"] = system
        if os.path.exists(self.stop_path):  # left by an interrupt() that came after its reply had ended
            os.remove(self.stop_path)
        self._write(req)
        interrupted = False
        while True:
            # the final reply is always read, also when interrupted: it must not be left in the pipe
            try:
                msg = self._read()
                if "delta" in msg:
                    if on_delta and not interrupted:
                        on_delta(msg["delta"])
                    continue
            except KeyboardInterrupt:  # Ctrl+C, or raised by on_delta
                interrupted = True
                open(self.stop_path, "w").close()
                continue
            break
        self.ctx_used = msg.get("ctx_used", self.ctx_used)
        hits, misses = msg.get("expert_hits", 0), msg.get("expert_misses", 0)
        if hits + misses:  # the VRAM expert cache: lookups it served in this reply, and since the start
            dh, dm = hits - self.cache["hits"], misses - self.cache["misses"]
            self.cache = {"hits": hits, "misses": misses, "last": dh / (dh + dm) if dh + dm else self.cache["last"],
                          "cached": msg.get("experts_cached", 0), "experts": msg.get("experts", 0)}
        if interrupted or msg.get("interrupted"):  # Ctrl+C here, or interrupt() from another thread
            if os.path.exists(self.stop_path):
                os.remove(self.stop_path)
            raise KeyboardInterrupt
        return msg

    def interrupt(self):
        """Stops the reply being generated, from another thread: turn() then raises KeyboardInterrupt there."""
        open(self.stop_path, "w").close()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def run_task(engine, workspace, task, on_event=lambda kind, data: None, effort="low", max_steps=60, fresh=True,
             prompt=WORKER_PROMPT, instructions="", stream=False):
    """Runs one request to completion on the local model. Returns (final reply, stats).

    fresh=False continues the conversation the engine already holds (a follow-up) instead of starting a new one.
    on_event(kind, data) reports progress: ("delta", text) while a reply is written (with stream), ("reply", None)
    when one is complete, ("call", (name, args)), ("result", text), ("stats", engine reply).
    """
    if fresh:
        engine.reset()
    stats = {"steps": 0, "gen_tokens": 0, "prefill_tokens": 0, "ctx_used": 0, "gen_s": 0.0}
    user, system = task, system_prompt(effort, prompt, instructions)
    on_delta = (lambda text: on_event("delta", text)) if stream else None
    answer = ""
    for step in range(max_steps):
        reply = engine.turn(user, system=system if step == 0 and fresh else None, on_delta=on_delta)
        stats["steps"] += 1
        stats["gen_tokens"] += reply["gen_tokens"]
        stats["prefill_tokens"] += reply["prefill_tokens"]
        stats["ctx_used"] = reply["ctx_used"]
        stats["gen_s"] += reply["gen_tokens"] / max(reply["gen_tps"], 1e-9)
        on_event("stats", reply)
        on_event("reply", None)
        _, answer, calls = parse_reply(reply["text"])
        if reply["ctx_used"] > reply["ctx"] - 12000:
            return (answer + "\n\n[stopped: the local model's context is full; the request is not finished]").strip(), stats
        if not calls:
            if not reply["finished"]:  # cut off by the token limit, mid-thought or mid-call
                user = "Your last reply was cut off at the length limit. Continue; write long files in several edits."
                continue
            return answer or "[the local model returned an empty reply]", stats
        results = []
        for name, args in calls:
            on_event("call", (name, args))
            # file contents must not be read as chat-template control tokens
            result = workspace.call(name, args).replace("<|im_", "<| im_")
            on_event("result", result)
            results.append(f"<tool_response>\n{result}\n</tool_response>")
        user = "\n".join(results)
    return f"[stopped after {max_steps} steps without finishing]\n{answer}", stats
