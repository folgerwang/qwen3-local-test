"""SiCode in the browser: tasks, a working folder, live progress, approvals, changed files.

`sicode --web` serves a single page on 127.0.0.1 and runs the same two layers as the terminal: requests go to the
local model, or to Claude as architect when the task is set to "Claude plans". The page talks to this server over
JSON and one server-sent event stream; every request needs the token from the URL the server prints.
"""
import difflib
import json
import os
import queue
import secrets
import string
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__, planners, sysinfo
from .cli import (ARCHITECT_PROMPT, ARCHITECT_TOOLS, DELEGATE, SOLO_API_PROMPT, COMPACT_REQUEST, FALLBACK_MODELS, INIT_REQUEST, INSTRUCTION_FILES,
                  MODEL_ALIASES, MODES, vram_cap, have_credentials, load_config, load_saved_key, save_config, short, valid_input)
from .tools import SKIP_DIRS, TOOLS, Workspace, clip
from .worker import DIRECT_PROMPT, WORKER_PROMPT, Engine, local_models, parse_reply, run_task

TASKS_DIR = os.path.join(os.path.expanduser("~"), ".sicode", "tasks")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "index.html")


class Splitter:
    """Splits a streamed local reply into reasoning and answer text, dropping the tool-call markup."""

    def __init__(self, on_think, on_answer):
        self.on_think, self.on_answer = on_think, on_answer
        self.buf, self.all, self.state, self.wrote = "", "", "think", False

    def _answer(self, text):
        if not self.wrote:
            text = text.lstrip("\n")
        if text:
            self.on_answer(text)
            self.wrote = True

    def feed(self, delta):
        self.buf += delta
        self.all += delta
        if self.state == "think":
            i = self.buf.find("</think>")
            if i < 0:
                keep = min(len(self.buf), 8)  # may be the start of </think>
                if len(self.buf) > keep:
                    self.on_think(self.buf[:len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                return
            if self.buf[:i]:
                self.on_think(self.buf[:i])
            self.buf, self.state = self.buf[i + 8:], "answer"
        if self.state == "answer":
            i = self.buf.find("<tool_call>")
            if i >= 0:
                self._answer(self.buf[:i].rstrip())
                self.buf, self.state = "", "call"
                return
            hold = max((k for k in range(1, 11) if self.buf.endswith("<tool_call>"[:k])), default=0)
            ready = self.buf[:len(self.buf) - hold].rstrip()  # trailing whitespace waits: it is dropped at the end
            self._answer(ready)
            self.buf = self.buf[len(ready):]

    def finish(self):
        """Returns True when the reply had no </think>: what was shown as reasoning was the answer."""
        if self.state == "answer":
            self._answer(self.buf.rstrip())
        elif self.state == "think" and self.buf:
            self.on_think(self.buf)
        return self.state == "think" and bool(self.all.strip())


def diff_lines(old, new, limit=400):
    if old is None:
        lines = ["+" + x for x in new.split("\n")]
        added, removed = len(lines), 0
    else:
        lines = list(difflib.unified_diff(old.replace("\r\n", "\n").split("\n"), new.replace("\r\n", "\n").split("\n"),
                                          lineterm="", n=3))[2:]
        added = sum(1 for x in lines if x.startswith("+"))
        removed = sum(1 for x in lines if x.startswith("-"))
    more = max(0, len(lines) - limit)
    return [x[:400] for x in lines[:limit]], added, removed, more


class Task:
    def __init__(self, workspace, tid=None):
        self.id = tid or uuid.uuid4().hex[:12]
        self.title = "New task"
        self.workspace = workspace
        self.created = time.time()
        self.items = []
        self.rev = 0
        self.summary = ""      # replaces the transcript as the seed of the next local conversation (compact)
        self.messages = []     # Claude's conversation for this task (kept in memory only)
        self.sessions = {}     # planner -> the session id of its command-line tool, to continue with
        self.undo = []         # (label, {path: content before}) per request that changed files
        self.changed = {}      # path -> content before the task first changed it (None: new file)

    def head(self):
        return {"id": self.id, "title": self.title, "workspace": self.workspace, "created": self.created}

    def save(self):
        os.makedirs(TASKS_DIR, exist_ok=True)
        with open(os.path.join(TASKS_DIR, self.id + ".json"), "w", encoding="utf-8") as f:
            json.dump({**self.head(), "items": self.items, "summary": self.summary, "sessions": self.sessions}, f)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        t = Task(d["workspace"], d["id"])
        t.title, t.created, t.items, t.summary = d["title"], d["created"], d["items"], d.get("summary", "")
        t.sessions = d.get("sessions") or {}
        for it in t.items:  # nothing is still running or waiting after a restart
            if it.get("status") in ("pending", "running"):
                it["status"] = "stopped"
        return t

    def transcript(self, limit=12000):
        """What was said so far, to seed a new conversation of the local model with."""
        parts = []
        for it in self.items:
            if it["type"] == "user":
                parts.append("User: " + it["text"])
            elif it["type"] == "assistant" and it.get("text"):
                parts.append("Assistant: " + it["text"])
            elif it["type"] == "diff" and it.get("status") == "applied":
                parts.append(f"[changed {it['path']}]")
            elif it["type"] == "command" and it.get("status") == "done":
                parts.append(f"[ran: {it['command'][:200]}]")
        text = "\n\n".join(parts)
        return text[-limit:]


class Web:
    def __init__(self, a):
        self.a = a
        self.mode = a.mode
        self.allowed = set()
        self.lock = threading.RLock()
        self.subs = []            # event queues of the connected pages
        self.tasks = {}
        self.engine = None
        self.engine_task = None   # the task whose conversation the engine holds
        self.local_open = False
        self.busy = None          # id of the task being worked on
        self.stop = False
        self.waiting = {}         # approval item id -> [threading.Event, answer]
        self.client = self.anthropic = None
        self.usage = {"in": 0, "out": 0, "cache_read": 0, "local": 0}
        self.task = self.ws = None  # the task being worked on and its workspace
        self.cmd_item = None
        self.sub = False          # inside a delegated task
        self.by = ""              # hybrid: who is acting now, "c" (the cloud model) or "l" (the local model)
        self.planner = a.planner if a.planner != "auto" else load_config().get("planner") or \
            planners.default_planner(have_credentials())
        self.planner_chat = None  # the Claude Code or Codex run in progress
        self.load_tasks()

    def load_tasks(self):
        if not os.path.isdir(TASKS_DIR):
            return
        for name in os.listdir(TASKS_DIR):
            if name.endswith(".json"):
                try:
                    t = Task.load(os.path.join(TASKS_DIR, name))
                    self.tasks[t.id] = t
                except (OSError, ValueError, KeyError):
                    continue

    # --- events to the page ---

    def send(self, event):
        data = json.dumps(event)
        with self.lock:
            for q in self.subs:
                q.put(data)

    def state(self):
        e = self.engine
        return {"version": __version__, "mode": self.mode, "modes": MODES, "model": self.a.model, "effort": self.a.effort,
                "busy": self.busy, "key": have_credentials(), "planner": self.planner, "vram_cap": vram_cap(self.a),
                "local_model": self.a.worker_model, "local_models": local_models(),
                "planners": [{"id": k, "label": v, "found": k == "api" or bool(planners.find(k))}
                             for k, v in planners.PLANNERS.items()], "default_workspace": os.path.realpath(self.a.dir),
                "engine": bool(e), "ctx": e.ctx if e else 0,
                "ctx_used": e.ctx_used if e and self.local_open else 0, "ctx_task": self.engine_task,
                "usage": self.usage,
                "tasks": [t.head() for t in sorted(self.tasks.values(), key=lambda t: -t.created)]}

    def sys(self):
        """Memory and expert-cache figures for the bar at the top of the page."""
        e = self.engine
        c = e.cache if e else None
        return {"ram": sysinfo.ram(), "vram": sysinfo.vram(),
                "engine_ram": sysinfo.process_ram(e.proc.pid) if e else None,
                "cache": {"last": c["last"], "all": c["hits"] / (c["hits"] + c["misses"]), "cached": c["cached"],
                          "experts": c["experts"]} if c and c["hits"] + c["misses"] else None}

    def tick(self):
        """Sends the memory figures to the open pages every two seconds."""
        while True:
            time.sleep(2)
            if self.subs:
                try:
                    self.send({"op": "sys", "sys": self.sys()})
                except Exception:  # a figure that cannot be read must not end the updates
                    pass

    def push_state(self):
        self.send({"op": "state", "state": self.state()})

    def add(self, **item):
        t = self.task
        with self.lock:
            item["id"] = len(t.items)
            if self.sub:
                item["sub"] = True
            if self.by and item["type"] in ("diff", "command", "tool", "delegate"):
                item["by"] = self.by
            t.items.append(item)
            t.rev += 1
            self.send({"op": "add", "task": t.id, "rev": t.rev, "item": item})
        return item

    def append(self, item, text):
        t = self.task
        with self.lock:
            item["text"] = item.get("text", "") + text
            t.rev += 1
            self.send({"op": "append", "task": t.id, "rev": t.rev, "id": item["id"], "text": text})

    def update(self, item, **fields):
        t = self.task
        with self.lock:
            item.update(fields)
            t.rev += 1
            self.send({"op": "update", "task": t.id, "rev": t.rev, "id": item["id"], "fields": fields})

    # --- permissions ---

    def wait_answer(self, item):
        ev = [threading.Event(), "no"]
        self.waiting[item["id"]] = ev
        try:
            while not ev[0].wait(0.3):
                if self.stop:
                    raise KeyboardInterrupt
        finally:
            self.waiting.pop(item["id"], None)
        return ev[1]

    def approve_edit(self, path, old, new):
        lines, added, removed, more = diff_lines(old, new)
        item = self.add(type="diff", path=path, new_file=old is None, added=added, removed=removed, lines=lines,
                        more=more, status="pending" if self.mode == "ask" else "applied")
        if self.mode == "ask":
            answer = self.wait_answer(item)
            if answer == "always":
                self.mode = "edits"
                self.push_state()
            ok = answer in ("yes", "always")
            self.update(item, status="applied" if ok else "declined")
            if not ok:
                return False
        self.task.changed.setdefault(path, old)
        self.send({"op": "changed", "task": self.task.id, "files": sorted(self.task.changed)})
        return True

    def approve_command(self, command):
        words = command.split()
        first = words[0].lower() if words else ""
        ask = not (self.mode == "auto" or first in self.allowed)
        item = self.add(type="command", command=command, first=first, status="pending" if ask else "running")
        if ask:
            answer = self.wait_answer(item)
            if answer == "always":
                self.allowed.add(first)
            if answer not in ("yes", "always"):
                self.update(item, status="declined")
                return False
            self.update(item, status="running")
        self.cmd_item = item
        return True

    def show_plan(self, plan):
        steps = plan if isinstance(plan, list) else planners.parse_plan(plan)
        if steps:
            self.add(type="plan", steps=steps)

    def on_call(self, name, args):
        self.tool_item = None
        if name in ("write_file", "edit_file", "run_command", "update_plan"):
            return  # these show themselves: the diff, the command, the plan
        self.tool_item = self.add(type="tool", name=name, args=short(args), status="running")

    def on_result(self, result):
        if self.cmd_item is not None:
            head, _, body = result.partition("\n")
            self.update(self.cmd_item, status="done" if head.startswith("exit code") else "error",
                        exit=head, output=clip(body, 6000))
            self.cmd_item = None
        elif getattr(self, "tool_item", None) is not None:
            err = result.startswith("error")
            self.update(self.tool_item, status="error" if err else "done",
                        output=result.splitlines()[0][:300] if err else clip(result, 3000))
            self.tool_item = None

    # --- the local model ---

    def start_engine(self):
        if self.engine is None:
            name = next((m["label"] for m in local_models() if os.path.normcase(m["path"]) ==
                         os.path.normcase(self.a.worker_model)), os.path.basename(self.a.worker_model))
            note = self.add(type="note", text=f"Loading the local model {name} ...", status="running")
            t0 = time.time()
            self.engine = Engine(self.a.exe, self.a.worker_model, self.a.backend, vram_cap_gb=vram_cap(self.a))
            self.update(note, text=f"{name} ready in {time.time() - t0:.0f} s "
                                   f"(context {self.engine.ctx:,} tokens)", status="done")
        return self.engine

    def instructions(self):
        for name in INSTRUCTION_FILES:
            p = os.path.join(self.ws.root, name)
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    return name, clip(f.read(), 12000)
        return None, ""

    def run_local(self, text, prompt, fresh, seed_stream=True):
        """One request on the local model, streamed into the task. Returns (answer, stats)."""
        engine = self.start_engine()
        cur = {"think": None, "answer": None}

        def think(t):
            if cur["think"] is None:
                cur["think"] = self.add(type="think", text="", status="running")
            self.append(cur["think"], t)

        def answer(t):
            if cur["think"] is not None and cur["think"].get("status") == "running":
                self.update(cur["think"], status="done")
            if cur["answer"] is None:
                cur["answer"] = self.add(type="assistant", who="local", text="")
            self.append(cur["answer"], t)

        split = [Splitter(think, answer)]

        def on_event(kind, data):
            if self.stop:
                raise KeyboardInterrupt
            if kind == "delta":
                split[0].feed(data)
            elif kind == "reply":
                if split[0].finish() and cur["think"] is not None:  # no </think>: it was the answer all along
                    self.update(cur["think"], type="assistant", who="local", status="done")
                elif cur["think"] is not None:
                    self.update(cur["think"], status="done")
                cur["think"] = cur["answer"] = None
                split[0] = Splitter(think, answer)
            elif kind == "call":
                self.on_call(*data)
            elif kind == "result":
                self.on_result(data)
            elif kind == "stats":
                self.usage["local"] += data["gen_tokens"]
                self.push_state()

        try:
            return run_task(engine, self.ws, text, on_event, effort=self.a.worker_effort, fresh=fresh, prompt=prompt,
                            instructions=self.instructions()[1], stream=True)
        finally:
            for it in cur.values():
                if it is not None and it.get("status") == "running":
                    self.update(it, status="done")

    def local_turn(self, text, prior):
        t = self.task
        fresh = not (self.local_open and self.engine_task == t.id)
        if fresh:
            seed = t.summary or prior
            if seed:
                text = f"Summary of the conversation so far:\n{seed}\n\nThe user's next request:\n{text}"
        self.engine_task, self.local_open = t.id, True
        t0 = time.time()
        _, stats = self.run_local(text, DIRECT_PROMPT, fresh)
        speed = stats["gen_tokens"] / max(stats["gen_s"], 1e-9)
        self.add(type="stats", text=f"Local model: {stats['steps']} steps, {stats['gen_tokens']:,} tokens at "
                                    f"{speed:.0f} tok/s, {time.time() - t0:.0f} s")

    def compact(self):
        t = self.task
        if not (self.local_open and self.engine_task == t.id):
            self.add(type="note", text="Nothing to compact: this task has no open local conversation.")
            return
        note = self.add(type="note", text="Summarising the conversation ...", status="running")
        reply = self.engine.turn(COMPACT_REQUEST)
        t.summary = parse_reply(reply["text"])[1]
        self.local_open = False
        self.update(note, status="done", text=f"Compacted: {reply['ctx_used']:,} tokens of context replaced by a "
                                              f"summary of {len(t.summary.split()):,} words.")

    # --- Claude as architect ---

    def delegate(self, brief):
        self.ws.changed.clear()
        by, self.by = self.by, "l" if self.by else ""
        item = self.add(type="delegate", text=brief, status="running")
        self.engine_task, self.local_open = None, False  # the engine now holds this brief
        self.sub = True
        t0 = time.time()
        try:
            report, stats = self.run_local(brief, WORKER_PROMPT, True)
        finally:
            self.sub = False
            self.by = by
        self.update(item, status="done", stats=f"{stats['steps']} steps, {stats['gen_tokens']:,} tokens, "
                                               f"{time.time() - t0:.0f} s")
        changed = sorted(self.ws.changed)
        return "WORKER REPORT\n" + report + "\n\nFILES CHANGED\n" + ("\n".join(changed) if changed else "none")

    def run_tool(self, block):
        tool = next((x for x in [DELEGATE] + TOOLS if x["name"] == block.name), None)
        if tool is None or not valid_input(tool, block.input):
            return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                    "content": "INVALID_JSON: the tool input did not match the schema; send the call again."}
        if block.name == "delegate":
            result = self.delegate(block.input["task"])
        else:
            self.on_call(block.name, block.input)
            result = self.ws.call(block.name, block.input)
            self.on_result(result)
        return {"type": "tool_result", "tool_use_id": block.id, "content": result}

    def claude_turn(self, text, solo=False):
        t = self.task
        if self.client is None:
            if not have_credentials():
                self.add(type="error", text="No Claude API key is set. Add one in Settings, or switch this request "
                                            "to the local model.")
                return
            import anthropic
            self.anthropic, self.client = anthropic, anthropic.Anthropic()
        an = self.anthropic
        mark = len(t.messages)
        t.messages.append({"role": "user", "content": text})
        name, notes = self.instructions()
        system = (SOLO_API_PROMPT if solo else ARCHITECT_PROMPT) + \
            (f"\n\n# Project instructions ({name})\n\n{notes}" if notes else "")
        tools = [{**x, "eager_input_streaming": True} for x in (TOOLS if solo else ARCHITECT_TOOLS)]
        extra = {}
        if self.a.model in FALLBACK_MODELS:
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            while True:
                item = None
                with self.client.beta.messages.stream(
                    model=self.a.model, max_tokens=32000, system=system, tools=tools, messages=t.messages,
                    thinking={"type": "adaptive"}, output_config={"effort": self.a.effort},
                    cache_control={"type": "ephemeral"}, **extra,
                ) as stream:
                    for piece in stream.text_stream:
                        if self.stop:
                            raise KeyboardInterrupt
                        if item is None:
                            item = self.add(type="assistant", who="claude", text="")
                        self.append(item, piece)
                    msg = stream.get_final_message()
                u = msg.usage
                self.usage["in"] += u.input_tokens + (u.cache_creation_input_tokens or 0)
                self.usage["cache_read"] += u.cache_read_input_tokens or 0
                self.usage["out"] += u.output_tokens
                self.push_state()
                t.messages.append({"role": "assistant", "content": msg.content})
                if msg.stop_reason == "pause_turn":
                    continue
                if msg.stop_reason == "refusal":
                    self.add(type="error", text="Claude declined this request.")
                    return
                if msg.stop_reason == "max_tokens":
                    self.add(type="error", text="Claude's reply hit the length limit.")
                    return
                calls = [b for b in msg.content if b.type == "tool_use"]
                if msg.stop_reason != "tool_use" or not calls:
                    return
                t.messages.append({"role": "user", "content": [self.run_tool(b) for b in calls]})
        except KeyboardInterrupt:
            del t.messages[mark:]
            raise
        except an.AuthenticationError:
            del t.messages[mark:]
            self.add(type="error", text="The Claude API rejected the key. Set a valid one in Settings.")
        except an.RateLimitError:
            del t.messages[mark:]
            self.add(type="error", text="Rate limited by the Claude API; try again in a minute.")
        except an.APIStatusError as e:
            del t.messages[mark:]
            self.add(type="error", text=f"Claude API error {e.status_code}: {e.message}")
        except an.APIConnectionError:
            del t.messages[mark:]
            self.add(type="error", text="Could not reach the Claude API (network).")

    def login_turn(self, text, solo=False):
        """Claude Code or Codex, signed in with the user's account: hybrid, or working alone (cloud)."""
        t = self.task
        label = planners.PLANNERS[self.planner]
        key = self.planner + (":solo" if solo else "")
        chat = planners.Planner(self.planner, t.workspace, session=t.sessions.get(key), solo=solo,
                                allow_shell=self.mode == "auto")
        self.planner_chat = chat
        who = "claude" if self.planner == "claude-code" else "codex"
        note = [None]
        snap = [None]

        def before_turn():  # the tool edits on its own: compare the workspace around each of its turns
            snap[0] = self.ws.snapshot()
            first = not any(it["type"] == "assistant" for it in t.items[-3:]) and note[0] is None
            verb = "working" if solo or first else "at work"
            note[0] = self.add(type="note", text=f"{label} is {verb} ...", status="running")

        def after_turn():
            if note[0] is not None:
                self.update(note[0], status="done", text=f"{label}")
                note[0] = None
            for path, old, new in self.ws.changes_since(snap[0]):
                lines, added, removed, more = diff_lines(old, new if new is not None else "")
                self.add(type="diff", path=path, new_file=old is None, deleted=new is None, added=added,
                         removed=removed, lines=lines, more=more, status="applied")
                t.changed.setdefault(path, old)
                self.send({"op": "changed", "task": t.id, "files": sorted(t.changed)})

        try:
            planners.run(chat, text, self.delegate, lambda reply: self.add(type="assistant", who=who, text=reply),
                         lambda: self.stop, on_plan=self.show_plan, before_turn=before_turn, after_turn=after_turn)
        finally:
            if chat.session:
                t.sessions[key] = chat.session
            self.planner_chat = None
            if note[0] is not None:
                self.update(note[0], status="done")

    # --- one request ---

    def attach(self, text):
        import re
        out = text
        for ref in dict.fromkeys(re.findall(r"(?<!\S)@([^\s@]+)", text)):
            try:
                p = self.ws.path(ref)
            except ValueError:
                continue
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    out += f"\n\n<file path=\"{self.ws.rel(p)}\">\n{clip(f.read())}\n</file>"
        return out

    def work(self, task, text, route):
        self.task, self.stop, self.sub, self.cmd_item = task, False, False, None
        self.by = "c" if route == "hybrid" else ""  # hybrid: every step is marked with who did it
        self.ws = Workspace(task.workspace, self.approve_command, self.approve_edit, self.show_plan)
        self.ws.begin_turn()
        try:
            prior = task.transcript()
            if text == "/compact":
                self.compact()
                return
            if text == "/init":
                text = INIT_REQUEST
                self.add(type="user", text="Write the project notes (SICODE.md)", route=route)
            else:
                self.add(type="user", text=text, route=route)
            if task.title == "New task":
                task.title = text.strip().splitlines()[0][:70]
            self.push_state()
            if route in ("hybrid", "cloud") and self.planner != "api":
                self.login_turn(self.attach(text), solo=route == "cloud")
            elif route in ("hybrid", "cloud"):
                self.claude_turn(self.attach(text), solo=route == "cloud")
            else:
                self.local_turn(self.attach(text), prior)
        except KeyboardInterrupt:
            self.add(type="note", text="Stopped.")
        except Exception as e:  # shown in the task instead of killing the worker thread
            self.add(type="error", text=str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")
        finally:
            for it in task.items:
                if it.get("status") in ("pending", "running"):
                    self.update(it, status="stopped")
            if self.ws.backups:
                task.undo.append((text.strip().splitlines()[0][:60], self.ws.backups))
            try:
                task.save()
            except OSError:
                pass
            self.busy = None
            self.push_state()

    # --- what the page asks for ---

    def api(self, path, q, body):
        if path == "/api/state":
            return self.state()
        if path == "/api/task":
            t = self.tasks.get(q.get("id", ""))
            if not t:
                return {"error": "no such task"}
            return {**t.head(), "rev": t.rev, "items": t.items, "changed": sorted(t.changed), "undo": len(t.undo)}
        if path == "/api/new":
            root = os.path.realpath(body.get("workspace") or self.a.dir)
            if not os.path.isdir(root):
                return {"error": f"{root} is not a folder"}
            t = Task(root)
            self.tasks[t.id] = t
            self.push_state()
            return t.head()
        if path == "/api/send":
            t = self.tasks.get(body.get("task", ""))
            text = (body.get("text") or "").strip()
            if not t or not text:
                return {"error": "nothing to send"}
            with self.lock:
                if self.busy:
                    return {"error": "another request is still running"}
                self.busy = t.id
            self.push_state()
            route = {"claude": "hybrid", "mix": "hybrid"}.get(body.get("route"), body.get("route"))  # "claude": pages before 0.2.1
            route = route if route in ("hybrid", "cloud") else "local"
            threading.Thread(target=self.work, args=(t, text, route), daemon=True).start()
            return {"ok": True}
        if path == "/api/stop":
            if self.busy:
                self.stop = True
                if self.engine:
                    self.engine.interrupt()
                if self.planner_chat:
                    self.planner_chat.stop()
            return {"ok": bool(self.busy)}
        if path == "/api/approve":
            ev = self.waiting.get(body.get("id"))
            if ev:
                ev[1] = body.get("answer", "no")
                ev[0].set()
            return {"ok": bool(ev)}
        if path == "/api/mode":
            if body.get("mode") in MODES:
                self.mode = body["mode"]
            self.push_state()
            return {"mode": self.mode}
        if path == "/api/model":
            if body.get("model"):
                self.a.model = MODEL_ALIASES.get(body["model"].lower(), body["model"])
            if body.get("effort") in ("low", "medium", "high", "xhigh", "max"):
                self.a.effort = body["effort"]
            self.push_state()
            return {"model": self.a.model}
        if path == "/api/localmodel":
            chosen = next((m for m in local_models() if m["path"] == body.get("path")), None)
            if not chosen:
                return {"error": "that model is not installed"}
            changed = os.path.normcase(chosen["path"]) != os.path.normcase(self.a.worker_model)
            with self.lock:
                if changed and self.busy:
                    return {"error": "a request is running; change the local model when it has finished"}
                self.a.worker_model = chosen["path"]
                if changed and self.engine:  # the next request loads the chosen model
                    self.engine.close()
                    self.engine, self.engine_task, self.local_open = None, None, False
            try:
                save_config(local_model=chosen["path"])
            except OSError:
                pass
            self.push_state()
            return {"path": chosen["path"], "changed": changed}
        if path == "/api/vramcap":
            try:
                gb = max(0.0, float(body.get("gb") or 0))
            except (TypeError, ValueError):
                return {"error": "not a number"}
            changed = gb != vram_cap(self.a)
            self.a.vram_cap = gb
            try:
                save_config(vram_cap_gb=gb)
            except OSError:
                pass
            reloaded = False
            with self.lock:
                if changed and self.engine and not self.busy:  # the cap is applied when the model loads
                    self.engine.close()
                    self.engine, self.engine_task, self.local_open = None, None, False
                    reloaded = True
            self.push_state()
            return {"gb": gb, "reloaded": reloaded, "pending": changed and bool(self.engine)}
        if path == "/api/planner":
            if body.get("planner") in planners.PLANNERS:
                self.planner = body["planner"]
                try:
                    save_config(planner=self.planner)
                except OSError:
                    pass
            self.push_state()
            return {"planner": self.planner}
        if path == "/api/key":
            key = (body.get("key") or "").strip()
            if not key:
                return {"error": "empty key"}
            os.environ["ANTHROPIC_API_KEY"] = key
            self.client = None
            try:
                save_config(anthropic_api_key=key)
            except OSError:
                pass
            self.push_state()
            return {"ok": True}
        if path == "/api/undo":
            t = self.tasks.get(body.get("task", ""))
            if not t or self.busy or not t.undo:
                return {"error": "nothing to undo"}
            label, backups = t.undo.pop()
            ws = Workspace(t.workspace, lambda c: False)
            files = ws.restore(backups)
            self.task = t
            self.add(type="note", text=f"Put back {', '.join(files)} (changed by: {label}).")
            t.save()
            return {"ok": True, "undo": len(t.undo)}
        if path == "/api/delete":
            t = self.tasks.get(body.get("task", ""))
            if not t or self.busy == t.id:
                return {"error": "cannot delete this task now"}
            del self.tasks[t.id]
            try:
                os.remove(os.path.join(TASKS_DIR, t.id + ".json"))
            except OSError:
                pass
            self.push_state()
            return {"ok": True}
        if path == "/api/tree":
            t = self.tasks.get(q.get("task", ""))
            root = t.workspace if t else os.path.realpath(self.a.dir)
            ws = Workspace(root, lambda c: False)
            d = ws.path(q.get("path", "") or ".")
            out = []
            for name in sorted(os.listdir(d), key=str.lower)[:600]:
                full = os.path.join(d, name)
                is_dir = os.path.isdir(full)
                if is_dir and name in SKIP_DIRS:
                    continue
                out.append({"name": name, "dir": is_dir, "path": ws.rel(full)})
            out.sort(key=lambda e: (not e["dir"], e["name"].lower()))
            return {"entries": out}
        if path == "/api/file":
            t = self.tasks.get(q.get("task", ""))
            ws = Workspace(t.workspace if t else self.a.dir, lambda c: False)
            p = ws.path(q.get("path", ""))
            if os.path.getsize(p) > 2_000_000:
                return {"path": ws.rel(p), "text": "[file larger than 2 MB]"}
            with open(p, "rb") as f:
                raw = f.read()
            if b"\0" in raw[:4000]:
                return {"path": ws.rel(p), "text": "[binary file]"}
            text = raw.decode("utf-8", errors="replace")
            before = t.changed.get(ws.rel(p), False) if t else False
            lines = diff_lines(before, text)[0] if before is not False else None
            return {"path": ws.rel(p), "text": text[:400000], "diff": lines}
        if path == "/api/dirs":
            p = q.get("path", "")
            if p == "::drives" or (not p and os.name != "nt" and False):
                drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.isdir(f"{c}:\\")]
                return {"path": "", "parent": None, "dirs": [{"name": d, "path": d} for d in drives]}
            p = os.path.realpath(p or self.a.dir)
            parent = os.path.dirname(p)
            if parent == p:
                parent = "::drives" if os.name == "nt" else None
            dirs = []
            try:
                for name in sorted(os.listdir(p), key=str.lower):
                    if not name.startswith(("$", ".")) and os.path.isdir(os.path.join(p, name)):
                        dirs.append({"name": name, "path": os.path.join(p, name)})
            except OSError as e:
                return {"path": p, "parent": parent, "dirs": [], "error": str(e)}
            files = []
            try:
                files = [n for n in sorted(os.listdir(p), key=str.lower) if os.path.isfile(os.path.join(p, n))]
            except OSError:
                pass
            return {"path": p, "parent": parent, "dirs": dirs[:800], "files": files[:200],
                    "more": max(0, len(files) - 200)}
        if path == "/api/pickdir":
            return pick_folder(body.get("start") or self.a.dir)
        return None


PICKER = """import sys, tkinter, tkinter.filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
print(tkinter.filedialog.askdirectory(initialdir=sys.argv[1], title='SiCode: choose the working folder', mustexist=True))
"""


def pick_folder(start):
    """Opens the system's folder dialog on this computer (the page runs on it too). {path} or {error}."""
    import subprocess
    import sys
    try:
        r = subprocess.run([sys.executable, "-c", PICKER, start if os.path.isdir(start) else os.path.expanduser("~")],
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"the folder dialog could not be opened: {e}"}
    chosen = r.stdout.strip()
    if r.returncode != 0:
        return {"error": "the folder dialog could not be opened (Python's tkinter is missing?)"}
    return {"path": os.path.realpath(chosen)} if chosen else {"path": ""}


def make_handler(web, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self, q):
            # only this machine's own page: the right Host (no DNS rebinding) and the token from the printed URL
            host = (self.headers.get("Host") or "").split(":")[0]
            given = self.headers.get("X-SiCode-Token") or q.get("t", "")
            return host in ("127.0.0.1", "localhost") and secrets.compare_digest(given, token)

        def handle_any(self, body):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    return self.reply(200, f.read(), "text/html; charset=utf-8")
            if not self.allowed(q):
                return self.reply(403, {"error": "open the address SiCode printed when it started (it carries the "
                                                 "access token)"})
            if u.path == "/api/events":
                return self.events()
            try:
                out = web.api(u.path, q, body)
            except (OSError, ValueError) as e:
                out = {"error": str(e)}
            if out is None:
                return self.reply(404, {"error": "not found"})
            self.reply(200, out)

        def events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            qu = queue.Queue()
            with web.lock:
                web.subs.append(qu)
                qu.put(json.dumps({"op": "state", "state": web.state()}))
                qu.put(json.dumps({"op": "sys", "sys": web.sys()}))
            try:
                while True:
                    try:
                        data = qu.get(timeout=15)
                        self.wfile.write(b"data: " + data.encode("utf-8") + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                with web.lock:
                    web.subs.remove(qu)

        def do_GET(self):
            self.handle_any({})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            self.handle_any(body if isinstance(body, dict) else {})

    return Handler


def serve(a):
    load_saved_key()
    cfg = load_config()
    token = cfg.get("web_token")
    if not token:
        token = secrets.token_urlsafe(18)
        try:
            save_config(web_token=token)
        except OSError:
            pass
    web = Web(a)
    threading.Thread(target=web.tick, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(web, token))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/?t={token}"
    print(f"SiCode {__version__} in the browser: {url}\nworkspace {os.path.realpath(a.dir)}; Ctrl+C here stops it")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if web.engine:
            web.engine.close()
