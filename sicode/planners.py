"""Planners that use an account login instead of an API key.

The cloud model can be the vendor's own command-line tool, which signs in with the user's subscription: Claude Code
(`claude`, a Claude account) or Codex (`codex`, a ChatGPT account). SiCode never sees the account's credentials: it
runs the tool non-interactively in the workspace and reads its reply.

In the hybrid mode the tool splits the job into steps marked [l] (the local model) and [c] (itself). It cannot call
SiCode's tools, so the hand-off is done in text: it ends a reply with a <delegate> block, SiCode runs that task on
the local model and sends the report back as the next message of the same session. Its own steps it does with its
own file tools; SiCode compares the workspace before and after each of its turns to show what it changed.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time

PLANNERS = {
    "api": "Claude API key",
    "claude-code": "Claude account (through Claude Code)",
    "codex": "ChatGPT account (through Codex)",
}

SPLIT = """Two of you do the work, and every step of the job goes to one of you:

- [l] the local model: Qwen3.8-Flash-Next (4-bit) on the user's own GPU. Free to run, but slow (about 35 tokens per \
second) and clearly less capable than you. Each task you hand it starts a fresh worker with no memory of earlier \
tasks or of this conversation; it works alone with read, write, edit, search and shell tools until it reports back. \
Its report describes what it did; treat that as a claim, not as proof.
- [c] you. Your tokens cost the user money or quota, the worker's cost nothing.

Give a step to [l] when a careful but ordinary engineer could do it from a written brief: mechanical or repetitive \
edits, boilerplate, tests for behaviour you specify, exploration ("find where X is handled and report the functions \
involved"), running builds, tests and other commands, and anything long but well defined. Take a step as [c] when \
it needs your judgement or precision: the design, a subtle bug, a change that must be right across several files, \
anything where a wrong edit is costly, a final review, or an edit so small that writing the brief would cost more \
than doing it. When in doubt between the two, prefer [l] and review its result."""

PLAN_FORMAT = """one step per line, each starting with its state and who does it: "[ ] [l] step" or "[ ] [c] step". \
States: [ ] to do, [~] in progress, [x] done."""


def hybrid_prompt(can_run):
    """The system prompt for Claude Code or Codex in the hybrid mode."""
    run = ("You can also run shell commands yourself." if can_run else
           "You cannot run shell commands yourself: builds, tests and other commands are always [l] steps.")
    return f"""You are the lead engineer in SiCode, a coding assistant. The user works in a repository (your working \
directory) and talks to you.

{SPLIT}

For a job with more than one step, start with the plan and send it again, complete, whenever a state changes:

<plan>
[ ] [l] first step
[ ] [c] second step
</plan>

In a plan block write {PLAN_FORMAT}

To run an [l] step, end your reply with exactly one block of this form:

<delegate>
The complete brief for the worker.
</delegate>

SiCode then runs it and sends you the worker's report and the list of changed files as the next message. Write each \
brief so that someone who has seen nothing can do it: the goal and why it matters, the files or areas involved, the \
constraints, what done looks like and how to check it, and what to report back. One coherent change per brief.

To do a [c] step, do it yourself in this turn with your own tools: read, search, edit and write files in the \
working directory. {run} SiCode shows the user the files you changed.

A reply without a <delegate> block ends your turn and is shown to the user. Answer questions that need no work on \
the repository directly, without a plan. In your final reply tell the user plainly what was done and by whom, what \
was verified and how, and what is still open. Keep it short."""


SOLO_PROMPT = """You are working through SiCode, a coding assistant, in the user's repository (your working \
directory). Carry out the request yourself: read the relevant files before changing them, make the changes the \
request asks for and no others, and check your work when you have a way to. If you cannot run commands here, say \
which checks are still to be run. Keep the final reply short: what you did or found, what you ran and its result, \
and anything unfinished."""

DELEGATE_RE = re.compile(r"<delegate>\s*(.*?)\s*</delegate>", re.S)
PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.S)
STEP_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\[([ xX~])\]\s*)?(?:\[([lLcC])\]\s*)?(.*\S)\s*$")
MAX_HANDOFFS = 12


def find(planner):
    """Path of the planner's command-line tool, or None."""
    return shutil.which({"claude-code": "claude", "codex": "codex"}.get(planner, ""))


def default_planner(have_key):
    """An account login when its tool is installed, else the API key."""
    for p in ("claude-code", "codex"):
        if find(p):
            return p
    return "api" if have_key else "claude-code"


def parse_plan(text):
    """[{state: todo|doing|done, who: "l"|"c"|"", text}] from plan lines such as "[~] [l] run the tests"."""
    steps = []
    for line in text.strip().splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        mark = m.group(1)
        state = "done" if mark in ("x", "X") else "doing" if mark == "~" else "todo"
        steps.append({"state": state, "who": (m.group(2) or "").lower(), "text": m.group(3)})
    return steps


def split_reply(text):
    """(text shown to the user, task for the worker or None, plan text or None)."""
    plan = PLAN_RE.search(text)
    if plan:
        text = text[:plan.start()] + text[plan.end():]
    m = DELEGATE_RE.search(text)
    if not m:
        return text.strip(), None, plan.group(1) if plan else None
    return text[:m.start()].strip(), m.group(1).strip(), plan.group(1) if plan else None


class Planner:
    """One conversation with Claude Code or Codex, run non-interactively in the workspace."""

    def __init__(self, planner, cwd, model=None, session=None, solo=False, allow_shell=False):
        self.planner, self.cwd, self.model = planner, cwd, model
        self.solo = solo                # the tool does the work itself instead of handing it to the local model
        self.allow_shell = allow_shell  # solo: it may also run shell commands
        self.session = session   # the tool's own session id, to continue the conversation
        self.proc = None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def _run(self, args, prompt, should_stop):
        exe = find(self.planner)
        if not exe:
            name = "Claude Code" if self.planner == "claude-code" else "Codex"
            raise RuntimeError(f"{name} is not installed or not on PATH; install it and sign in, or choose another "
                               "planner")
        # the point is the account login: a key in the environment would make the tool bill the API instead
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        self.proc = subprocess.Popen([exe, *args], cwd=self.cwd, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.proc.stdin.write(prompt.encode("utf-8"))
            self.proc.stdin.close()
            # the pipes are drained by helpers so a long reply cannot block the tool
            chunks, errs = [], []
            t1 = threading.Thread(target=lambda: chunks.append(self.proc.stdout.read()), daemon=True)
            t2 = threading.Thread(target=lambda: errs.append(self.proc.stderr.read()), daemon=True)
            t1.start(); t2.start()
            while self.proc.poll() is None:
                if should_stop():
                    self.proc.kill()
                    raise KeyboardInterrupt
                time.sleep(0.2)
            t1.join(5); t2.join(5)
        except KeyboardInterrupt:
            self.stop()
            raise
        text = b"".join(chunks).decode("utf-8", errors="replace")
        err = b"".join(errs).decode("utf-8", errors="replace")
        return self.proc.returncode, text, err

    def ask(self, prompt, should_stop=lambda: False):
        """Sends one message and returns the planner's reply text."""
        if self.planner == "claude-code":
            tools = "Read,Grep,Glob,Edit,Write" + (",Bash" if self.allow_shell else "")
            args = ["-p", "--output-format", "json", "--tools", tools, "--permission-mode", "acceptEdits",
                    "--append-system-prompt", SOLO_PROMPT if self.solo else hybrid_prompt(self.allow_shell)]
            if self.allow_shell:
                args += ["--allowedTools", "Bash"]
            if self.session:
                args += ["--resume", self.session]
            if self.model:
                args += ["--model", self.model]
            code, text, err = self._run(args, prompt, should_stop)
            data = None
            for line in reversed(text.strip().splitlines()):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
            if not isinstance(data, dict):
                raise RuntimeError("Claude Code gave no reply" + (": " + (err or text).strip()[-400:] if err or text else ""))
            self.session = data.get("session_id") or self.session
            if data.get("is_error"):
                raise RuntimeError("Claude Code: " + str(data.get("result") or data.get("subtype") or "error")[:400]
                                   + " (run `claude` once in a terminal to sign in, if that is the problem)")
            return str(data.get("result") or "")
        if self.planner == "codex":
            if self.session:
                args = ["exec", "resume", self.session, "-", "--json", "--skip-git-repo-check"]
            else:
                args = ["exec", "--json", "--skip-git-repo-check", "-"]
                prompt = (SOLO_PROMPT if self.solo else hybrid_prompt(True)) + "\n\n# The user's request\n\n" + prompt
            # Codex edits and runs commands inside its own sandbox, limited to the workspace
            args += ["-c", 'sandbox_mode="workspace-write"']
            if self.model:
                args += ["-m", self.model]
            code, text, err = self._run(args, prompt, should_stop)
            reply, failure = [], ""
            for line in text.splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "thread.started":
                    self.session = e.get("thread_id") or self.session
                elif e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message":
                    reply.append(e["item"].get("text", ""))
                elif e.get("type") in ("error", "turn.failed"):
                    failure = json.dumps(e)[:400]
            if not reply:
                raise RuntimeError("Codex gave no reply: " + (failure or err.strip()[-400:] or f"exit code {code}")
                                   + " (run `codex login` in a terminal to sign in, if that is the problem)")
            return reply[-1]
        raise RuntimeError(f"unknown planner {self.planner}")


def run(planner, text, delegate, show, should_stop=lambda: False, on_plan=lambda steps: None,
        before_turn=lambda: None, after_turn=lambda: None):
    """One user request. The tool's replies go to `show`, its plan to `on_plan(steps)`, its [l] tasks to
    `delegate(task) -> report`. before_turn / after_turn bracket every run of the tool, which may edit files."""
    def ask(prompt):
        before_turn()
        try:
            return planner.ask(prompt, should_stop)
        finally:
            after_turn()

    reply = ask(text)
    if planner.solo:
        show(reply.strip() or "[no reply]")
        return
    for _ in range(MAX_HANDOFFS):
        shown, task, plan = split_reply(reply)
        if plan:
            on_plan(parse_plan(plan))
        if shown:
            show(shown)
        if task is None:
            return
        report = delegate(task)
        reply = ask("The worker has finished the [l] task.\n\n" + report)
    show("[stopped: more hand-offs than SiCode allows in one request]")
