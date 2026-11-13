"""File and shell tools shared by the architect (Claude) and the local worker. Every path is confined to the workspace."""
import fnmatch
import os
import re
import subprocess

MAX_OUT = 20000  # characters of tool output handed back to a model
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".vs", "models"}


def clip(text, limit=MAX_OUT):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters omitted] ...\n" + text[-half:]


class Workspace:
    """`approve(command) -> bool` is asked before each shell command; `changed` collects the files written."""

    def __init__(self, root, approve):
        self.root = os.path.realpath(root)
        self.approve = approve
        self.changed = set()

    def path(self, rel):
        p = os.path.realpath(os.path.join(self.root, rel))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ValueError(f"{rel} is outside the workspace")
        return p

    def rel(self, p):
        return os.path.relpath(p, self.root).replace("\\", "/")

    def walk(self):
        for d, dirs, files in os.walk(self.root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for f in sorted(files):
                yield os.path.join(d, f)

    # --- tools: each returns the text shown to the model ---

    def read_file(self, path, offset=1, limit=400):
        with open(self.path(path), encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
        offset, limit = max(1, int(offset)), max(1, int(limit))
        part = lines[offset - 1:offset - 1 + limit]
        out = "\n".join(f"{offset + i}\t{line}" for i, line in enumerate(part))
        if offset - 1 + limit < len(lines):
            out += f"\n[{len(lines)} lines in total; read more with offset={offset + limit}]"
        return clip(out) if out else "[empty file]"

    def write_file(self, path, content):
        p = self.path(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        self.changed.add(self.rel(p))
        return f"wrote {self.rel(p)} ({content.count(chr(10)) + 1} lines)"

    def edit_file(self, path, old_text, new_text):
        p = self.path(path)
        with open(p, encoding="utf-8", newline="") as f:
            text = f.read()
        crlf = "\r\n" in text
        if crlf:
            text = text.replace("\r\n", "\n")
        n = text.count(old_text)
        if n == 0:
            return "error: old_text was not found in the file (it must match exactly, including indentation)"
        if n > 1:
            return f"error: old_text occurs {n} times; include more surrounding lines so it is unique"
        text = text.replace(old_text, new_text)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace("\n", "\r\n") if crlf else text)
        self.changed.add(self.rel(p))
        return f"edited {self.rel(p)}"

    def list_files(self, pattern="*"):
        hits = [r for r in (self.rel(p) for p in self.walk())
                if fnmatch.fnmatch(r, pattern) or fnmatch.fnmatch(os.path.basename(r), pattern)]
        if not hits:
            return "no files match"
        return clip("\n".join(hits[:500]) + (f"\n[{len(hits) - 500} more]" if len(hits) > 500 else ""))

    def search(self, pattern, glob="*"):
        rx = re.compile(pattern)
        out = []
        for p in self.walk():
            r = self.rel(p)
            if not (fnmatch.fnmatch(r, glob) or fnmatch.fnmatch(os.path.basename(r), glob)):
                continue
            try:
                if os.path.getsize(p) > 2_000_000:
                    continue
                with open(p, encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            out.append(f"{r}:{i}: {line.rstrip()[:300]}")
                            if len(out) >= 200:
                                return clip("\n".join(out) + "\n[stopped at 200 matches]")
            except (UnicodeDecodeError, OSError):
                continue
        return clip("\n".join(out)) if out else "no matches"

    def run_command(self, command, timeout=300):
        if not self.approve(command):
            return "error: the user declined to run this command"
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, timeout=int(timeout),
                               encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return f"error: the command did not finish within {timeout} s"
        return clip(f"exit code {r.returncode}\n{r.stdout}{r.stderr}".rstrip())

    def call(self, name, args):
        fn = {"read_file": self.read_file, "write_file": self.write_file, "edit_file": self.edit_file,
              "list_files": self.list_files, "search": self.search, "run_command": self.run_command}.get(name)
        if fn is None:
            return f"error: unknown tool {name}"
        try:
            return fn(**args)
        except Exception as e:  # the model reads the error and corrects itself
            return f"error: {type(e).__name__}: {e}"


def _tool(name, description, props, required):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}


S, I = {"type": "string"}, {"type": "integer"}
# Tool definitions in the Claude API shape; the worker converts them to the Qwen chat template's shape.
TOOLS = [
    _tool("read_file", "Read a text file from the workspace. Returns numbered lines.",
          {"path": {**S, "description": "Path relative to the workspace root."},
           "offset": {**I, "description": "First line to read, from 1. Default 1."},
           "limit": {**I, "description": "Number of lines. Default 400."}}, ["path"]),
    _tool("write_file", "Create a file or replace its whole content. Use edit_file for a change to an existing file.",
          {"path": S, "content": {**S, "description": "The complete new content of the file."}}, ["path", "content"]),
    _tool("edit_file", "Replace one exact occurrence of old_text in a file with new_text.",
          {"path": S, "old_text": {**S, "description": "Text to replace; must occur exactly once in the file."},
           "new_text": S}, ["path", "old_text", "new_text"]),
    _tool("list_files", "List workspace files whose path or name matches a glob pattern such as *.py or src/*.cpp.",
          {"pattern": {**S, "description": "Glob pattern. Default *."}}, []),
    _tool("search", "Search file contents with a regular expression. Returns path:line: text.",
          {"pattern": {**S, "description": "Python regular expression."},
           "glob": {**S, "description": "Only search files matching this glob. Default *."}}, ["pattern"]),
    _tool("run_command", "Run a shell command in the workspace root (Windows cmd) and return its exit code and output. "
          "Use it to build and to run tests.",
          {"command": S, "timeout": {**I, "description": "Seconds. Default 300."}}, ["command"]),
]
