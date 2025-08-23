#!/usr/bin/env python3
"""
push_project.py — Push the MLB_SITE project (or any folder) to ChatGPT in chunked parts.

Usage examples:
  python push_project.py --path . --task "Full code review + bugs + tests"
  python push_project.py --only_changed --task "Review my latest changes"
  python push_project.py --glob "*.py,*.md,*.toml" --max_chars 120000
"""

import os
import re
import sys
import fnmatch
import argparse
from pathlib import Path
from typing import Iterable, List

# --- Load .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # optional; script still works if python-dotenv not installed

# --- OpenAI SDK ---
try:
    from openai import OpenAI
except Exception:
    print("Please install dependencies:  pip install openai python-dotenv", file=sys.stderr)
    sys.exit(1)

# -------------------- Config --------------------
DEFAULT_MODEL = "gpt-5"          # large context model
DEFAULT_MAX_CHARS = 150_000      # conservative per-request character cap

# Project-aware ignores (tailored to your MLB_SITE layout)
DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "__pycache__",
    "venv", "env", ".env", "node_modules", "dist", "build",
    ".next", ".turbo", "cache", "instance", "ignore", ".mypy_cache"
}

# Skip heavy/binary/derived or secrets
DEFAULT_IGNORE_GLOBS = {
    "*.pyc", "*.pyo", "*.pyd",
    "*.so", "*.dll", "*.dylib",
    "*.log", "*.lock",
    "*.zip", "*.tar", "*.gz", "*.bz2", "*.xz",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.pdf",
    "*.mp4", "*.mov", "*.webm", "*.mp3", "*.wav",
    "*.sqlite", "*.db",
    "*.env", ".env*", ".env.*",
    # project-specific artifacts
    "*.csv", "*.parquet", "*.feather",  # e.g., predictions/data dumps
}

# Code-ish files worth sending by default
DEFAULT_CODE_GLOBS = {
    "*.py", "*.ipynb",  # notebooks optional; large ones will be chunked or can be excluded
    "*.js", "*.jsx", "*.ts", "*.tsx",
    "*.json", "*.toml", "*.ini",
    "*.yaml", "*.yml",
    "*.md", "*.rst",
    "*.html", "*.css",
}

FENCE_LANG_MAP = {
    ".py": "python", ".ipynb": "json",
    ".js": "javascript", ".jsx": "jsx",
    ".ts": "typescript", ".tsx": "tsx",
    ".json": "json", ".toml": "toml", ".ini": "",
    ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".rst": "markdown",
    ".html": "html", ".css": "css",
}

# -------------------- Helpers --------------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def should_skip(path: Path, ignore_dirs: set, ignore_globs: set) -> bool:
    # skip if any parent directory is in ignore list
    for part in path.parts:
        if part in ignore_dirs:
            return True
    # skip if file name matches ignore globs
    name = path.name
    for pat in ignore_globs:
        if fnmatch.fnmatch(name, pat):
            return True
    return False

def matches_any_glob(path: Path, globs: Iterable[str]) -> bool:
    if not globs:
        return True
    for pat in globs:
        if fnmatch.fnmatch(path.name, pat):
            return True
    return False

def list_files(root: Path, include_globs: Iterable[str], ignore_dirs: set, ignore_globs: set) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if should_skip(p, ignore_dirs, ignore_globs):
            continue
        if matches_any_glob(p, include_globs):
            files.append(p)
    return files

def list_changed_files_from_git() -> List[Path]:
    """Return files changed vs HEAD (staged or unstaged)."""
    try:
        import subprocess
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        paths = []
        for line in out.splitlines():
            # typical formats: " M path", "A  path", "?? path"
            m = re.match(r"^\s*[MADRCU\?\!]{1,2}\s+(.*)$", line)
            if m:
                paths.append(Path(m.group(1)))
        return [p for p in paths if p.exists()]
    except Exception:
        return []

def format_file_block(path: Path) -> str:
    ext = path.suffix.lower()
    lang = FENCE_LANG_MAP.get(ext, "")
    code = read_text(path)
    rel = str(path)
    header = f"### {rel}\n"
    if lang:
        return header + f"```{lang}\n{code}\n```\n\n"
    else:
        return header + f"```\n{code}\n```\n\n"

def chunk_text(big_text: str, max_chars: int) -> List[str]:
    """Chunk on double newlines when possible; hard-split giant blocks."""
    if len(big_text) <= max_chars:
        return [big_text]

    chunks: List[str] = []
    current: List[str] = []

    for block in big_text.split("\n\n"):
        candidate = ("\n\n".join(current) + ("\n\n" if current else "") + block)
        if len(candidate) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = [block]
            else:
                b = block
                while len(b) > max_chars:
                    chunks.append(b[:max_chars])
                    b = b[max_chars:]
                if b:
                    current = [b]
        else:
            current.append(block)

    if current:
        chunks.append("\n\n".join(current))
    return chunks

def make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY. Put it in a .env file or export it.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)

def send_chunk(client: OpenAI, model: str, task: str, chunk: str, part_idx: int, total_parts: int, run_id: str):
    system_msg = (
        "You are a senior software engineer for an MLB predictions Flask app. "
        "Keep a running mental model across multi-part dumps. For each part, reply 'Part X/Y received' "
        "and highlight any immediate red flags. On the final part, deliver the requested output."
    )
    user_msg = (
        f"RUN ID: {run_id}\n"
        f"TASK: {task}\n"
        f"PART: {part_idx}/{total_parts}\n\n"
        f"{chunk}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )
    content = resp.choices[0].message.content
    print("\n" + "="*80 + f"\nAssistant response (Part {part_idx}/{total_parts}):\n" + "="*80)
    print(content)
    print()

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=".", help="Project root or subfolder to push")
    parser.add_argument("--task", default=(
        "Audit architecture, Flask routes, services, and ML model code. "
        "Find bugs, security issues, fragile paths, and performance problems. "
        "Propose tests (pytest) and a CI plan. Summarize the API contract for /routes."
    ))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--glob", default=",".join(sorted(DEFAULT_CODE_GLOBS)),
                        help="Comma-separated whitelist globs, e.g. *.py,*.md")
    parser.add_argument("--ignore_globs", default=",".join(sorted(DEFAULT_IGNORE_GLOBS)),
                        help="Comma-separated ignore globs")
    parser.add_argument("--ignore_dirs", default=",".join(sorted(DEFAULT_IGNORE_DIRS)),
                        help="Comma-separated ignore directories")
    parser.add_argument("--only_changed", action="store_true",
                        help="Send only files changed vs git status")
    parser.add_argument("--run_id", default=None,
                        help="Optional run/session id (auto-generated if omitted)")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    include_globs = [g.strip() for g in args.glob.split(",") if g.strip()]
    ignore_globs = set(g.strip() for g in args.ignore_globs.split(",") if g.strip())
    ignore_dirs = set(d.strip() for d in args.ignore_dirs.split(",") if d.strip())

    if args.only_changed:
        files = [p for p in list_changed_files_from_git()
                 if p.exists()
                 and not should_skip(p, ignore_dirs, ignore_globs)
                 and matches_any_glob(p, include_globs)]
        if not files:
            print("No changed files found (or all filtered).")
            return
    else:
        files = list_files(root, include_globs, ignore_dirs, ignore_globs)
        if not files:
            print("No matching files found.")
            return

    files = sorted(files, key=lambda p: str(p))
    blocks = [format_file_block(p) for p in files]
    project_dump = "".join(blocks)

    chunks = chunk_text(project_dump, args.max_chars)
    total = len(chunks)

    run_id = args.run_id or f"{root.name}:{os.getpid()}"

    client = make_client()
    for i, ch in enumerate(chunks, start=1):
        send_chunk(client, args.model, args.task, ch, i, total, run_id)

if __name__ == "__main__":
    main()
