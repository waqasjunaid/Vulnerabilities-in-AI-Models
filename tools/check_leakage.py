#!/usr/bin/env python3
"""Fail if benchmark-specific answers are hard-coded in the guardian code.

A defense that mentions concrete AgentDojo task content (URLs, names, amounts,
task ids) is tuned to the test set, and any result obtained with it is invalid.
Run this before every benchmark launch; the exit code is 1 on any hit.

    python check_leakage.py PATH [PATH ...] [--extra REGEX ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_PATTERNS = {
    "benchmark URL": r"\bwww\.[\w-]+\.(?:com|org|net)\b",
    "task-specific name": r"\b(?:eve-blog|our-company|bob'?s inbox|alice enjoys|charlie enjoys)\b",
    "task content": r"\b(?:swimming|painting|hobb(?:y|ies))\b",
    "task id literal": r"\b(?:user_task|injection_task)_\d+\b",
    "rent special case": r"\brent (?:notice|change|amount)\b|legitimate rent",
    "hard-coded tool arguments": r"\"url\"\s*:\s*\"www\.",
}

SKIP_DIRS = {".git", "__pycache__", "runs", "node_modules", ".venv", "venv"}
SKIP_SUFFIXES = (".bak", ".pyc", ".json", ".jsonl", ".log", ".csv", ".png")


def iter_files(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.parts) & SKIP_DIRS):
                    if f.suffix == ".py" or f.suffix == ".sh":
                        if not f.name.endswith(SKIP_SUFFIXES):
                            yield f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--extra", action="append", default=[], help="additional regex to flag")
    ap.add_argument("--allow", action="append", default=[],
                    help="file name (basename) to skip, e.g. this script or test fixtures")
    args = ap.parse_args()

    patterns = {k: re.compile(v, re.I) for k, v in DEFAULT_PATTERNS.items()}
    for i, extra in enumerate(args.extra):
        patterns["extra[%d]" % i] = re.compile(extra, re.I)

    hits = 0
    scanned = 0
    for f in iter_files(args.paths):
        if f.name in args.allow or f.name == Path(__file__).name:
            continue
        scanned += 1
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print("cannot read %s: %s" % (f, exc), file=sys.stderr)
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for label, rx in patterns.items():
                if rx.search(line):
                    hits += 1
                    print("%s:%d: [%s] %s" % (f, lineno, label, line.strip()[:140]))
    print("\nscanned %d file(s), %d suspicious line(s)" % (scanned, hits))
    if hits:
        print("FAIL: remove benchmark-specific content before running a benchmark.")
        return 1
    print("OK: no benchmark-specific content found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
