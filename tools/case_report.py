#!/usr/bin/env python3
"""Per-case AgentDojo report: which tools were called, which were blocked, and the outcome.

Use it to understand WHY a case ended the way it did, e.g. an attack that
succeeded even though the guardian denied calls.

    python case_report.py RUN_ROOT [RUN_ROOT ...] [--suite slack] [--attacked-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(str(block.get("content", "")))
        else:
            parts.append(str(block))
    return " ".join(parts)


def analyse(record):
    calls = []
    blocked = []
    for m in record.get("messages", []):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function")
                fn = fn.get("name") if isinstance(fn, dict) else fn
                calls.append(str(fn))
        elif m.get("role") == "tool":
            body = text_of(m.get("content")) + " " + str(m.get("error") or "")
            if "GuardianBlocked" in body or "[BLOCKED by guardian" in body:
                tc = m.get("tool_call") or {}
                fn = tc.get("function")
                fn = fn.get("name") if isinstance(fn, dict) else fn
                blocked.append(str(fn))
    return calls, blocked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--suite")
    ap.add_argument("--attacked-only", action="store_true")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        for f in sorted(Path(root).rglob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or "utility" not in d or "suite_name" not in d:
                continue
            if str(d.get("user_task_id", "")).startswith("injection"):
                continue
            if args.suite and d["suite_name"] != args.suite:
                continue
            if args.attacked_only and d.get("injection_task_id") is None:
                continue
            calls, blocked = analyse(d)
            rows.append((d["suite_name"], d["user_task_id"], d.get("injection_task_id") or "-",
                         d["utility"], d["security"], calls, blocked))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    for suite, ut, inj, util, sec, calls, blocked in rows:
        print("%-9s %-13s %-17s utility=%-5s attack_succeeded=%-5s blocked=%s" % (
            suite, ut, inj, util, sec, blocked or "-"))
        print("          calls: %s" % " > ".join(calls) if calls else "          calls: (none)")
    print("\n%d case(s)" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
