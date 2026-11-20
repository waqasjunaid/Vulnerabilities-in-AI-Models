#!/usr/bin/env python3
"""Print the tool calls, tool errors and final answer of one case in several runs.

    python show_case.py SUITE USER_TASK INJECTION_TASK RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import load  # noqa: E402


def text_of(msg):
    c = msg.get("content")
    if isinstance(c, list):
        return " ".join(str(p.get("content", "")) for p in c if isinstance(p, dict))
    return str(c or "")


def main() -> int:
    suite, ut, inj, *runs = sys.argv[1:]
    for root in runs:
        rec = load(root).get((suite, ut, inj))
        print("=" * 100)
        print(root, "->", "MISSING" if rec is None else "utility=%s security=%s" % (rec["utility"], rec["security"]))
        if rec is None:
            continue
        msgs = rec.get("messages") or []
        for m in msgs:
            if m.get("role") != "tool":
                continue
            tc = m.get("tool_call") or {}
            err = m.get("error")
            print("  tool %-28s %s%s" % (tc.get("function"), json.dumps(tc.get("args"), default=str)[:90],
                                         ("   ERROR: " + str(err)[:110]) if err else ""))
        finals = [m for m in msgs if m.get("role") == "assistant"]
        if finals:
            print("  FINAL ANSWER:", text_of(finals[-1]).replace("\n", " ")[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
