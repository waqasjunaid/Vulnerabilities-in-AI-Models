#!/usr/bin/env python3
"""Show the four replay residuals with and without privileged-tool grounding.

The main replay passes an empty environment, so v2 treats every tool as read-only
and only emails, URLs and IBANs can be tainted. This replays the same four cases
again with state-changing tools marked privileged.

    python tools/show_residual.py --project . \
        --baseline runs/agentdojo/final_baseline/attack \
        --scores detectors/span-v1-scores.json
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_guardian as R  # noqa: E402

KEYS = [
    ("banking", "user_task_0", "injection_task_5"),
    ("travel", "user_task_0", "injection_task_2"),
    ("travel", "user_task_0", "injection_task_4"),
    ("travel", "user_task_1", "injection_task_2"),
]
_READ = ("get_", "read_", "search_", "list_", "check_")


def replay(adapter, case, privileged):
    msgs = case["messages"]
    goal = R.user_goal(msgs)
    runtime = SimpleNamespace(functions={})
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            args = dict(tc.get("args") or {})
            action = tc["function"]
            if privileged and not action.startswith(_READ):
                env = SimpleNamespace(tools={action: SimpleNamespace(privileged=True)})
            else:
                env = SimpleNamespace()
            try:
                d = adapter.inspect(
                    goal=goal, action=action, action_input=args,
                    runtime=runtime, env=env, messages=msgs[: i + 1],
                )
                allow, reason = bool(d.allow), str(d.reason)
            except Exception as exc:
                allow, reason = True, "ERROR %s" % exc
            out.append((action, args, allow, reason))
    return out


def show(label, rows):
    print("  -- %s" % label)
    for action, args, allow, reason in rows:
        print("  %-5s %-28s %s" % ("BLOCK" if not allow else "allow", action, json.dumps(args, default=str)[:170]))
        if not allow:
            print("       ", reason[:170])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--threshold", type=float, default=0.0155)
    args = ap.parse_args()
    cases = R.load_cases(args.baseline, {"banking", "travel"})
    adapter = R.make_adapter(args.project, "v2")
    saved = []
    for key in KEYS:
        print("\n==", key)
        saved.append(replay(adapter, cases[key], privileged=True))
        show("v2, state-changing tools privileged", saved[-1])
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    R.install_span_scores(args.threshold, scores)
    for key, prev in zip(KEYS, saved):
        print("\n==", key, "(detector spans)")
        now = replay(adapter, cases[key], privileged=True)
        show("v2+span, state-changing tools privileged", now)
        if [(a, x, b) for a, x, b, _ in now] != [(a, x, b) for a, x, b, _ in prev]:
            print("  decision changed")
        else:
            print("  same decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
