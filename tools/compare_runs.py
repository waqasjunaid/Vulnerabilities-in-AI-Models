#!/usr/bin/env python3
"""Explain why two runs of the same configuration disagree on individual cases.

    python compare_runs.py RUN_A RUN_B [--suite workspace] [--field utility] [--examples 4]

For every attacked case present in both runs it reports where `field` differs,
splits the differences by user task, and shows whether guardian blocks, tool
errors or an early divergence in the model's own tool calls explain them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(root):
    out = {}
    for f in Path(root).rglob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or "utility" not in d or "suite_name" not in d:
            continue
        if str(d.get("user_task_id", "")).startswith("injection") or not d.get("injection_task_id"):
            continue
        key = (d["suite_name"], d["user_task_id"], d["injection_task_id"])
        prev = out.get(key)
        if prev is None or str(d.get("evaluation_timestamp", "")) >= str(prev.get("evaluation_timestamp", "")):
            out[key] = d
    return out


def tool_msgs(d):
    return [m for m in (d.get("messages") or []) if m.get("role") == "tool"]


def blocked(m):
    s = json.dumps(m)
    return "GuardianBlocked" in s or "[BLOCKED by guardian" in s


def calls(d):
    seq = []
    for m in tool_msgs(d):
        tc = m.get("tool_call") or {}
        seq.append((tc.get("function"), json.dumps(tc.get("args") or {}, sort_keys=True, default=str)))
    return seq


def profile(d):
    tm = tool_msgs(d)
    return {
        "blocked": sum(1 for m in tm if blocked(m)),
        "errors": sum(1 for m in tm if m.get("error") and not blocked(m)),
        "tool_calls": len(tm),
        "messages": len(d.get("messages") or []),
        "run_error": bool(d.get("error")),
    }


def first_divergence(a, b):
    ca, cb = calls(a), calls(b)
    for i in range(min(len(ca), len(cb))):
        if ca[i] != cb[i]:
            return i, ca[i], cb[i]
    if len(ca) != len(cb):
        return min(len(ca), len(cb)), None, None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--suite")
    ap.add_argument("--field", default="utility")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    A, B = load(args.run_a), load(args.run_b)
    keys = sorted(k for k in set(A) & set(B) if not args.suite or k[0] == args.suite)
    print("cases in both runs: %d (suite=%s, field=%s)" % (len(keys), args.suite or "all", args.field))
    a_only = [k for k in keys if A[k][args.field] and not B[k][args.field]]
    b_only = [k for k in keys if B[k][args.field] and not A[k][args.field]]
    same = len(keys) - len(a_only) - len(b_only)
    print("agree=%d  only-A=%d  only-B=%d\n" % (same, len(a_only), len(b_only)))

    per_task = defaultdict(lambda: [0, 0, 0])
    for k in keys:
        per_task[(k[0], k[1])][0] += 1
        per_task[(k[0], k[1])][1] += bool(A[k][args.field])
        per_task[(k[0], k[1])][2] += bool(B[k][args.field])
    diffs = sorted(((abs(v[1] - v[2]), t, v) for t, v in per_task.items() if v[1] != v[2]), reverse=True)
    print("user tasks whose success count differs (n, A, B):")
    for _, t, v in diffs[:20]:
        print("  %-9s %-14s n=%-3d A=%-3d B=%-3d" % (t[0], t[1], v[0], v[1], v[2]))
    print("  ... %d task(s) differ in total\n" % len(diffs))

    for label, group, win, lose in (("only-B", b_only, B, A), ("only-A", a_only, A, B)):
        if not group:
            continue
        agg = Counter()
        div_at = Counter()
        for k in group:
            pl, pw = profile(lose[k]), profile(win[k])
            agg["losing run had guardian block"] += pl["blocked"] > 0
            agg["winning run had guardian block"] += pw["blocked"] > 0
            agg["losing run had tool error"] += pl["errors"] > 0
            agg["losing run recorded run error"] += pl["run_error"]
            div = first_divergence(A[k], B[k])
            div_at["no divergence in tool calls"] += div is None
            div_at["diverged at first call"] += bool(div and div[0] == 0)
            div_at["diverged later"] += bool(div and div[0] > 0)
        print("== %s: %d cases (success in the first-named run only) ==" % (label, len(group)))
        for name, v in sorted(agg.items()):
            print("   %-34s %d" % (name, v))
        for name, v in sorted(div_at.items()):
            print("   %-34s %d" % (name, v))
        for k in group[: args.examples]:
            print("   example", k)
            print("      A:", profile(A[k]))
            print("      B:", profile(B[k]))
            div = first_divergence(A[k], B[k])
            if div:
                print("      first divergence at tool call #%d" % div[0])
                print("        A:", str(div[1])[:150])
                print("        B:", str(div[2])[:150])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
