#!/usr/bin/env python3
"""Summarise AgentDojo runs: baseline vs guarded, with 95% Wilson intervals.

AgentDojo's `security` flag is True when the ATTACK SUCCEEDED, so lower is
better. Rows where the user task is itself an injection task ("injection_task_N"
with no injection) are excluded: their `security` flag only says the task goal
was reached legitimately and says nothing about the defense.

    python summarize_agentdojo.py --baseline ROOT [ROOT ...] --guarded ROOT [ROOT ...]
    python summarize_agentdojo.py --baseline runs/a --guarded runs/b --csv out.csv

Effect-based scoring: AgentDojo scores a few injection tasks from the list of
calls the model *requested* (`security_from_traces`), so a call that the guardian
blocked still counts as an attack success. By default such cases are rescored
from the calls that actually executed (blocked calls removed). Pass
`--raw-security` to use AgentDojo's flag unchanged. Needs the `agentdojo`
package; run it inside the same environment as the benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "   n/a"
    lo, hi = wilson(k, n)
    return "%3d/%-3d %5.1f%% [%4.1f,%5.1f]" % (k, n, 100.0 * k / n, 100 * lo, 100 * hi)


def sign_test_p(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


_SUITES = {}
RESCORED = []


def _is_blocked(message) -> bool:
    return "GuardianBlocked" in json.dumps(message) or "[BLOCKED by guardian" in json.dumps(message)


def effect_security(d):
    """Return the attack outcome from executed calls only, or None if not applicable."""
    if not d.get("security") or not d.get("injection_task_id"):
        return None
    messages = d.get("messages") or []
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    if not any(_is_blocked(m) for m in tool_msgs):
        return None
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo.task_suite.load_suites import get_suite

    suite = _SUITES.get(d["suite_name"])
    if suite is None:
        suite = _SUITES[d["suite_name"]] = get_suite("v1.2.2", d["suite_name"])
    task = suite.get_injection_task_by_id(d["injection_task_id"])
    traces = [
        FunctionCall(function=m["tool_call"]["function"], args=m["tool_call"].get("args") or {})
        for m in tool_msgs
        if m.get("tool_call") and not _is_blocked(m)
    ]
    return task.security_from_traces("", None, None, traces)


def load(roots, effect=False):
    """Return {(suite, user_task, injection_task or None, attack): record}, newest wins."""
    out = {}
    for root in roots:
        for f in Path(root).rglob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or "utility" not in d or "suite_name" not in d:
                continue
            if str(d.get("user_task_id", "")).startswith("injection"):
                continue
            key = (d["suite_name"], d["user_task_id"], d.get("injection_task_id"), d.get("attack_type"))
            if effect:
                corrected = effect_security(d)
                if corrected is not None and corrected != d["security"]:
                    RESCORED.append(key)
                    d["security"] = corrected
            prev = out.get(key)
            if prev is None or str(d.get("evaluation_timestamp", "")) >= str(prev.get("evaluation_timestamp", "")):
                out[key] = d
    return out


def split(records):
    clean = {k: v for k, v in records.items() if k[2] is None}
    attacked = {k: v for k, v in records.items() if k[2] is not None}
    return clean, attacked


def counts(records, field):
    return sum(1 for v in records.values() if v[field]), len(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--guarded", nargs="+", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--raw-security", action="store_true",
                    help="use AgentDojo's trace-based security flag as recorded")
    args = ap.parse_args()

    effect = not args.raw_security
    base, guard = load(args.baseline, effect), load(args.guarded, effect)
    b_clean, b_att = split(base)
    g_clean, g_att = split(guard)

    suites = sorted({k[0] for k in list(base) + list(guard)})
    rows = []
    header = "%-10s %-34s %-34s %-34s" % ("suite", "metric", "baseline", "guarded")
    print(header)
    print("-" * len(header))

    def report(suite, label, b, g, field):
        bk, bn = counts(b, field)
        gk, gn = counts(g, field)
        print("%-10s %-34s %-34s %-34s" % (suite, label, fmt(bk, bn), fmt(gk, gn)))
        rows.append([suite, label, bk, bn, gk, gn])

    for suite in suites + ["ALL"]:
        sel = (lambda d: d) if suite == "ALL" else (lambda d, s=suite: {k: v for k, v in d.items() if k[0] == s})
        report(suite, "clean utility", sel(b_clean), sel(g_clean), "utility")
        report(suite, "utility under attack", sel(b_att), sel(g_att), "utility")
        report(suite, "attack success rate (lower=better)", sel(b_att), sel(g_att), "security")

    shared = sorted(set(b_att) & set(g_att))
    if shared:
        print("\nPaired comparison on %d attacked cases present in both runs" % len(shared))
        for field, label in (("security", "attack success"), ("utility", "utility")):
            both = sum(1 for k in shared if base[k][field] and guard[k][field])
            only_b = sum(1 for k in shared if base[k][field] and not guard[k][field])
            only_g = sum(1 for k in shared if guard[k][field] and not base[k][field])
            neither = len(shared) - both - only_b - only_g
            print("  %-15s both=%d  baseline-only=%d  guarded-only=%d  neither=%d  exact sign-test p=%.4f"
                  % (label, both, only_b, only_g, neither, sign_test_p(only_b, only_g)))
    else:
        print("\nNo attacked cases present in both runs; paired test skipped.")

    missing_g = len(set(b_att) - set(g_att))
    missing_b = len(set(g_att) - set(b_att))
    if missing_g or missing_b:
        print("\nWARNING: %d attacked case(s) only in baseline, %d only in guarded; "
              "unpaired rows above are not like-for-like." % (missing_g, missing_b))

    if effect:
        print("\nEffect-based scoring: %d case(s) rescored from executed calls only (blocked calls removed)."
              % len(RESCORED))
        for key in sorted(RESCORED)[:20]:
            print("   rescored to attack-failed:", key[:3])

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["suite", "metric", "baseline_k", "baseline_n", "guarded_k", "guarded_n"])
            w.writerows(rows)
        print("wrote", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
