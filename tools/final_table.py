#!/usr/bin/env python3
"""One table for the paper: every run of every configuration, plus per-configuration ranges.

    python final_table.py baseline/1=runs/agentdojo/final_baseline baseline/2=runs/agentdojo/baseline_run2 \
        v2/1=runs/agentdojo/v2_run1 ...

Label format is CONFIG/RUN. Attack success uses effect-based scoring (blocked calls
removed). Runs that do not cover all attacked cases are marked partial and are left
out of the per-configuration ranges.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_agentdojo as S  # noqa: E402

SUITES = ("workspace", "banking", "slack", "travel")
FULL = {"workspace": 560, "banking": 144, "slack": 105, "travel": 140}


def run_stats(path):
    clean, att = S.split(S.load([path], effect=True))
    out = {"n_att": len(att), "n_clean": len(clean)}
    out["asr"] = S.counts(att, "security")
    out["uua"] = S.counts(att, "utility")
    out["clean"] = S.counts(clean, "utility")
    out["suite_asr"] = {}
    out["suite_n"] = {}
    for s in SUITES:
        sub = {k: v for k, v in att.items() if k[0] == s}
        out["suite_asr"][s] = S.counts(sub, "security")
        out["suite_n"][s] = len(sub)
    out["complete"] = all(out["suite_n"][s] == FULL[s] for s in SUITES)
    return out


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="CONFIG/RUN=PATH")
    args = ap.parse_args()

    results = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        results.append((label, run_stats(path)))

    head = "%-14s %-9s %-26s %-26s %-18s" % ("run", "attacked", "attack success", "utility under attack", "clean utility")
    print(head)
    print("-" * len(head) + "".join("-" * 12 for _ in SUITES))
    for label, r in results:
        asr_k, asr_n = r["asr"]
        flag = "" if r["complete"] else "  (partial)"
        suite_cells = " ".join("%s %d/%d" % (s[:3], r["suite_asr"][s][0], r["suite_n"][s]) for s in SUITES)
        print("%-14s %-9d %-26s %-26s %-18s %s%s" % (
            label, r["n_att"], S.fmt(*r["asr"]).strip(), S.fmt(*r["uua"]).strip(),
            "n/a" if not r["n_clean"] else S.fmt(*r["clean"]).strip().split(" [")[0], suite_cells, flag))

    groups = defaultdict(list)
    for label, r in results:
        if r["complete"]:
            groups[label.split("/")[0]].append(r)
    print("\nper configuration (complete runs only): mean [min, max] over runs")
    for cfg, rs in groups.items():
        a = [pct(*r["asr"]) for r in rs]
        u = [pct(*r["uua"]) for r in rs]
        print("  %-14s runs=%d  attack success %.2f%% [%.2f, %.2f]   utility under attack %.1f%% [%.1f, %.1f]" % (
            cfg, len(rs), sum(a) / len(a), min(a), max(a), sum(u) / len(u), min(u), max(u)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
