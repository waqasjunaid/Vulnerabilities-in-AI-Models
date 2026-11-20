#!/usr/bin/env python3
"""Explain the replay table: which attacks a guardian version misses and which useful calls it blocks.

    python tools/replay_guardian.py --project . --baseline BASE --dump /tmp/replay_dump.jsonl
    python tools/diag_replay.py --baseline BASE --dump /tmp/replay_dump.jsonl [--version v2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_guardian as R  # noqa: E402


def short(value, n=70):
    return json.dumps(value, default=str)[:n]


def field(reason, name):
    m = re.search(name + r"=([\w.]+)", reason)
    return m.group(1) if m else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--version", default="v2")
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()

    suites = set(R.DEV_SUITES) | ({"banking", "travel"} if args.heldout else set())
    cases = {}
    for root in args.baseline:
        cases.update(R.load_cases(root, suites))
    calls = defaultdict(list)
    for line in open(args.dump, encoding="utf-8"):
        r = json.loads(line)
        if r["version"] == args.version:
            calls[tuple(r["case"])].append(r)

    print("== ATTACKS NOT STOPPED by %s (baseline-successful cases) ==" % args.version)
    kinds = Counter()
    for k in sorted(cases):
        if not cases[k].get("security"):
            continue
        sigs = R.attack_signatures(k[0], k[2])
        matched = [c for c in calls[k] if R.is_attack_call(c, sigs)]
        if any(not c["allow"] for c in matched):
            continue
        if not matched:
            kinds["no call matched the ground-truth signature"] += 1
            names = sorted({c["function"] for c in calls[k]})
            print("  %s %s %s: NO SIGNATURE MATCH; tools called: %s" % (k[0], k[1], k[2], ",".join(names)[:100]))
            continue
        c = matched[0]
        kinds["attack call allowed (%s)" % c["function"]] += 1
        print("  %s %s %s: %s %s" % (k[0], k[1], k[2], c["function"], short(c["args"], 60)))
        print("       decision: %s" % c["reason"][:110])

    print("\n== USEFUL CALLS BLOCKED by %s (baseline runs that completed the user task safely) ==" % args.version)
    classes = Counter()
    align = Counter()
    for k in sorted(cases):
        if not (cases[k].get("utility") and not cases[k].get("security")):
            continue
        sigs = R.attack_signatures(k[0], k[2])
        bad = [c for c in calls[k] if not c["allow"] and not R.is_attack_call(c, sigs)]
        if not bad:
            continue
        c = bad[0]
        if c["reason"].startswith("blocked by argument grounding"):
            cls = "taint"
            detail = re.findall(r"value\(s\) (.*?) reach", c["reason"])
            detail = detail[0][:60] if detail else ""
        else:
            cls = "v1"
            detail = "content=%s goal_alignment=%s known=%s risk=%s" % (
                field(c["reason"], "content"), field(c["reason"], "goal_alignment"),
                field(c["reason"], "goal_compatibility_known"), field(c["reason"], "risk"))
            align[field(c["reason"], "goal_alignment")] += 1
        classes[cls] += 1
        print("  %s %s %s: %s [%s] %s" % (k[0], k[1], k[2], c["function"], cls, detail))
    print("\nsummary: attacks not stopped:", dict(kinds))
    print("         useful runs hit by class:", dict(classes), " goal_alignment of v1-class blocks:", dict(align))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
