#!/usr/bin/env python3
"""Print the clean-run sentences a frozen detector threshold flags.

    python show_clean_flags.py --clean runs/agentdojo/final_baseline/clean \
        runs/agentdojo/baseline_run2/clean --model detectors/span-v1 --threshold 0.0155
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_detector_data as B  # noqa: E402
import eval_detector as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    found = {}
    for d in B.records(args.clean):
        if d.get("injection_task_id"):
            continue
        for m in d["messages"]:
            if m.get("role") != "tool" or not m.get("tool_call"):
                continue
            text = B.tool_text(m)
            if not text or B.is_blocked(text):
                continue
            func = m["tool_call"]["function"]
            for s in B.sentences(text):
                found.setdefault(s, (d["suite_name"], func))
    sents = sorted(found)
    scores = E.score_hf(args.model, sents, args.batch)
    rows = sorted(
        ((scores[i], found[s][0], found[s][1], s) for i, s in enumerate(sents) if scores[i] >= args.threshold),
        key=lambda r: (r[1], -r[0]),
    )
    print("%d flagged sentences in clean runs" % len(rows))
    for sc, suite, func, sent in rows:
        print("[%5.3f] %-10s (%s) %s" % (sc, suite, func, sent[:180].replace("\n", " ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
