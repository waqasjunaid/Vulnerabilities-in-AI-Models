#!/usr/bin/env python3
"""Apply the frozen detector threshold to recorded tool outputs.

Counts, per suite, how many tool outputs contain a sentence the detector flags, and how many
of those flags are sentences the marker regex does not already catch. Clean runs are the
false-positive check: a flag there is a span the guardian would treat as an injection on a
benign task.

    python score_traces.py --attack runs/agentdojo/final_baseline/attack \
        --clean runs/agentdojo/final_baseline/clean --model detectors/span-v1 --threshold 0.0155
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_detector_data as B  # noqa: E402
import build_spans as S  # noqa: E402
import eval_detector as E  # noqa: E402


def outputs(roots, attacked):
    for d in B.records(roots):
        inj = d.get("injection_task_id")
        if bool(inj) != attacked:
            continue
        for m in d["messages"]:
            if m.get("role") != "tool" or not m.get("tool_call"):
                continue
            text = B.tool_text(m)
            if not text or B.is_blocked(text):
                continue
            sents = B.sentences(text)
            if sents:
                yield d["suite_name"], d["user_task_id"], inj, m["tool_call"]["function"], sents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", nargs="+", required=True)
    ap.add_argument("--clean", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    groups = {"attack": list(outputs(args.attack, True)), "clean": list(outputs(args.clean, False))}
    uniq = sorted({s for rows in groups.values() for _a, _b, _c, _d, sents in rows for s in sents})
    print("scoring %d distinct sentences from %d attacked and %d clean outputs" % (
        len(uniq), len(groups["attack"]), len(groups["clean"])))
    scored = dict(zip(uniq, E.score_hf(args.model, uniq, args.batch)))

    for phase, rows in groups.items():
        print("\n== %s ==" % phase)
        stat = defaultdict(lambda: [0, 0, 0])
        samples = defaultdict(list)
        for suite, _ut, _inj, func, sents in rows:
            flags = [s for s in sents if scored[s] >= args.threshold]
            stat[suite][0] += 1
            stat[suite][1] += bool(flags)
            novel = [s for s in flags if not S.MARKER.search(s)]
            stat[suite][2] += bool(novel)
            if novel and len(samples[suite]) < 40:
                samples[suite].append((func, novel[0], scored[novel[0]]))
        print("%-10s %8s %18s %28s" % ("suite", "outputs", "any sentence flagged", "flagged, no marker"))
        for suite in sorted(stat):
            n, hit, new = stat[suite]
            print("%-10s %8d %8d %6.1f%% %12d %6.1f%%" % (suite, n, hit, 100.0 * hit / n, new, 100.0 * new / n))
        if phase == "attack":
            for suite in ("banking", "travel"):
                print("  %s, sentences flagged without a marker:" % suite)
                for func, sent, sc in samples[suite][: args.samples]:
                    print("    [%.3f] (%s) %s" % (sc, func, sent[:150].replace("\n", " ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
