#!/usr/bin/env python3
"""Score every recorded tool-output sentence once and write a lookup table.

The replay reads this table, so the benchmark environment never loads the model.

    python export_span_scores.py --runs runs/agentdojo/final_baseline/attack \
        runs/agentdojo/final_baseline/clean --model detectors/span-v1 \
        --out detectors/span-v1-scores.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_detector_data as B  # noqa: E402
import eval_detector as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    found = set()
    n_out = 0
    for d in B.records(args.runs):
        for m in d["messages"]:
            if m.get("role") != "tool" or not m.get("tool_call"):
                continue
            text = B.tool_text(m)
            if not text or B.is_blocked(text):
                continue
            sents = B.sentences(text)
            if not sents:
                continue
            n_out += 1
            found.update(sents)
    sents = sorted(found)
    print("scoring %d distinct sentences from %d tool outputs" % (len(sents), n_out))
    scores = E.score_hf(args.model, sents, args.batch)
    payload = {s: round(float(sc), 6) for s, sc in zip(sents, scores)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    hot = sum(sc >= 0.0155 for sc in payload.values())
    print("wrote %s (%d scores, %d at or above 0.0155)" % (args.out, len(payload), hot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
