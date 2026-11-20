#!/usr/bin/env python3
"""Sanity-check the diff-derived injection labels before training on them.

    python audit_labels.py data/detector_sentences.jsonl [--samples 10]

Prints, per suite: how many positives look like instructions, the most repeated positive
openings, and random examples of positives without any instruction cue (possible label noise).
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict

CUE = re.compile(r"ignore|previous instruction|important|instruction|you must|please (do|send|make|transfer|delete|forward)|"
                 r"before you|first,? |do the following|strictly|task|kindly", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    random.seed(0)
    by = defaultdict(list)
    for r in rows:
        by[r["suite"]].append(r)
    for suite, rs in sorted(by.items()):
        pos = [r for r in rs if r["label"] == 1]
        neg = [r for r in rs if r["label"] == 0]
        cue = [r for r in pos if CUE.search(r["sentence"])]
        neg_cue = [r for r in neg if CUE.search(r["sentence"])]
        print("=" * 100)
        print("%s: positives=%d (with an instruction cue: %d = %.0f%%)  negatives=%d (with a cue: %d = %.0f%%)" % (
            suite, len(pos), len(cue), 100.0 * len(cue) / max(1, len(pos)), len(neg), len(neg_cue),
            100.0 * len(neg_cue) / max(1, len(neg))))
        opening = Counter(" ".join(r["sentence"].split()[:6]) for r in pos)
        print("  most repeated positive openings:")
        for text, n in opening.most_common(6):
            print("    %4d x %s" % (n, text[:90]))
        funcs = Counter(r["function"] for r in pos)
        print("  positives by tool:", dict(funcs.most_common(6)))
        odd = [r for r in pos if not CUE.search(r["sentence"])]
        print("  positives WITHOUT an instruction cue (%d), random sample:" % len(odd))
        for r in random.sample(odd, min(args.samples, len(odd))):
            print("    [%s] %s" % (r["function"], r["sentence"][:140].replace("\n", " ")))
        hard = random.sample(neg_cue, min(4, len(neg_cue)))
        if hard:
            print("  negatives that contain an instruction cue (hard negatives):")
            for r in hard:
                print("    [%s] %s" % (r["function"], r["sentence"][:140].replace("\n", " ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
