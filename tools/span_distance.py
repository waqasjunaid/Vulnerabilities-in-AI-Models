#!/usr/bin/env python3
"""How far is each positive sentence from the injection marker in its block?

Distance 0 carries the marker, distance 1-4 is usually the attack goal, and a long tail of
large distances is benign text that the block happened to swallow.

    python span_distance.py --clean RUN [RUN ...] --attack RUN [RUN ...] [--samples 6]
"""
import argparse
import difflib
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_detector_data as B  # noqa: E402
import build_spans as S  # noqa: E402


def block_sentences(text, refs):
    """Yield (sentence, distance) for sentences inside marker blocks."""
    sents = B.sentences(text)
    if text in refs:
        return
    best = max(refs, key=lambda r: difflib.SequenceMatcher(
        None, B.sentences(r), sents, autojunk=False).ratio())
    sm = difflib.SequenceMatcher(None, B.sentences(best), sents, autojunk=False)
    runs, cur = [], []
    for tag, _a, _b, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.extend(range(j1, j2))
    if cur:
        runs.append(cur)
    for run in runs:
        block = [sents[j] for j in run]
        markers = [k for k, s in enumerate(block) if S.MARKER.search(s)]
        if not markers:
            continue
        for k, s in enumerate(block):
            yield s, min(abs(k - m) for m in markers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", nargs="+", required=True)
    ap.add_argument("--attack", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()
    random.seed(0)

    refs = defaultdict(set)
    for d in B.records(args.clean):
        if d.get("injection_task_id"):
            continue
        for m in d["messages"]:
            if m.get("role") == "tool" and m.get("tool_call"):
                t = B.tool_text(m)
                if t and not B.is_blocked(t):
                    refs[(d["suite_name"], d["user_task_id"]) + B.call_key(m)].add(t)

    dist = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(list))
    for d in B.records(args.attack):
        if not d.get("injection_task_id"):
            continue
        suite, ut = d["suite_name"], d["user_task_id"]
        for m in d["messages"]:
            if m.get("role") != "tool" or not m.get("tool_call"):
                continue
            text = B.tool_text(m)
            if not text or B.is_blocked(text):
                continue
            ref = refs.get((suite, ut) + B.call_key(m))
            if not ref:
                continue
            for s, k in block_sentences(text, ref):
                dist[suite][k] += 1
                if len(examples[suite][k]) < 40:
                    examples[suite][k].append(s)

    for suite in sorted(dist):
        c = dist[suite]
        total = sum(c.values())
        near = sum(v for k, v in c.items() if k <= 4)
        print("=" * 100)
        print("%s: %d positive sentences, %d (%.0f%%) within distance 4 of a marker" % (
            suite, total, near, 100.0 * near / total))
        print("  distance histogram:", " ".join("%d:%d" % (k, c[k]) for k in sorted(c)[:12]))
        for label, bucket in (("distance 1 (the attack goal)", 1), ("distance >= 6 (likely benign)", 6)):
            pool = []
            for k in sorted(examples[suite]):
                if (bucket == 1 and k == 1) or (bucket == 6 and k >= 6):
                    pool.extend(examples[suite][k])
            print("  %s, %d examples:" % (label, len(pool)))
            for s in random.sample(pool, min(args.samples, len(pool))):
                print("    %s" % s[:150].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
