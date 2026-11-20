#!/usr/bin/env python3
"""Rebuild the sentence labels from inserted blocks instead of raw differences.

A sentence is an injection only when it belongs to a contiguous block of text that was
inserted into a tool output and that block contains an injection marker. Inserted text
with no marker is excluded and reported, not labelled.

    python build_spans.py --clean RUN [RUN ...] --attack RUN [RUN ...] --out data/detector_sentences_v2.jsonl
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_detector_data as B  # noqa: E402

MARKER = re.compile(
    r"IMPORTANT\s*!*"
    r"|ignore\s+(?:all\s+)?(?:previous|prior|earlier)\s+instructions?"
    r"|following instruction"
    r"|you are now|new instructions?",
    re.I,
)


def label_output(text, refs):
    """Return (sentence, label) pairs and a count of excluded sentences."""
    sents = B.sentences(text)
    if text in refs:
        return [(s, 0) for s in sents], 0
    best = max(refs, key=lambda r: difflib.SequenceMatcher(
        None, B.sentences(r), sents, autojunk=False).ratio())
    sm = difflib.SequenceMatcher(None, B.sentences(best), sents, autojunk=False)
    groups, cur = [], []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            if cur:
                groups.append(cur)
                cur = []
            groups.append([(j, 0) for j in range(j1, j2)])
        else:
            cur.extend(range(j1, j2))
    if cur:
        groups.append(cur)
    out, excluded = [], 0
    for g in groups:
        if g and isinstance(g[0], tuple):
            out.extend((sents[j], 0) for j, _ in g)
            continue
        block = [sents[j] for j in g]
        if any(MARKER.search(s) for s in block):
            out.extend((s, 1) for s in block)
        else:
            excluded += len(block)
    return out, excluded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", nargs="+", required=True)
    ap.add_argument("--attack", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clean_ref = defaultdict(set)
    for d in B.records(args.clean):
        if d.get("injection_task_id"):
            continue
        for m in d["messages"]:
            if m.get("role") == "tool" and m.get("tool_call"):
                t = B.tool_text(m)
                if t and not B.is_blocked(t):
                    clean_ref[(d["suite_name"], d["user_task_id"]) + B.call_key(m)].add(t)

    rows, seen = [], set()
    stats = Counter()
    for d in B.records(args.attack):
        inj = d.get("injection_task_id")
        if not inj:
            continue
        suite, ut = d["suite_name"], d["user_task_id"]
        for m in d["messages"]:
            if m.get("role") != "tool" or not m.get("tool_call"):
                continue
            text = B.tool_text(m)
            if not text or B.is_blocked(text):
                continue
            refs = clean_ref.get((suite, ut) + B.call_key(m))
            if not refs:
                stats["outputs without a clean reference (skipped)"] += 1
                continue
            labelled, excluded = label_output(text, refs)
            stats["sentences excluded (inserted, no marker)"] += excluded
            func = m["tool_call"]["function"]
            for s, y in labelled:
                key = (suite, func, s, y)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"suite": suite, "split": B.SPLIT.get(suite, "dev"), "user_task": ut,
                             "injection_task": inj, "function": func, "sentence": s, "label": y})
                stats["sentences label=%d" % y] += 1
    for key, texts in clean_ref.items():
        suite, ut, func = key[0], key[1], key[2]
        for t in texts:
            for s in B.sentences(t):
                k = (suite, func, s, 0)
                if k not in seen:
                    seen.add(k)
                    rows.append({"suite": suite, "split": B.SPLIT.get(suite, "dev"), "user_task": ut,
                                 "injection_task": None, "function": func, "sentence": s, "label": 0})
                    stats["sentences label=0 (clean runs)"] += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))
    for k, v in sorted(stats.items()):
        print("  %-48s %d" % (k, v))
    by = Counter((r["suite"], r["label"]) for r in rows)
    for suite in sorted({r["suite"] for r in rows}):
        print("  %-10s split=%-8s positives=%-5d negatives=%-6d" % (
            suite, B.SPLIT.get(suite, "dev"), by[(suite, 1)], by[(suite, 0)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
