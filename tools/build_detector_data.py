#!/usr/bin/env python3
"""Build sentence-level injection labels from AgentDojo run logs, without using attack templates.

For every attacked case, each tool output is compared with the output of the same tool call
(same function and arguments) in a clean run of the same user task. Sentences present only in
the attacked output are the injection (label 1); every other sentence is label 0.

    python build_detector_data.py --clean RUN [RUN ...] --attack RUN [RUN ...] --out data/detector.jsonl
        [--calls data/calls.jsonl --project /path/to/guardian_lab]

Rows carry suite and injection-task ids so the train/dev/held-out split can be made by suite.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLIT = {"workspace": "dev", "slack": "dev", "banking": "heldout", "travel": "heldout"}
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    out = []
    for s in _SENT.split(text or ""):
        s = s.strip(" \t-*")
        if len(s) >= 3:
            out.append(s)
    return out


def tool_text(msg) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(str(p.get("content", "")) for p in c if isinstance(p, dict))
    return str(c or "")


def call_key(msg):
    tc = msg.get("tool_call") or {}
    return tc.get("function"), json.dumps(tc.get("args") or {}, sort_keys=True, default=str)


def records(roots):
    for root in roots:
        for f in Path(root).rglob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and "suite_name" in d and "messages" in d:
                if str(d.get("user_task_id", "")).startswith("injection"):
                    continue
                yield d


def is_blocked(text: str) -> bool:
    return "GuardianBlocked" in text or "[BLOCKED by guardian" in text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", nargs="+", required=True)
    ap.add_argument("--attack", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", help="also write call-level labels (needs agentdojo and tools/replay_guardian.py)")
    ap.add_argument("--project", default=".")
    args = ap.parse_args()

    clean_ref = defaultdict(set)
    for d in records(args.clean):
        if d.get("injection_task_id"):
            continue
        for m in d["messages"]:
            if m.get("role") == "tool" and m.get("tool_call"):
                t = tool_text(m)
                if t and not is_blocked(t):
                    clean_ref[(d["suite_name"], d["user_task_id"]) + call_key(m)].add(t)

    rows, seen = [], set()
    stats = Counter()
    injections = defaultdict(set)
    for d in records(args.attack):
        inj = d.get("injection_task_id")
        if not inj:
            continue
        suite, ut = d["suite_name"], d["user_task_id"]
        for m in d["messages"]:
            if m.get("role") != "tool" or not m.get("tool_call"):
                continue
            text = tool_text(m)
            if not text or is_blocked(text):
                continue
            refs = clean_ref.get((suite, ut) + call_key(m))
            if not refs:
                stats["tool outputs without a clean reference (skipped)"] += 1
                continue
            sents = sentences(text)
            ref_sents = set()
            for r in refs:
                ref_sents.update(sentences(r))
            if text in refs:
                labels = [0] * len(sents)
            else:
                best = max(refs, key=lambda r: difflib.SequenceMatcher(None, sentences(r), sents, autojunk=False).ratio())
                sm = difflib.SequenceMatcher(None, sentences(best), sents, autojunk=False)
                labels = [1] * len(sents)
                for tag, _, _, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        for j in range(j1, j2):
                            labels[j] = 0
                labels = [0 if sents[i] in ref_sents else labels[i] for i in range(len(sents))]
            func = m["tool_call"]["function"]
            for s, y in zip(sents, labels):
                key = (suite, func, s, y)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"suite": suite, "split": SPLIT.get(suite, "dev"), "user_task": ut,
                             "injection_task": inj, "function": func, "sentence": s, "label": y})
                stats["sentences label=%d" % y] += 1
                if y:
                    injections[suite].add(hashlib.md5(s.encode()).hexdigest()[:8])
    for key, texts in clean_ref.items():
        suite, ut, func = key[0], key[1], key[2]
        for t in texts:
            for s in sentences(t):
                k = (suite, func, s, 0)
                if k not in seen:
                    seen.add(k)
                    rows.append({"suite": suite, "split": SPLIT.get(suite, "dev"), "user_task": ut,
                                 "injection_task": None, "function": func, "sentence": s, "label": 0})
                    stats["sentences label=0 (clean runs)"] += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))
    for k, v in sorted(stats.items()):
        print("  %-52s %d" % (k, v))
    by = Counter((r["suite"], r["label"]) for r in rows)
    for suite in sorted({r["suite"] for r in rows}):
        print("  %-10s split=%-8s positives=%-5d negatives=%-6d distinct injected sentences=%d"
              % (suite, SPLIT.get(suite, "dev"), by[(suite, 1)], by[(suite, 0)], len(injections[suite])))

    if args.calls:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import replay_guardian as R
        n = Counter()
        with open(args.calls, "w", encoding="utf-8") as fh:
            for d in records(args.attack):
                inj = d.get("injection_task_id")
                if not inj:
                    continue
                sigs = R.attack_signatures(d["suite_name"], inj)
                for i, m in enumerate(x for x in d["messages"] if x.get("role") == "tool" and x.get("tool_call")):
                    tc = m["tool_call"]
                    call = {"function": tc["function"], "args": tc.get("args") or {}}
                    y = int(R.is_attack_call(call, sigs))
                    fh.write(json.dumps({"suite": d["suite_name"], "split": SPLIT.get(d["suite_name"], "dev"),
                                         "user_task": d["user_task_id"], "injection_task": inj, "step": i,
                                         "function": call["function"], "args": call["args"], "label": y,
                                         "blocked": is_blocked(tool_text(m))}, ensure_ascii=False) + "\n")
                    n[(d["suite_name"], y)] += 1
        print("wrote call labels to %s: %s" % (args.calls, dict(n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
