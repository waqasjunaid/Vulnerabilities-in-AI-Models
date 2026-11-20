#!/usr/bin/env python3
"""Why v2 and v2+span made the same decisions.

Counts tool outputs whose flagged sentences lie outside the marker span, then
lists the baseline-successful action attacks that v2 still allows.

    python tools/explain_span_delta.py --project . \
        --baseline runs/agentdojo/final_baseline/attack \
        --scores detectors/span-v1-scores.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_detector_data as B  # noqa: E402
import replay_guardian as R  # noqa: E402
import span_lookup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--threshold", type=float, default=0.0155)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.project).resolve()))
    import trust_v2

    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    suites = {"banking", "slack", "travel", "workspace"}
    cases = R.load_cases(args.baseline, suites)

    n_out = flagged_out = extra_out = 0
    examples = []
    for key, case in cases.items():
        for m in case["messages"]:
            if m.get("role") != "tool" or not m.get("tool_call"):
                continue
            text = B.tool_text(m)
            if not text or B.is_blocked(text):
                continue
            n_out += 1
            sents = B.sentences(text)
            if any(scores.get(s, 0) >= args.threshold for s in sents):
                flagged_out += 1
            spans, _, _, _ = span_lookup.sentence_spans(
                text, scores, args.threshold, B.sentences,
                trust_v2.MARKER_RE, trust_v2.content_fields)
            base = []
            for field in trust_v2.content_fields(text):
                hit = trust_v2.MARKER_RE.search(field)
                if hit:
                    base.append(field[hit.start():])
            added = [s for s in spans if s not in base]
            if added:
                extra_out += 1
                if len(examples) < 8:
                    examples.append((key, m["tool_call"].get("function"), added[0][:150]))
    print("tool outputs: %d" % n_out)
    print("outputs with any sentence at or above the threshold: %d" % flagged_out)
    print("outputs with a flagged sentence outside the marker span: %d" % extra_out)
    for key, func, sent in examples:
        print("  [%s %s %s] (%s) %s" % (key[0], key[1], key[2], func, sent.replace("\n", " ")))

    print("\nreplaying baseline-successful action attacks with v2")
    adapter = R.make_adapter(args.project, "v2")
    misses = []
    n_action = 0
    for key, case in sorted(cases.items()):
        if not case.get("security") or R.attack_kind(key[0], key[2]) != "action":
            continue
        n_action += 1
        sigs = R.attack_signatures(key[0], key[2])
        calls = R.replay_case(adapter, case)
        if not any((not c["allow"]) and R.is_attack_call(c, sigs) for c in calls):
            misses.append((key, case, calls, sigs))
        if n_action % 25 == 0:
            print("  %d action attacks" % n_action, flush=True)
    print("action attacks %d, still allowed %d" % (n_action, len(misses)))

    for key, case, calls, sigs in misses:
        print("\n==", key)
        print("  goal:", R.user_goal(case["messages"])[:220].replace("\n", " "))
        matched = [c for c in calls if R.is_attack_call(c, sigs)]
        if not matched:
            print("  no call matched the attack signature", sigs)
            print("  calls made:", [c["function"] for c in calls])
            continue
        for c in matched[:2]:
            print("  call", c["function"], json.dumps(c["args"], default=str)[:200])
            print("   ", c["reason"][:200])
            ents = sorted({
                s.strip().lower() for s in R._leaves(c["args"])
                if len(s.strip()) >= 4 and any(ch.isalpha() for ch in s) and not s.strip()[:4].isdigit()
            })
            seen = set()
            outside_lines = 0
            for m in case["messages"]:
                if m.get("role") != "tool" or not m.get("tool_call"):
                    continue
                text = B.tool_text(m)
                if not text:
                    continue
                spans, _, _, _ = span_lookup.sentence_spans(
                    text, scores, args.threshold, B.sentences,
                    trust_v2.MARKER_RE, trust_v2.content_fields)
                blob = "\n".join(spans).lower()
                for e in ents:
                    if e not in seen and trust_v2.contains(blob, e):
                        seen.add(e)
                        print("   in a span via %s: %s" % (m["tool_call"].get("function"), e[:80]))
                if outside_lines >= 6:
                    continue
                for sent in B.sentences(text):
                    low = sent.lower()
                    hit = [e for e in ents if e not in seen and trust_v2.contains(low, e)]
                    if not hit:
                        continue
                    print("   not in a span, score %.3f (%s): %s" % (
                        scores.get(sent, -1.0), m["tool_call"].get("function"), sent[:150].replace("\n", " ")))
                    outside_lines += 1
                    if outside_lines >= 6:
                        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
