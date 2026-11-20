#!/usr/bin/env python3
"""Pick the detector threshold on the validation tasks and apply it to the held-out suites.

The threshold is the one that holds the false-positive rate to a target on the validation
tasks (workspace and slack user tasks held out of training). It is then frozen and applied
to banking and travel. Reports the same split for a TF-IDF model trained on the same
training tasks, and repeats the numbers on the sentences that carry no injection marker.

    python eval_operating_point.py --data data/detector_sentences_v2.jsonl --model detectors/span-v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_spans as S  # noqa: E402
import eval_detector as E  # noqa: E402


def split_rows(rows):
    dev = [r for r in rows if r["split"] == "dev"]
    held = [r for r in rows if r["split"] == "heldout"]
    tasks = sorted({(r["suite"], r["user_task"]) for r in dev})
    val_tasks = set(tasks[::5])
    train = [r for r in dev if (r["suite"], r["user_task"]) not in val_tasks]
    val = [r for r in dev if (r["suite"], r["user_task"]) in val_tasks]
    return train, val, held


def threshold_at_fpr(y, s, target):
    """Lowest score at which the false-positive rate is still within the target."""
    y, s = np.asarray(y), np.asarray(s)
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0:
        return 1.0
    allowed = max(1, int(np.floor(target * len(neg))))
    return float(neg[allowed - 1])


def rates(y, s, thr):
    y, s = np.asarray(y), np.asarray(s)
    pred = s >= thr
    pos, neg = y == 1, y == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr, int(pos.sum()), int(neg.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    train, val, held = split_rows(rows)
    print("train=%d val=%d (neg=%d) heldout=%d" % (
        len(train), len(val), sum(r["label"] == 0 for r in val), len(held)))

    panels = {
        "fine-tuned": E.score_hf(args.model, [r["sentence"] for r in val + held], args.batch),
        "tfidf": E.score_tfidf(train, [r["sentence"] for r in val + held]),
    }
    n_val = len(val)
    for name, scores in panels.items():
        val_s, held_s = scores[:n_val], scores[n_val:]
        val_y = [r["label"] for r in val]
        thr = threshold_at_fpr(val_y, val_s, args.fpr)
        print("\n%s  threshold=%.4f  (validation AUROC=%.3f)" % (
            name, thr, roc_auc_score(val_y, val_s)))
        for suite in sorted({r["suite"] for r in held}) + ["ALL"]:
            sub = [(r, sc) for r, sc in zip(held, held_s) if suite == "ALL" or r["suite"] == suite]
            for label, group in (("all", sub), ("no marker", [x for x in sub if not S.MARKER.search(x[0]["sentence"])])):
                if not group:
                    continue
                tpr, fpr, pos, neg = rates([r["label"] for r, _ in group], [sc for _, sc in group], thr)
                print("  %-10s %-10s pos=%-4d neg=%-4d TPR=%.3f  FPR=%.3f" % (suite, label, pos, neg, tpr, fpr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
