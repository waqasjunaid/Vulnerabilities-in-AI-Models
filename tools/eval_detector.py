#!/usr/bin/env python3
"""Score the labelled sentences with a detector and report ranking quality per suite.

    # zero-shot with a local sequence-classification checkpoint (needs torch + transformers)
    python eval_detector.py --data data/detector_sentences.jsonl --model deberta-pi-v2 --eval-split all

    # lexical reference trained on the dev suites, evaluated on the held-out suites (needs scikit-learn)
    python eval_detector.py --data data/detector_sentences.jsonl --model tfidf --eval-split heldout

Reports AUROC, true-positive rate at 1% and 5% false-positive rate, and precision/recall/F1 at 0.5.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr[fpr <= fpr_target]
    return float(ok.max()) if len(ok) else 0.0


def report(name, y, s):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or y.sum() == len(y):
        print("%-10s pos=%-5d neg=%-6d (one class only)" % (name, int(y.sum()), int(len(y) - y.sum())))
        return
    pred = s >= 0.5
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print("%-10s pos=%-5d neg=%-6d AUROC=%.3f  TPR@1%%FPR=%.3f  TPR@5%%FPR=%.3f  P=%.3f R=%.3f F1=%.3f" % (
        name, int(y.sum()), int(len(y) - y.sum()), roc_auc_score(y, s), tpr_at(y, s, 0.01), tpr_at(y, s, 0.05),
        prec, rec, f1))


def score_hf(path, texts, batch):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if dev == "cuda":
        model.half()
    inj = [int(i) for i, lab in model.config.id2label.items() if str(lab).upper() in ("INJECTION", "LABEL_1", "1")]
    inj = inj[0] if inj else 1
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros(len(texts), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
            prob = torch.softmax(model(**enc).logits.float(), dim=-1)[:, inj].cpu().numpy()
            out[idx] = prob
    return out


def score_tfidf(train_rows, texts):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    tr_text = [r["sentence"] for r in train_rows]
    tr_y = [r["label"] for r in train_rows]
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True)
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    X = hstack([cv.fit_transform(tr_text), wv.fit_transform(tr_text)]).tocsr()
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(X, tr_y)
    Xt = hstack([cv.transform(texts), wv.transform(texts)]).tocsr()
    return clf.predict_proba(Xt)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="'tfidf' or a local checkpoint directory")
    ap.add_argument("--eval-split", default="all", choices=["all", "dev", "heldout"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dump")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    ev = [r for r in rows if args.eval_split == "all" or r["split"] == args.eval_split]
    texts = [r["sentence"] for r in ev]
    if args.model == "tfidf":
        train = [r for r in rows if r["split"] == "dev"]
        if args.eval_split != "heldout":
            print("note: tfidf is trained on the dev suites, so dev scores below are in-sample")
        scores = score_tfidf(train, texts)
    else:
        scores = score_hf(args.model, texts, args.batch)

    by = defaultdict(lambda: ([], []))
    for r, s in zip(ev, scores):
        by[r["suite"]][0].append(r["label"])
        by[r["suite"]][1].append(float(s))
    print("detector=%s  eval split=%s  sentences=%d" % (args.model, args.eval_split, len(ev)))
    for suite in sorted(by):
        report(suite, *by[suite])
    report("ALL", [r["label"] for r in ev], list(scores))
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for r, s in zip(ev, scores):
                fh.write(json.dumps({"suite": r["suite"], "label": r["label"], "score": float(s), "sentence": r["sentence"]},
                                    ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
