#!/usr/bin/env python3
"""Fine-tune the local DeBERTa injection detector on the dev suites and test on the held-out suites.

    python train_detector.py --data data/detector_sentences_v2.jsonl --base deberta-pi-v2 \
        --out detectors/span-v1 --max-dist 4 --epochs 3

Training and validation are split by user task, so sentences from one document never fall on
both sides. Banking and travel are never seen during training. Needs torch and transformers.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_spans as S  # noqa: E402


def tpr_at(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr[fpr <= target]
    return float(ok.max()) if len(ok) else 0.0


class SentDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        enc = self.tok(self.rows[i]["sentence"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.rows[i]["label"], dtype=torch.long)
        return item


def evaluate(model, rows, tok, max_len, device, batch):
    if not rows:
        return {}
    model.eval()
    probs = []
    loader = DataLoader(SentDataset(rows, tok, max_len), batch_size=batch)
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            p = torch.softmax(model(**b).logits.float(), dim=-1)[:, 1]
            probs.extend(p.cpu().tolist())
    y = np.array([r["label"] for r in rows])
    s = np.array(probs)
    out = {"n": len(rows), "pos": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["auroc"] = roc_auc_score(y, s)
        out["tpr@1%fpr"] = tpr_at(y, s, 0.01)
        out["tpr@5%fpr"] = tpr_at(y, s, 0.05)
    return out


def show(title, stats):
    if "auroc" not in stats:
        print("%-28s n=%d pos=%d (one class only)" % (title, stats.get("n", 0), stats.get("pos", 0)))
        return
    print("%-28s n=%-5d pos=%-5d AUROC=%.3f  TPR@1%%FPR=%.3f  TPR@5%%FPR=%.3f" % (
        title, stats["n"], stats["pos"], stats["auroc"], stats["tpr@1%fpr"], stats["tpr@5%fpr"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="deberta-pi-v2")
    ap.add_argument("--out", default="detectors/span-v1")
    ap.add_argument("--max-dist", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    dev = [r for r in rows if r["split"] == "dev"]
    held = [r for r in rows if r["split"] == "heldout"]
    tasks = sorted({(r["suite"], r["user_task"]) for r in dev})
    val_tasks = set(tasks[::5])
    train = [r for r in dev if (r["suite"], r["user_task"]) not in val_tasks]
    val = [r for r in dev if (r["suite"], r["user_task"]) in val_tasks]
    print("train=%d val=%d heldout=%d (val user tasks=%d)" % (len(train), len(val), len(held), len(val_tasks)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).to(device)
    n_pos = sum(r["label"] for r in train)
    weights = torch.tensor([1.0, (len(train) - n_pos) / max(1, n_pos)], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    loader = DataLoader(SentDataset(train, tok, args.max_len), batch_size=args.batch, shuffle=True)

    best, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                logits = model(**b).logits
                loss = loss_fn(logits, b["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(b["labels"])
        stats = evaluate(model, val, tok, args.max_len, device, args.batch)
        show("epoch %d val (loss %.3f)" % (epoch, total / len(train)), stats)
        if stats.get("auroc", -1) > best:
            best = stats["auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    print("\nheld-out suites, best validation epoch:")
    for suite in sorted({r["suite"] for r in held}):
        sub = [r for r in held if r["suite"] == suite]
        show(suite, evaluate(model, sub, tok, args.max_len, device, args.batch))
        goal = [r for r in sub if not S.MARKER.search(r["sentence"])]
        show(suite + " (no marker)", evaluate(model, goal, tok, args.max_len, device, args.batch))
    show("ALL held-out", evaluate(model, held, tok, args.max_len, device, args.batch))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
