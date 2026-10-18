"""Один запуск: предсказание рёбер одного типа PPI на SHS27k.

Цикл обучения тот же, что train_sgwt_gat() в исходном train.py:
full-batch, Adam, BCE с логитами, ранняя остановка по валидационному
ROC-AUC, метрики теста берутся с лучшей по валидации эпохи.

Пример:
    python train.py --ppimode reaction --feature-mode spectral --seed 1
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from sgwt_ppi.data import load_data
from sgwt_ppi.models import UnifiedGNN

HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = [
    "neg_mode", "split_seed", "encoder", "feature_mode", "ppitype", "ppimode", "seed",
    "best_epoch", "stopped_epoch", "n_params", "seconds",
    "val_roc", "val_ap", "val_acc", "val_precision", "val_recall", "val_f1",
    "val_p_at_50", "val_r_at_50", "val_p_at_100", "val_r_at_100",
    "test_roc", "test_ap", "test_acc", "test_precision", "test_recall", "test_f1",
    "test_p_at_50", "test_r_at_50", "test_p_at_100", "test_r_at_100",
]
RUN_KEY = ["neg_mode", "split_seed", "encoder", "feature_mode", "ppitype", "ppimode", "seed"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ppimode", required=True,
                   choices=["reaction", "binding", "ptmod", "activation",
                            "inhibition", "catalysis", "expression"])
    p.add_argument("--feature-mode", default="spectral",
                   choices=["none", "proj", "spectral", "diffusion"])
    p.add_argument("--encoder", default="gat", choices=["gat", "gcn", "sage", "gin"])
    p.add_argument("--neg-mode", default="clean", choices=["clean", "legacy"],
                   help="clean: val/test пары не попадают в обучающие негативы; "
                        "legacy: как в статье (ошибка из HGCN)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=1234)
    p.add_argument("--ppitype", default="SHS27")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data", "ppi"))
    p.add_argument("--out", default=os.path.join(HERE, "results", "results.csv"))
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--cpu", action="store_true", help="считать на CPU даже при наличии GPU")
    p.add_argument("--skip-existing", action="store_true",
                   help="не запускать, если такой запуск уже есть в --out")
    return p.parse_args()


def edge_index_from_train(edges, device):
    ei = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return torch.cat([ei, ei[[1, 0], :]], dim=1)  # граф неориентированный


def label_index(pos, neg, device):
    idx = torch.cat([torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long)])
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    return idx.t().contiguous().to(device), y.to(device)


def precision_recall_at_k(y, p, k):
    k = min(k, len(y))
    top = np.argsort(p)[::-1][:k]
    tp = y[top].sum()
    return float(tp / k), float(tp / y.sum()) if y.sum() > 0 else 0.0


def metrics(logits, labels, threshold):
    p = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    pred = (p >= threshold).astype(int)
    p50, r50 = precision_recall_at_k(y, p, 50)
    p100, r100 = precision_recall_at_k(y, p, 100)
    return {
        "roc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
        "acc": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "p_at_50": p50, "r_at_50": r50, "p_at_100": p100, "r_at_100": r100,
    }


def already_done(path, key):
    if not os.path.isfile(path):
        return False
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all(str(row.get(k)) == str(v) for k, v in key.items()):
                return True
    return False


def main():
    args = parse_args()
    key = {"neg_mode": args.neg_mode, "split_seed": args.split_seed, "encoder": args.encoder,
           "feature_mode": args.feature_mode, "ppitype": args.ppitype,
           "ppimode": args.ppimode, "seed": args.seed}
    if args.skip_existing and already_done(args.out, key):
        print("skip (already in results):", key)
        return

    # Тот же порядок инициализации генераторов, что в исходном train.py
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data = load_data(args.data_dir, args.ppitype, args.ppimode,
                     split_seed=args.split_seed, neg_mode=args.neg_mode)
    x = torch.tensor(data["features"], dtype=torch.float32, device=device)
    train_ei = edge_index_from_train(data["train_edges"], device)
    tr_idx, tr_y = label_index(data["train_edges"], data["train_edges_false"], device)
    va_idx, va_y = label_index(data["val_edges"], data["val_edges_false"], device)
    te_idx, te_y = label_index(data["test_edges"], data["test_edges_false"], device)

    model = UnifiedGNN(
        in_channels=x.size(1), encoder_name=args.encoder, feature_mode=args.feature_mode,
        hidden_channels=64, out_channels=64, heads=4, dropout=0.3,
        spectral_scales=(0.5, 1.0, 2.0, 4.0), diffusion_scales=(1, 2, 3, 4), cheb_order=5,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = int(sum(p.numel() for p in model.parameters()))

    print(f"{args.ppimode} | {args.feature_mode} | seed {args.seed} | neg {args.neg_mode} | "
          f"train +{len(data['train_edges'])} / -{len(data['train_edges_false'])} | "
          f"test {len(data['test_edges'])}+{len(data['test_edges_false'])} | params {n_params} | {device}")

    best_val, best_epoch, counter = -1.0, 0, 0
    best_val_m = best_test_m = None
    t0 = time.time()
    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x, train_ei, tr_idx), tr_y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_ei)
            val_m = metrics(model.decode(z, va_idx), va_y, args.threshold)
            if val_m["roc"] > best_val:
                best_val, best_epoch, counter = val_m["roc"], epoch + 1, 0
                best_val_m = val_m
                best_test_m = metrics(model.decode(z, te_idx), te_y, args.threshold)
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    break

    row = dict(key, best_epoch=best_epoch, stopped_epoch=epoch + 1, n_params=n_params,
               seconds=round(time.time() - t0, 1))
    row.update({f"val_{k}": v for k, v in best_val_m.items()})
    row.update({f"test_{k}": v for k, v in best_test_m.items()})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.isfile(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"  best epoch {best_epoch} | test ROC {best_test_m['roc']:.4f} "
          f"AP {best_test_m['ap']:.4f} F1 {best_test_m['f1']:.4f} | {row['seconds']} s")


if __name__ == "__main__":
    main()
