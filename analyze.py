"""Сводка по results/results.csv: средние по типам, парные тесты, коллапс.

    python analyze.py                      # neg-mode clean
    python analyze.py --neg-mode legacy    # протокол статьи

Пишет results/summary_<neg-mode>.md и печатает его же.
"""
import argparse
import warnings
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ["reaction", "binding", "ptmod", "activation", "inhibition", "catalysis", "expression"]
PAIRS = [("spectral", "none"), ("diffusion", "none"), ("spectral", "diffusion"),
         ("proj", "none"), ("spectral", "proj"), ("diffusion", "proj")]
COLLAPSE_F1 = 0.01


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def one_sided(a, b):
    """H1: a > b. Возвращает (p Уилкоксона, p t-теста)."""
    d = a - b
    if np.allclose(d, 0):
        return 1.0, 1.0
    return (wilcoxon(a, b, alternative="greater").pvalue,
            ttest_rel(a, b, alternative="greater").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "results", "results.csv"))
    ap.add_argument("--neg-mode", default="clean")
    ap.add_argument("--split-seed", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.neg_mode == args.neg_mode]
    if args.split_seed is not None:
        df = df[df.split_seed == args.split_seed]
    key = ["split_seed", "encoder", "feature_mode", "ppimode", "seed"]
    df = df.drop_duplicates(key, keep="last")
    modes = [m for m in ["none", "proj", "diffusion", "spectral"] if m in set(df.feature_mode)]

    out = [f"# Сводка: neg-mode = {args.neg_mode}", ""]
    out.append("Запусков: " + ", ".join(f"{m} — {(df.feature_mode == m).sum()}" for m in modes))
    out.append("")

    # 1. Средние по типам
    out.append("## ROC-AUC и AP по типам (среднее ± std)")
    out.append("")
    out.append("| Тип | " + " | ".join(f"ROC {m}" for m in modes) + " | " + " | ".join(f"AP {m}" for m in modes) + " |")
    out.append("|" + "---|" * (1 + 2 * len(modes)))
    for t in TYPES + ["Среднее"]:
        sub = df if t == "Среднее" else df[df.ppimode == t]
        cells = []
        for metric in ["test_roc", "test_ap"]:
            for m in modes:
                v = sub[sub.feature_mode == m][metric]
                cells.append(f"{v.mean():.3f}" if t == "Среднее" else f"{v.mean():.3f} ± {v.std():.3f}")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    out.append("")

    # 2. Парные сравнения
    out.append("## Парные сравнения (односторонний Уилкоксон, H1: первая лучше)")
    out.append("")
    out.append("| Сравнение | n | ΔROC | p | ΔAP | p | ΔF1 | p | ΔP@50 | p |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    per_type = {}
    for a, b in PAIRS:
        if a not in modes or b not in modes:
            continue
        A = df[df.feature_mode == a].set_index(["ppimode", "seed"]).sort_index()
        B = df[df.feature_mode == b].set_index(["ppimode", "seed"]).sort_index()
        idx = A.index.intersection(B.index)
        A, B = A.loc[idx], B.loc[idx]
        cells = []
        for metric in ["test_roc", "test_ap", "test_f1", "test_p_at_50"]:
            pw, _ = one_sided(A[metric].values, B[metric].values)
            cells += [f"{(A[metric] - B[metric]).mean():+.4f}", f"{pw:.1e}{stars(pw)}"]
        out.append(f"| {a} vs {b} | {len(idx)} | " + " | ".join(cells) + " |")
        rows = []
        for t in TYPES:
            if t not in A.index.get_level_values(0):
                continue
            a_t, b_t = A.loc[t], B.loc[t]
            pw, _ = one_sided(a_t.test_roc.values, b_t.test_roc.values)
            rows.append((t, (a_t.test_roc - b_t.test_roc).mean(), pw))
        per_type[(a, b)] = rows
    out.append("")

    out.append("## ΔROC-AUC по типам")
    out.append("")
    hdr = [f"{a} vs {b}" for (a, b) in per_type]
    out.append("| Тип | " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * (1 + len(hdr)))
    for i, t in enumerate(TYPES):
        cells = []
        for rows in per_type.values():
            r = [x for x in rows if x[0] == t]
            cells.append(f"{r[0][1]:+.3f}{stars(r[0][2])}" if r else "")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    out.append("")

    # 3. Коллапс и калибровка
    out.append(f"## Коллапс классов (F1 < {COLLAPSE_F1}) и лучшая эпоха")
    out.append("")
    out.append("| Тип | " + " | ".join(f"коллапс {m}" for m in modes) + " | " + " | ".join(f"эпоха {m}" for m in modes) + " |")
    out.append("|" + "---|" * (1 + 2 * len(modes)))
    for t in TYPES + ["Всего"]:
        sub = df if t == "Всего" else df[df.ppimode == t]
        c = [str(int((sub[sub.feature_mode == m].test_f1 < COLLAPSE_F1).sum())) for m in modes]
        e = [f"{sub[sub.feature_mode == m].best_epoch.median():.0f}" for m in modes]
        out.append(f"| {t} | " + " | ".join(c) + " | " + " | ".join(e) + " |")
    out.append("")
    early = df[df.best_epoch <= 2]
    out.append(f"Запусков с лучшей эпохой 1–2: {len(early)} "
               + "(" + ", ".join(f"{m}: {(early.feature_mode == m).sum()}" for m in modes) + ")")
    out.append("")

    text = "\n".join(out)
    path = os.path.join(os.path.dirname(os.path.abspath(args.csv)), f"summary_{args.neg_mode}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()
