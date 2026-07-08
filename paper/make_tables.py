"""Generate the LaTeX table bodies for the paper from results/benchmark.csv.

Run from the repository root:
    python paper/make_tables.py
Prints each table body to stdout, ready to paste into main.tex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {
    "lee_carter": "Lee--Carter",
    "lee_miller": "Lee--Miller",
    "bms": "BMS",
    "poisson_lc": "Poisson LC",
    "cbd": "CBD (60+ only)",
    "hyndman_ullah": "Hyndman--Ullah",
    "random_walk": "Random walk",
    "frozen_rates": "Frozen rates",
    "lstm_kt": r"LSTM $\kappa_t$",
    "gru_kt": r"GRU $\kappa_t$",
    "bilstm_kt": r"Bi-LSTM $\kappa_t$",
    "transformer_kt": r"Transformer $\kappa_t$",
    "ffnn_embeddings": "FFNN embeddings",
    "cnn_surface": "CNN surface",
    "lc_resnet": r"\textbf{LC-ResNet}",
}

GROUPS = [
    ("Classical", ["lee_carter", "lee_miller", "bms", "poisson_lc", "cbd", "hyndman_ullah"]),
    ("Baselines", ["random_walk", "frozen_rates"]),
    ("Neural", ["lstm_kt", "gru_kt", "bilstm_kt", "transformer_kt",
                "ffnn_embeddings", "cnn_surface"]),
    ("Hybrid", ["lc_resnet"]),
]


def fmt(v: float, best: float, exclude_bold: bool = False) -> str:
    if np.isnan(v):
        return "--"
    s = f"{v:.3f}"
    if not exclude_bold and abs(v - best) < 5e-4:
        return rf"\textbf{{{s}}}"
    return s


def emit(pivot: pd.DataFrame, horizons: list[int], bold_exclude: set[str] | None = None) -> None:
    bold_exclude = bold_exclude or set()
    # Best per column among full-grid models only (CBD excluded when told to).
    eligible = [m for m in pivot.index if m not in bold_exclude]
    best = {h: pivot.loc[eligible, h].min() for h in horizons if h in pivot.columns}
    for group, models in GROUPS:
        print(rf"\multicolumn{{{len(horizons)+1}}}{{l}}{{\textit{{{group}}}}} \\")
        for m in models:
            if m not in pivot.index:
                continue
            cells = [fmt(pivot.loc[m, h], best.get(h, np.nan), m in bold_exclude)
                     if h in pivot.columns else "--" for h in horizons]
            print(f"{LABELS[m]:<28} & " + " & ".join(cells) + r" \\")
        print(r"\midrule")


def main() -> None:
    df = pd.read_csv("results/benchmark.csv")

    print("=" * 30, "TABLE 1: RMSE log mx (rolling origins only, origin<2019)")
    main_df = df[(df.metric == "rmse_log_mx") & (df.origin < 2019)]
    p = main_df.groupby(["model_name", "horizon"])["value"].mean().unstack()
    emit(p, [1, 5, 10, 20], bold_exclude={"cbd"})

    print("=" * 30, "TABLE 2: actuarial metrics (e0 / annuity), origin<2019")
    for metric in ["rmse_e0", "rmse_annuity_65"]:
        print(f"--- {metric} ---")
        sub = df[(df.metric == metric) & (df.origin < 2019)]
        p = sub.groupby(["model_name", "horizon"])["value"].mean().unstack()
        best5 = p[5].min() if 5 in p.columns else np.nan
        best20 = p[20].min() if 20 in p.columns else np.nan
        for _, models in GROUPS:
            for m in models:
                if m not in p.index:
                    continue
                c5 = fmt(p.loc[m, 5], best5)
                c20 = fmt(p.loc[m, 20], best20)
                print(f"{LABELS[m]:<28} & {c5} & {c20}")

    print("=" * 30, "TABLE 3: COVID shock (origin 2019), h=1..3 -> 2020..2022")
    shock = df[(df.origin == 2019) & (df.metric == "rmse_log_mx")
               & (df.horizon <= 3)]
    p = shock.groupby(["model_name", "horizon"])["value"].mean().unstack()
    emit(p, [1, 2, 3], bold_exclude={"cbd"})

    print("=" * 30, "TABLE 4: short history (rmse_log_mx_hist{20,50})")
    for n in [20, 50]:
        sub = df[df.metric == f"rmse_log_mx_hist{n}"]
        s = sub.groupby("model_name")["value"].mean()
        print(f"--- hist{n} ---")
        best = s.min()
        for m, v in s.sort_values().items():
            print(f"{LABELS[m]:<28} & {fmt(v, best)}")

    print("=" * 30, "TABLE 5: age groups (young / elderly)")
    for grp in ["young", "elderly"]:
        sub = df[df.metric == f"rmse_log_mx_{grp}"]
        s = sub.groupby("model_name")["value"].mean().dropna()
        print(f"--- {grp} ---")
        best = s.min()
        for m, v in s.sort_values().items():
            print(f"{LABELS[m]:<28} & {fmt(v, best)}")

    print("=" * 30, "TRAIN TIMES (mean s)")
    tt = df.groupby("model_name")["train_time"].mean().sort_values()
    for m, v in tt.items():
        print(f"{LABELS[m]:<28} & {v:.2f}")


if __name__ == "__main__":
    main()
