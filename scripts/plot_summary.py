"""Generate the README summary chart from the stored benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "benchmark.csv"
OUTPUT = ROOT / "results" / "figures" / "rmse_by_horizon.png"

LABELS = {
    "random_walk": "Random walk with drift",
    "frozen_rates": "Frozen rates",
    "lc_resnet": "LC-ResNet",
    "lee_miller": "Lee-Miller",
    "lee_carter": "Lee-Carter",
    "lstm_kt": "LSTM on κₜ",
}
COLORS = ["#0072B2", "#999999", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]


def main() -> None:
    data = pd.read_csv(INPUT)
    subset = data[
        (data["metric"] == "rmse_log_mx")
        & (data["origin"] < 2019)
        & (data["model_name"].isin(LABELS))
    ]
    summary = subset.groupby(["model_name", "horizon"], as_index=False)["value"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    for (model, label), color in zip(LABELS.items(), COLORS, strict=True):
        series = summary[summary["model_name"] == model].sort_values("horizon")
        ax.plot(
            series["horizon"],
            series["value"],
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=label,
            color=color,
        )

    ax.set_title("Full-grid log-mortality forecast error", loc="left", weight="bold")
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel("Mean RMSE on log mortality rates")
    ax.set_xticks([1, 5, 10, 20])
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2)
    fig.text(
        0.01,
        0.01,
        "Unweighted mean over valid country–origin pairs; "
        "CBD excluded because it covers ages 60–100 only.",
        fontsize=8,
        color="#52606D",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
