"""Reproduce the French 2033 annuity provision and longevity-shock case study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mortality.actuarial.case_study import longevity_risk_analysis, price_annuity_portfolio
from mortality.data.loader import load_country
from mortality.models.classical import CLASSICAL_MODELS
from mortality.models.hybrid import HYBRID_MODELS

MODEL_NAMES = ["lee_carter", "poisson_lc", "lc_resnet", "random_walk", "cbd"]


def projected_rates(horizon: int = 10) -> tuple[dict[str, np.ndarray], int]:
    """Fit the five case-study models and return rates in the final forecast year."""
    data = load_country("FRATNP")
    factories = {**CLASSICAL_MODELS, **HYBRID_MODELS}
    mx_by_model: dict[str, np.ndarray] = {}

    for name in MODEL_NAMES:
        model = factories[name]()
        model.fit(
            data["log_mx"],
            data["ages"],
            data["years"],
            data.get("exposures"),
            data.get("deaths"),
        )
        forecast = model.forecast(horizon)
        if name == "cbd":
            mx = data["mx"][:, -1].copy()
            positions = np.searchsorted(data["ages"], model.forecast_ages)
            mx[positions] = np.exp(forecast[:, -1])
        else:
            mx = np.exp(forecast[:, -1])
        mx_by_model[name] = mx

    return mx_by_model, int(data["years"][-1] + horizon)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output", default="results/case_study.csv")
    args = parser.parse_args()

    mx_by_model, forecast_year = projected_rates(args.horizon)
    pricing = price_annuity_portfolio(mx_by_model)
    risk = longevity_risk_analysis(mx_by_model)
    output = pricing.merge(risk, on="model", validate="one_to_one")
    output.insert(1, "forecast_year", forecast_year)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
