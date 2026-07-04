"""Run the full benchmark pipeline.

Usage:
    python scripts/run_benchmark.py                          # all countries, full
    python scripts/run_benchmark.py --countries FRATNP --quick  # 1 country, fast mode
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import yaml

from mortality.data.loader import load_country
from mortality.evaluation.rolling_origin import rolling_origin_eval
from mortality.evaluation.scenarios import mortality_shock_eval
from mortality.models.classical import CLASSICAL_MODELS
from mortality.models.neural import NEURAL_MODELS


def get_all_model_factories(quick: bool = False) -> dict[str, callable]:
    """Return a dict of model_name -> factory callable."""
    factories = {}

    for name, cls in CLASSICAL_MODELS.items():
        factories[name] = cls

    neural_kw = {"epochs": 10, "patience": 5} if quick else {"epochs": 200, "patience": 20}
    for name, factory in NEURAL_MODELS.items():
        factories[name] = lambda f=factory, kw=neural_kw: f(**kw)

    return factories


def main() -> None:
    parser = argparse.ArgumentParser(description="Mortality benchmark")
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", default="results/benchmark.csv")
    args = parser.parse_args()

    with open("config/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("config/evaluation.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    country_codes = args.countries or [c["code"] for c in data_cfg["countries"]]
    origins = list(range(
        eval_cfg["rolling_origin"]["first_origin"],
        eval_cfg["rolling_origin"]["last_origin"] + 1,
        eval_cfg["rolling_origin"]["step"],
    ))
    horizons = eval_cfg["horizons"]

    if args.quick:
        origins = origins[-3:]
        horizons = [1, 5, 10]

    factories = get_all_model_factories(quick=args.quick)
    if args.models:
        factories = {k: v for k, v in factories.items() if k in args.models}

    all_results = []

    for code in country_codes:
        print(f"\n{'='*60}")
        print(f"Country: {code}")
        print(f"{'='*60}")

        try:
            data = load_country(code)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        for model_name, factory in factories.items():
            print(f"  {model_name}...", end=" ", flush=True)
            try:
                results = rolling_origin_eval(
                    factory,
                    data["log_mx"], data["ages"], data["years"],
                    data.get("exposures"), data.get("deaths"),
                    origins=origins, horizons=horizons,
                    country=code, sex="Total",
                )
                all_results.extend(results)
                print(f"{len(results)} results")

                shock_results = mortality_shock_eval(
                    factory,
                    data["log_mx"], data["ages"], data["years"],
                    country=code, sex="Total",
                    exposures=data.get("exposures"), deaths=data.get("deaths"),
                )
                all_results.extend(shock_results)

            except Exception as e:
                print(f"ERROR: {e}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=asdict(all_results[0]).keys())
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))

    print(f"\n{len(all_results)} results saved to {output_path}")


if __name__ == "__main__":
    main()
