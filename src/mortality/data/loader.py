from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config/data.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_hmd_file(filepath: Path) -> pd.DataFrame:
    """Parse an HMD fixed-width file (Mx_1x1.txt or Exposures_1x1.txt)."""
    rows = []
    with open(filepath) as f:
        lines = f.readlines()

    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not header_found:
            if stripped.startswith("Year") or "Age" in stripped:
                header_found = True
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        year = int(parts[0])
        age_str = parts[1].replace("+", "").replace("-", "")
        try:
            age = int(age_str)
        except ValueError:
            continue
        female = _parse_float(parts[2])
        male = _parse_float(parts[3])
        total = _parse_float(parts[4])
        rows.append({"Year": year, "Age": age, "Female": female, "Male": male, "Total": total})

    return pd.DataFrame(rows)


def _parse_float(s: str) -> float:
    s = s.strip()
    if s == "." or s == "":
        return np.nan
    return float(s)


def load_country(
    country_code: str,
    raw_dir: str = "data/raw",
    ages: tuple[int, int] = (0, 100),
    years: tuple[int, int] = (1950, 2023),
    sex: str = "Total",
) -> dict[str, np.ndarray]:
    """Load and clean HMD data for one country.

    Returns dict with keys: log_mx, mx, exposures, ages, years
    Each matrix is (n_ages x n_years).
    """
    country_path = Path(raw_dir) / country_code
    mx_file = country_path / "Mx_1x1.txt"
    exp_file = country_path / "Exposures_1x1.txt"

    if not mx_file.exists():
        raise FileNotFoundError(
            f"HMD file not found: {mx_file}. "
            f"Download from mortality.org into {country_path}/"
        )

    mx_df = parse_hmd_file(mx_file)
    mx_df = mx_df[(mx_df["Year"] >= years[0]) & (mx_df["Year"] <= years[1])]
    mx_df = mx_df[(mx_df["Age"] >= ages[0]) & (mx_df["Age"] <= ages[1])]

    age_arr = np.arange(ages[0], ages[1] + 1)
    year_arr = np.arange(years[0], years[1] + 1)

    mx_pivot = mx_df.pivot(index="Age", columns="Year", values=sex)
    mx_pivot = mx_pivot.reindex(index=age_arr, columns=year_arr)
    mx_matrix = mx_pivot.values.astype(float)

    # Some countries stop before the requested end year (e.g. no 2023 data yet).
    # Flooring a fully-missing year would fabricate absurd rates, so trim
    # trailing all-NaN columns instead of imputing them.
    valid_cols = ~np.all(np.isnan(mx_matrix), axis=0)
    last_valid = int(np.max(np.nonzero(valid_cols))) if valid_cols.any() else -1
    if last_valid < mx_matrix.shape[1] - 1:
        mx_matrix = mx_matrix[:, : last_valid + 1]
        year_arr = year_arr[: last_valid + 1]

    floor = 1e-6
    mx_matrix = np.where(np.isnan(mx_matrix) | (mx_matrix <= 0), floor, mx_matrix)
    log_mx = np.log(mx_matrix)

    result = {"log_mx": log_mx, "mx": mx_matrix, "ages": age_arr, "years": year_arr}

    if exp_file.exists():
        exp_df = parse_hmd_file(exp_file)
        exp_df = exp_df[(exp_df["Year"] >= years[0]) & (exp_df["Year"] <= years[1])]
        exp_df = exp_df[(exp_df["Age"] >= ages[0]) & (exp_df["Age"] <= ages[1])]
        exp_pivot = exp_df.pivot(index="Age", columns="Year", values=sex)
        exp_pivot = exp_pivot.reindex(index=age_arr, columns=year_arr)
        exposures = exp_pivot.values.astype(float)
        exposures = np.where(np.isnan(exposures) | (exposures <= 0), 1.0, exposures)
        result["exposures"] = exposures
        deaths = mx_matrix * exposures
        result["deaths"] = deaths

    return result


def load_all_countries(
    config_path: str = "config/data.yaml",
    sex: str = "Total",
) -> dict[str, dict[str, np.ndarray]]:
    """Load all countries defined in the config."""
    cfg = load_config(config_path)
    data = {}
    for country in cfg["countries"]:
        code = country["code"]
        try:
            data[code] = load_country(
                code,
                raw_dir=cfg["paths"]["raw"],
                ages=(cfg["ages"]["start"], cfg["ages"]["end"]),
                years=(cfg["years"]["start"], cfg["years"]["end"]),
                sex=sex,
            )
        except FileNotFoundError as e:
            print(f"Skipping {code}: {e}")
    return data
