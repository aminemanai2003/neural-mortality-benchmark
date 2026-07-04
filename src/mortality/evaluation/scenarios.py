"""Scenario-based evaluation for the decision framework."""
from __future__ import annotations

import numpy as np

from mortality.evaluation.rolling_origin import EvalResult, rolling_origin_eval


def short_history_eval(
    model_factory: callable,
    log_mx: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    train_lengths: list[int],
    horizons: list[int] | None = None,
    country: str = "",
    sex: str = "Total",
    exposures: np.ndarray | None = None,
    deaths: np.ndarray | None = None,
) -> list[EvalResult]:
    """Evaluate model with truncated training history.

    The origin is placed so that max(horizons) evaluation years remain in the
    FULL series; only the training window is truncated to `length` years.
    """
    horizons = horizons or [5, 10]
    results = []
    n_years = log_mx.shape[1]
    max_h = max(horizons)

    origin_idx = n_years - 1 - max_h
    if origin_idx < 1:
        return results
    origin_year = int(years[origin_idx])

    for length in train_lengths:
        start = origin_idx - length + 1
        if start < 0:
            continue

        # Window starts `length` years before the origin but keeps the full
        # tail so the evaluator can find the truth for every horizon.
        log_mx_sub = log_mx[:, start:]
        years_sub = years[start:]
        exp_sub = exposures[:, start:] if exposures is not None else None
        dth_sub = deaths[:, start:] if deaths is not None else None

        sub_results = rolling_origin_eval(
            model_factory, log_mx_sub, ages, years_sub,
            exp_sub, dth_sub,
            origins=[origin_year],
            horizons=horizons,
            country=country, sex=sex,
        )
        for r in sub_results:
            r.metric = f"{r.metric}_hist{length}"
        results.extend(sub_results)
    return results


def mortality_shock_eval(
    model_factory: callable,
    log_mx: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    train_end: int = 2019,
    eval_years: list[int] | None = None,
    country: str = "",
    sex: str = "Total",
    exposures: np.ndarray | None = None,
    deaths: np.ndarray | None = None,
) -> list[EvalResult]:
    """Train up to train_end, evaluate on shock years (COVID)."""
    if eval_years is None:
        eval_years = [2020, 2021, 2022, 2023]

    year_list = years.tolist()
    if train_end not in year_list:
        return []

    train_idx = year_list.index(train_end)
    max_h = log_mx.shape[1] - train_idx - 1
    if max_h <= 0:
        return []

    return rolling_origin_eval(
        model_factory, log_mx, ages, years,
        exposures, deaths,
        origins=[train_end],
        horizons=[min(h, max_h) for h in range(1, max_h + 1)],
        country=country, sex=sex,
    )


def age_group_eval(
    model_factory: callable,
    log_mx: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    age_groups: dict[str, tuple[int, int]] | None = None,
    origins: list[int] | None = None,
    horizons: list[int] | None = None,
    country: str = "",
    sex: str = "Total",
    exposures: np.ndarray | None = None,
    deaths: np.ndarray | None = None,
) -> list[EvalResult]:
    """Evaluate per age group."""
    if age_groups is None:
        age_groups = {"young": (0, 19), "working": (20, 64), "elderly": (65, 100)}

    results = []
    for group_name, (a_lo, a_hi) in age_groups.items():
        mask = (ages >= a_lo) & (ages <= a_hi)
        log_mx_sub = log_mx[mask, :]
        ages_sub = ages[mask]
        exp_sub = exposures[mask, :] if exposures is not None else None
        dth_sub = deaths[mask, :] if deaths is not None else None

        sub_results = rolling_origin_eval(
            model_factory, log_mx_sub, ages_sub, years,
            exp_sub, dth_sub,
            origins=origins, horizons=horizons,
            country=country, sex=sex,
        )
        for r in sub_results:
            r.metric = f"{r.metric}_{group_name}"
        results.extend(sub_results)
    return results
