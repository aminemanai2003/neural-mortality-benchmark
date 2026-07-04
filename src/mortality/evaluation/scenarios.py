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
    """Evaluate model with truncated history."""
    results = []
    for length in train_lengths:
        if length >= log_mx.shape[1]:
            continue
        start = log_mx.shape[1] - length - 20
        if start < 0:
            start = 0
        end = start + length
        log_mx_sub = log_mx[:, start:end]
        years_sub = years[start:end]
        exp_sub = exposures[:, start:end] if exposures is not None else None
        dth_sub = deaths[:, start:end] if deaths is not None else None

        sub_results = rolling_origin_eval(
            model_factory, log_mx_sub, ages, years_sub,
            exp_sub, dth_sub,
            origins=[int(years_sub[-5])],
            horizons=horizons or [5, 10],
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
