"""Rolling-origin evaluation framework."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mortality.evaluation.metrics import (
    ACTUARIAL_METRIC_REGISTRY,
    METRIC_REGISTRY,
)


@dataclass
class EvalResult:
    model_name: str
    country: str
    sex: str
    origin: int
    horizon: int
    metric: str
    value: float
    train_time: float


def rolling_origin_eval(
    model_factory: callable,
    log_mx: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    exposures: np.ndarray | None = None,
    deaths: np.ndarray | None = None,
    origins: list[int] | None = None,
    horizons: list[int] | None = None,
    country: str = "",
    sex: str = "Total",
) -> list[EvalResult]:
    """Run rolling-origin evaluation on a single model + country."""
    if origins is None:
        origins = list(range(1990, 2019, 2))
    if horizons is None:
        horizons = [1, 5, 10, 20]

    results = []
    year_list = years.tolist()

    for origin in origins:
        if origin not in year_list:
            continue
        origin_idx = year_list.index(origin)

        log_mx_train = log_mx[:, : origin_idx + 1]
        years_train = years[: origin_idx + 1]
        exp_train = exposures[:, : origin_idx + 1] if exposures is not None else None
        dth_train = deaths[:, : origin_idx + 1] if deaths is not None else None

        model = model_factory()
        t0 = time.time()
        model.fit(log_mx_train, ages, years_train, exp_train, dth_train)
        train_time = time.time() - t0

        # Ages the model actually forecasts (CBD restricts to 60+).
        model_ages = getattr(model, "forecast_ages", None)
        if model_ages is None:
            model_ages = ages
        # Position of the model's ages within the full evaluation age grid.
        age_pos = np.searchsorted(ages, model_ages)
        # Actuarial metrics (e0 from birth, annuity ä65) are only meaningful when
        # the forecast covers the whole lifespan grid — from birth up to the old
        # ages — otherwise e0 is truncated and ä65 is zero or nonsense.
        full_grid = (
            len(model_ages) == len(ages)
            and int(ages[0]) == 0
            and int(ages[-1]) >= 90
        )

        for h in horizons:
            end_idx = origin_idx + h
            if end_idx >= log_mx.shape[1]:
                continue

            log_mx_true = log_mx[age_pos, origin_idx + 1 : end_idx + 1]
            fc = model.forecast(h)

            n_fc = min(fc.shape[1], log_mx_true.shape[1])
            if n_fc == 0:
                continue
            fc = fc[:, :n_fc]
            log_mx_true = log_mx_true[:, :n_fc]

            for metric_name, metric_fn in METRIC_REGISTRY.items():
                val = metric_fn(log_mx_true, fc)
                results.append(EvalResult(
                    model.name, country, sex, origin, h, metric_name, val, train_time
                ))

            if full_grid:
                for metric_name, metric_fn in ACTUARIAL_METRIC_REGISTRY.items():
                    mx_true = np.exp(log_mx_true)
                    mx_pred = np.exp(fc)
                    val = metric_fn(mx_true, mx_pred)
                    results.append(EvalResult(
                        model.name, country, sex, origin, h, metric_name, val, train_time
                    ))

    return results
