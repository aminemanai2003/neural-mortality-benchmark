"""Decision framework: 'which model to use when' scorecard."""
from __future__ import annotations

import pandas as pd

CRITERIA = [
    "short_history",
    "long_horizon",
    "elderly_ages",
    "mortality_shock",
    "interpretability",
    "training_speed",
]

INTERPRETABILITY_SCORES = {
    "lee_carter": 5,
    "lee_miller": 5,
    "poisson_lc": 5,
    "cbd": 5,
    "hyndman_ullah": 4,
    "random_walk": 5,
    "frozen_rates": 5,
    "lstm_kt": 3,
    "gru_kt": 3,
    "bilstm_kt": 3,
    "transformer_kt": 2,
    "ffnn_embeddings": 2,
    "cnn_surface": 2,
    "lc_resnet": 4,
}


def build_scorecard(
    benchmark_df: pd.DataFrame,
    training_times: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a scorecard ranking models on each criterion.

    Uses benchmark results to assign 1-5 scores per criterion.
    """
    models = benchmark_df["model_name"].unique()
    scores = {m: {} for m in models}

    for model in models:
        model_df = benchmark_df[benchmark_df["model_name"] == model]

        # Short history: how well does the model perform with limited data
        short_metrics = model_df[
            model_df["metric"].isin(["rmse_log_mx_hist20", "rmse_log_mx_hist30"])
        ]
        if len(short_metrics) > 0:
            peer_means = [
                benchmark_df[
                    (benchmark_df["model_name"] == m)
                    & benchmark_df["metric"].isin(
                        ["rmse_log_mx_hist20", "rmse_log_mx_hist30"]
                    )
                ]["value"].mean()
                for m in models
            ]
            scores[model]["short_history"] = _relative_score(
                short_metrics["value"].mean(), peer_means
            )
        else:
            h5 = model_df[(model_df["horizon"] <= 5) & (model_df["metric"] == "rmse_log_mx")]
            scores[model]["short_history"] = _relative_score(
                h5["value"].mean() if len(h5) > 0 else 999,
                [_get_mean(benchmark_df, m, "rmse_log_mx", max_h=5) for m in models],
            )

        # Long horizon performance
        long = model_df[
            (model_df["horizon"] >= 10) & (model_df["metric"] == "rmse_log_mx")
        ]
        scores[model]["long_horizon"] = _relative_score(
            long["value"].mean() if len(long) > 0 else 999,
            [_get_mean(benchmark_df, m, "rmse_log_mx", min_h=10) for m in models],
        )

        # Elderly ages performance
        elderly = model_df[model_df["metric"].str.contains("elderly", na=False)]
        if len(elderly) > 0:
            scores[model]["elderly_ages"] = _relative_score(
                elderly["value"].mean(),
                [
                    benchmark_df[
                        (benchmark_df["model_name"] == m)
                        & (benchmark_df["metric"].str.contains("elderly", na=False))
                    ]["value"].mean()
                    for m in models
                ],
            )
        else:
            scores[model]["elderly_ages"] = 3

        # COVID shock robustness: origin 2019 is produced only by the shock scenario.
        shock_metric = model_df[
            (model_df["origin"] == 2019) & (model_df["metric"] == "rmse_log_mx")
        ]
        scores[model]["mortality_shock"] = _relative_score(
            shock_metric["value"].mean() if len(shock_metric) > 0 else 999,
            [
                benchmark_df[
                    (benchmark_df["model_name"] == m)
                    & (benchmark_df["origin"] == 2019)
                    & (benchmark_df["metric"] == "rmse_log_mx")
                ]["value"].mean()
                for m in models
            ],
        )

        # Interpretability (fixed scores)
        scores[model]["interpretability"] = INTERPRETABILITY_SCORES.get(model, 3)

        # Training speed
        if training_times and model in training_times:
            all_times = list(training_times.values())
            scores[model]["training_speed"] = _relative_score(
                training_times[model], all_times, lower_is_better=True
            )
        else:
            avg_time = model_df["train_time"].mean() if "train_time" in model_df.columns else 1
            scores[model]["training_speed"] = _relative_score(
                avg_time,
                [
                    benchmark_df[benchmark_df["model_name"] == m]["train_time"].mean()
                    for m in models
                ],
                lower_is_better=True,
            )

    rows = []
    for model in models:
        row = {"model": model}
        total = 0
        for c in CRITERIA:
            val = scores[model].get(c, 3)
            row[c] = val
            total += val
        row["total"] = total
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("total", ascending=False)
    return df.reset_index(drop=True)


def recommend_model(
    history_length: int,
    horizon: int,
    age_focus: str = "all",
    needs_interpretability: bool = False,
    compute_budget: str = "any",
    objective: str = "log_rates",
    structured_only: bool = False,
) -> str:
    """Return the empirical winner for a stated use case.

    Recommendations summarize this benchmark; they are not universal rankings.
    ``structured_only`` excludes the two simple baselines and fully neural models.
    ``compute_budget`` is retained for API compatibility.
    """
    del compute_budget
    structured_only = structured_only or needs_interpretability

    if structured_only:
        if history_length < 25 or horizon <= 5:
            return "lc_resnet"
        if age_focus == "elderly" or objective == "life_expectancy":
            return "hyndman_ullah"
        if objective == "annuity":
            return "lee_carter"
        return "lee_miller"

    if history_length < 25 or age_focus == "elderly":
        return "frozen_rates"
    if objective == "life_expectancy" and horizon > 10:
        return "hyndman_ullah"
    if objective == "annuity":
        return "random_walk" if horizon <= 10 else "ffnn_embeddings"
    return "random_walk"


def _get_mean(df: pd.DataFrame, model: str, metric: str,
              min_h: int = 0, max_h: int = 999) -> float:
    sub = df[
        (df["model_name"] == model)
        & (df["metric"] == metric)
        & (df["horizon"] >= min_h)
        & (df["horizon"] <= max_h)
    ]
    return sub["value"].mean() if len(sub) > 0 else 999


def _relative_score(value: float, all_values: list[float],
                    lower_is_better: bool = True) -> int:
    valid = [v for v in all_values if pd.notna(v) and v < 900]
    if not valid or pd.isna(value) or value >= 900:
        return 3
    sorted_vals = sorted(valid, reverse=not lower_is_better)
    n = len(sorted_vals)
    try:
        ordered = sorted(valid) if lower_is_better else sorted(valid, reverse=True)
        rank = ordered.index(value)
    except ValueError:
        return 3
    return max(1, min(5, 5 - int(4 * rank / max(1, n - 1))))
