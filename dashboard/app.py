"""Streamlit dashboard for the mortality benchmark."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from mortality.actuarial.case_study import (  # noqa: E402
    longevity_risk_analysis,
    price_annuity_portfolio,
)
from mortality.actuarial.life_table import life_expectancy_at  # noqa: E402
from mortality.data.loader import load_country  # noqa: E402
from mortality.evaluation.decision_framework import recommend_model  # noqa: E402
from mortality.models.classical import CLASSICAL_MODELS  # noqa: E402
from mortality.models.hybrid import HYBRID_MODELS  # noqa: E402

st.set_page_config(
    page_title="Neural Mortality Benchmark",
    page_icon="📊",
    layout="wide",
)

st.title("Neural Mortality Benchmark")
st.markdown(
    "**When should actuaries trust neural networks for mortality forecasting?**"
)

COUNTRIES = {
    "FRATNP": "France",
    "GBRTENW": "England & Wales",
    "USA": "United States",
    "JPN": "Japan",
    "ITA": "Italy",
    "ESP": "Spain",
    "SWE": "Sweden",
    "NLD": "Netherlands",
}

ALL_MODELS = {**CLASSICAL_MODELS}
ALL_MODELS["lc_resnet"] = HYBRID_MODELS["lc_resnet"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Mortality Surface", "Model Comparison", "Actuarial Case Study", "Which Model?"
])

# ── Tab 1: Mortality Surface Heatmap ──

with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        country_code = st.selectbox("Country", list(COUNTRIES.keys()),
                                    format_func=lambda x: COUNTRIES[x])
        sex = st.selectbox("Sex", ["Total", "Male", "Female"])

    with col2:
        try:
            data = load_country(country_code, sex=sex)
            fig, ax = plt.subplots(figsize=(14, 7))
            log_mx = data["log_mx"]
            vmin, vmax = np.nanpercentile(log_mx, [1, 99])
            im = ax.imshow(
                log_mx, aspect="auto", origin="lower", cmap="RdYlBu_r",
                vmin=vmin, vmax=vmax,
                extent=[data["years"][0], data["years"][-1],
                        data["ages"][0], data["ages"][-1]],
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Age")
            ax.set_title(f"Log mortality surface — {COUNTRIES[country_code]} ({sex})")
            fig.colorbar(im, ax=ax, shrink=0.8, label="log m(x,t)")
            st.pyplot(fig)
            plt.close()

            e0_series = [
                life_expectancy_at(data["mx"][:, t]) for t in range(data["mx"].shape[1])
            ]
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(data["years"], e0_series, linewidth=2)
            ax2.set_xlabel("Year")
            ax2.set_ylabel("e0 (truncated at age 100)")
            ax2.set_title(
                f"Age-100-truncated life expectancy — {COUNTRIES[country_code]}"
            )
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()
        except FileNotFoundError:
            st.error(f"Data not found for {country_code}. Run `python scripts/download_hmd.py`.")

# ── Tab 2: Model Comparison ──

with tab2:
    results_path = Path("results/benchmark.csv")
    if results_path.exists():
        df = pd.read_csv(results_path)
        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox("Metric", df["metric"].unique())
        with col2:
            horizon = st.selectbox("Horizon", sorted(df["horizon"].unique()))

        filtered = df[(df["metric"] == metric) & (df["horizon"] == horizon)]
        if len(filtered) > 0:
            summary = filtered.groupby("model_name")["value"].agg(["mean", "std"]).round(4)
            summary = summary.sort_values("mean")
            st.dataframe(summary, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            summary["mean"].plot.barh(ax=ax, xerr=summary["std"], capsize=3)
            ax.set_xlabel(metric)
            ax.set_title(f"{metric} at horizon {horizon}")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Run `python scripts/run_benchmark.py` first to generate results.")

# ── Tab 3: Actuarial Case Study ──

with tab3:
    st.subheader("Annuity Pricing: Model Risk in EUR")
    st.markdown(
        "Portfolio of **1,000 French pensioners aged 65**, annual payment **12,000 EUR**."
    )

    try:
        data = load_country("FRATNP")
        models_to_compare = ["lee_carter", "poisson_lc", "random_walk", "lc_resnet", "cbd"]
        mx_by_model = {}

        for name in models_to_compare:
            if name in ALL_MODELS:
                model = ALL_MODELS[name]()
                model.fit(data["log_mx"], data["ages"], data["years"],
                          data.get("exposures"), data.get("deaths"))
                fc = model.forecast(10)
                if name == "cbd":
                    # CBD forecasts ages 60--100. Younger rates are immaterial to an
                    # annuity starting at 65, but a full 0--100 vector is required.
                    mx = data["mx"][:, -1].copy()
                    positions = np.searchsorted(data["ages"], model.forecast_ages)
                    mx[positions] = np.exp(fc[:, -1])
                    mx_by_model[name] = mx
                else:
                    mx_by_model[name] = np.exp(fc[:, -1])

        if mx_by_model:
            pricing = price_annuity_portfolio(mx_by_model)
            st.dataframe(pricing, use_container_width=True)

            st.subheader("Illustrative Longevity Stress")
            st.caption(
                "Liability impact of a 20% mortality-rate reduction; "
                "not a complete insurer-level SCR."
            )
            risk = longevity_risk_analysis(mx_by_model)
            st.dataframe(risk, use_container_width=True)

            spread = pricing["total_provision_eur"].max() - pricing["total_provision_eur"].min()
            st.metric("Model risk spread", f"{spread:,.0f} EUR")
    except FileNotFoundError:
        st.error("Download HMD data first.")

# ── Tab 4: Which Model? ──

with tab4:
    st.subheader("Model Recommendation Engine")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hist_len = st.slider("History length (years)", 15, 70, 50)
    with col2:
        fc_horizon = st.slider("Forecast horizon (years)", 1, 30, 10)
    with col3:
        age_focus = st.selectbox("Age focus", ["all", "young", "working", "elderly"])
    with col4:
        objective = st.selectbox(
            "Primary objective", ["log_rates", "life_expectancy", "annuity"]
        )

    interp = st.checkbox("Restrict to models with a transparent statistical skeleton?")

    recommended = recommend_model(
        hist_len, fc_horizon, age_focus, interp, objective=objective
    )
    st.success(f"**Recommended model: {recommended}**")

    st.markdown("""
    **Decision logic:** recommendations reproduce the empirical winners in this
    benchmark and depend on the selected loss. The random walk leads full-grid
    log-rate accuracy; H-U leads 20-year life-expectancy accuracy; the FFNN leads
    20-year annuity-factor accuracy; and LC-ResNet leads structured models at one
    year. A transparent-skeleton restriction excludes simple baselines and fully
    neural models. These are benchmark-specific recommendations, not universal rules.
    """)
