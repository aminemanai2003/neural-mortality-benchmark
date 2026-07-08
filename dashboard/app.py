"""Streamlit dashboard for the mortality benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import os
os.chdir(PROJECT_ROOT)

from mortality.actuarial.case_study import longevity_risk_analysis, price_annuity_portfolio
from mortality.actuarial.life_table import life_expectancy_at
from mortality.data.loader import load_country
from mortality.evaluation.decision_framework import recommend_model
from mortality.models.classical import CLASSICAL_MODELS
from mortality.models.hybrid import HYBRID_MODELS

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
            ax2.set_ylabel("e0")
            ax2.set_title(f"Life expectancy at birth — {COUNTRIES[country_code]}")
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
        models_to_compare = ["lee_carter", "poisson_lc", "random_walk", "lc_resnet"]
        mx_by_model = {}

        for name in models_to_compare:
            if name in ALL_MODELS:
                model = ALL_MODELS[name]()
                model.fit(data["log_mx"], data["ages"], data["years"],
                          data.get("exposures"), data.get("deaths"))
                fc = model.forecast(10)
                mx_by_model[name] = np.exp(fc[:, -1])

        if mx_by_model:
            pricing = price_annuity_portfolio(mx_by_model)
            st.dataframe(pricing, use_container_width=True)

            st.subheader("Solvency II Longevity Shock")
            risk = longevity_risk_analysis(mx_by_model)
            st.dataframe(risk, use_container_width=True)

            spread = pricing["total_provision_eur"].max() - pricing["total_provision_eur"].min()
            st.metric("Model risk spread", f"{spread:,.0f} EUR")
    except FileNotFoundError:
        st.error("Download HMD data first.")

# ── Tab 4: Which Model? ──

with tab4:
    st.subheader("Model Recommendation Engine")
    col1, col2, col3 = st.columns(3)
    with col1:
        hist_len = st.slider("History length (years)", 15, 70, 50)
    with col2:
        fc_horizon = st.slider("Forecast horizon (years)", 1, 30, 10)
    with col3:
        age_focus = st.selectbox("Age focus", ["all", "young", "working", "elderly"])

    interp = st.checkbox("Needs regulatory interpretability?")

    recommended = recommend_model(hist_len, fc_horizon, age_focus, interp)
    st.success(f"**Recommended model: {recommended}**")

    st.markdown("""
    **Decision logic:**
    - Short history (<25 years) → Random walk (neural networks need more data)
    - Elderly ages → CBD (designed for 60+, used by regulators)
    - Short horizon (≤5 years) → LC-ResNet (neural correction helps most)
    - Medium horizon (≤10 years) → Poisson Lee-Carter
    - Long horizon (>10 years) → Lee-Carter (stable extrapolation)
    - If interpretability required → LC-ResNet at short horizons, LC otherwise
    """)
