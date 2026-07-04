"""Actuarial case study: annuity pricing and Solvency II longevity shock.

Prices a whole-life annuity-due for a portfolio of 1,000 French pensioners
aged 65, comparing provisions across all models. Then applies a Solvency II
type longevity shock (-20% to mortality rates) to quantify model risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mortality.actuarial.life_table import annuity_due


def price_annuity_portfolio(
    mx_by_model: dict[str, np.ndarray],
    annual_payment: float = 12_000.0,
    n_pensioners: int = 1_000,
    start_age: int = 65,
    interest_rate: float = 0.02,
) -> pd.DataFrame:
    """Price a portfolio of annuities under each model's mortality projection.

    Args:
        mx_by_model: {model_name: mx_vector} for a single forecast year
        annual_payment: annual payment per pensioner (EUR)
        n_pensioners: number of pensioners
        start_age: annuity start age
        interest_rate: discount rate

    Returns:
        DataFrame with columns: model, annuity_factor, provision_per_person, total_provision
    """
    rows = []
    for model_name, mx in mx_by_model.items():
        a = annuity_due(mx, start_age, interest_rate)
        provision_pp = a * annual_payment
        total = provision_pp * n_pensioners
        rows.append({
            "model": model_name,
            "annuity_factor": round(a, 4),
            "provision_per_person": round(provision_pp, 0),
            "total_provision_eur": round(total, 0),
        })

    df = pd.DataFrame(rows).sort_values("total_provision_eur", ascending=False)
    return df.reset_index(drop=True)


def solvency_ii_shock(
    mx: np.ndarray,
    shock_factor: float = 0.80,
) -> np.ndarray:
    """Apply a Solvency II type longevity shock.

    Standard approach: reduce mortality rates by 20% (shock_factor=0.80),
    meaning people live longer -> higher provisions needed.
    """
    return mx * shock_factor


def longevity_risk_analysis(
    mx_by_model: dict[str, np.ndarray],
    annual_payment: float = 12_000.0,
    n_pensioners: int = 1_000,
    start_age: int = 65,
    interest_rate: float = 0.02,
    shock_factor: float = 0.80,
) -> pd.DataFrame:
    """Compare base vs shocked provisions across models."""
    rows = []
    for model_name, mx in mx_by_model.items():
        a_base = annuity_due(mx, start_age, interest_rate)
        mx_shocked = solvency_ii_shock(mx, shock_factor)
        a_shocked = annuity_due(mx_shocked, start_age, interest_rate)

        base_total = a_base * annual_payment * n_pensioners
        shocked_total = a_shocked * annual_payment * n_pensioners
        scr_longevity = shocked_total - base_total

        rows.append({
            "model": model_name,
            "base_provision_eur": round(base_total, 0),
            "shocked_provision_eur": round(shocked_total, 0),
            "scr_longevity_eur": round(scr_longevity, 0),
            "scr_pct": round(100 * scr_longevity / base_total, 2) if base_total > 0 else 0,
        })

    df = pd.DataFrame(rows).sort_values("scr_longevity_eur", ascending=False)
    return df.reset_index(drop=True)
