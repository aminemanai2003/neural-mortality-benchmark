import numpy as np

from mortality.actuarial.case_study import (
    longevity_risk_analysis,
    price_annuity_portfolio,
    solvency_ii_shock,
)


def test_solvency_ii_shock():
    mx = np.full(101, 0.01)
    shocked = solvency_ii_shock(mx, 0.80)
    assert np.allclose(shocked, 0.008)


def test_price_annuity_portfolio():
    mx = np.full(101, 0.01)
    result = price_annuity_portfolio({"model_a": mx, "model_b": mx * 0.9})
    assert len(result) == 2
    assert "total_provision_eur" in result.columns
    assert all(result["total_provision_eur"] > 0)


def test_longevity_risk_analysis():
    mx = np.full(101, 0.01)
    result = longevity_risk_analysis({"model_a": mx})
    assert len(result) == 1
    assert result["scr_longevity_eur"].iloc[0] > 0
