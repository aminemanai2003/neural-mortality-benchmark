import numpy as np
import pytest

from mortality.data.loader import load_country
from mortality.models.classical import CLASSICAL_MODELS

DATA_AVAILABLE = True
try:
    load_country("FRATNP")
except FileNotFoundError:
    DATA_AVAILABLE = False


@pytest.fixture
def france_data():
    return load_country("FRATNP")


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HMD data not downloaded")
class TestLeeCarter:
    def test_fit_and_forecast(self, france_data):
        model = CLASSICAL_MODELS["lee_carter"]()
        d = france_data
        model.fit(d["log_mx"], d["ages"], d["years"], d["exposures"], d["deaths"])

        assert model.ax.shape == (101,)
        assert model.bx.shape == (101,)
        assert model.kt.shape == (74,)
        assert abs(model.bx.sum() - 1.0) < 1e-6

        fc = model.forecast(10)
        assert fc.shape == (101, 10)
        assert not np.any(np.isnan(fc))

    def test_simulate_shape(self, france_data):
        model = CLASSICAL_MODELS["lee_carter"]()
        d = france_data
        model.fit(d["log_mx"], d["ages"], d["years"])
        sim = model.simulate(10, n_paths=50)
        assert sim.shape == (50, 101, 10)

    def test_kt_decreasing_trend(self, france_data):
        """kt should generally decrease (mortality improvement)."""
        model = CLASSICAL_MODELS["lee_carter"]()
        d = france_data
        model.fit(d["log_mx"], d["ages"], d["years"])
        assert model.drift < 0, "kt drift should be negative (mortality improving)"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HMD data not downloaded")
class TestAllClassicalModels:
    @pytest.mark.parametrize("name", [
        "lee_carter", "lee_miller", "bms", "poisson_lc",
        "hyndman_ullah", "random_walk", "frozen_rates",
    ])
    def test_fit_forecast_shape(self, name, france_data):
        model = CLASSICAL_MODELS[name]()
        d = france_data
        model.fit(d["log_mx"], d["ages"], d["years"], d["exposures"], d["deaths"])
        fc = model.forecast(5)
        assert fc.shape[1] == 5
        assert not np.any(np.isnan(fc))

    def test_cbd_elderly_ages(self, france_data):
        model = CLASSICAL_MODELS["cbd"]()
        d = france_data
        model.fit(d["log_mx"], d["ages"], d["years"])
        fc = model.forecast(5)
        assert fc.shape == (41, 5)  # ages 60..100
        assert not np.any(np.isnan(fc))
