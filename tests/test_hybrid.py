import numpy as np
import pytest

from mortality.data.loader import load_country
from mortality.models.hybrid import LCResNet

DATA_AVAILABLE = True
try:
    load_country("FRATNP")
except FileNotFoundError:
    DATA_AVAILABLE = False


@pytest.fixture
def france_data():
    return load_country("FRATNP")


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HMD data not downloaded")
class TestLCResNet:
    def test_fit_and_forecast(self, france_data):
        d = france_data
        model = LCResNet(epochs=10, patience=5, seed=42)
        model.fit(d["log_mx"], d["ages"], d["years"], d["exposures"], d["deaths"])
        fc = model.forecast(10)
        assert fc.shape == (101, 10)
        assert not np.any(np.isnan(fc))

    def test_simulate_shape(self, france_data):
        d = france_data
        model = LCResNet(epochs=10, patience=5, seed=42)
        model.fit(d["log_mx"], d["ages"], d["years"])
        sim = model.simulate(5, n_paths=10)
        assert sim.shape == (10, 101, 5)

    def test_lc_skeleton_accessible(self, france_data):
        d = france_data
        model = LCResNet(epochs=5, patience=3, seed=42)
        model.fit(d["log_mx"], d["ages"], d["years"])
        assert model.lc.ax is not None
        assert model.lc.bx is not None
        assert model.lc.kt is not None
        assert model.lc.drift < 0

    def test_shrinkage_reduces_correction(self, france_data):
        d = france_data
        model = LCResNet(epochs=10, patience=5, seed=42, shrinkage_lambda=0.5)
        model.fit(d["log_mx"], d["ages"], d["years"])
        fc = model.forecast(20)
        lc_fc = model.lc.forecast(20)
        correction_h1 = np.abs(fc[:, 0] - lc_fc[:, 0]).mean()
        correction_h20 = np.abs(fc[:, -1] - lc_fc[:, -1]).mean()
        assert correction_h20 < correction_h1, "Shrinkage should reduce correction at long horizons"
