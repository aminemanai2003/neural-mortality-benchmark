import numpy as np
import pytest

from mortality.data.loader import load_country, parse_hmd_file
from mortality.actuarial.life_table import life_expectancy_at

DATA_AVAILABLE = True
try:
    load_country("FRATNP")
except FileNotFoundError:
    DATA_AVAILABLE = False


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HMD data not downloaded")
class TestLoader:
    def test_france_shape(self):
        data = load_country("FRATNP")
        assert data["log_mx"].shape == (101, 74)
        assert data["ages"][0] == 0
        assert data["ages"][-1] == 100

    def test_france_e0_2019(self):
        data = load_country("FRATNP")
        idx = data["years"].tolist().index(2019)
        e0 = life_expectancy_at(data["mx"][:, idx])
        assert 81 < e0 < 84, f"e0 France 2019 = {e0:.1f}, expected ~82.8"

    def test_no_nan_in_log_mx(self):
        data = load_country("FRATNP")
        assert not np.any(np.isnan(data["log_mx"]))

    def test_exposures_loaded(self):
        data = load_country("FRATNP")
        assert "exposures" in data
        assert data["exposures"].shape == (101, 74)
        assert np.all(data["exposures"] > 0)
