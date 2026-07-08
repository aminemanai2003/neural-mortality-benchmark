import numpy as np
import pytest

from mortality.actuarial.life_table import life_expectancy_at
from mortality.data.loader import load_country

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

    def test_missing_trailing_years_trimmed(self):
        """A country without data for the requested end year (e.g. GBRTENW in
        2023) must have that year dropped, not floored to m_x = 1e-6."""
        data = load_country("GBRTENW")
        # No column may consist entirely of the 1e-6 floor.
        floored_cols = np.all(data["mx"] <= 1e-6, axis=0)
        assert not floored_cols.any()
        assert data["years"][-1] < 2023
        assert data["log_mx"].shape[1] == len(data["years"])
        assert data["exposures"].shape[1] == len(data["years"])
