import numpy as np
import pytest

from mortality.data.loader import load_country
from mortality.evaluation.diebold_mariano import diebold_mariano_test
from mortality.evaluation.rolling_origin import rolling_origin_eval
from mortality.evaluation.scenarios import age_group_eval, short_history_eval
from mortality.models.classical import CairnsBlakeDowd, LeeCarter

DATA_AVAILABLE = True
try:
    load_country("FRATNP")
except FileNotFoundError:
    DATA_AVAILABLE = False


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HMD data not downloaded")
class TestRollingOrigin:
    def test_cbd_subrange_no_crash(self):
        """CBD forecasts only 60+ ages; evaluation must align, not crash."""
        d = load_country("FRATNP")
        results = rolling_origin_eval(
            lambda: CairnsBlakeDowd(),
            d["log_mx"], d["ages"], d["years"], d["exposures"], d["deaths"],
            origins=[2010], horizons=[5],
        )
        assert len(results) > 0
        metrics = {r.metric for r in results}
        # log-mx metrics present, actuarial-from-birth metrics correctly omitted
        assert "rmse_log_mx" in metrics
        assert "rmse_e0" not in metrics

    def test_full_grid_has_actuarial_metrics(self):
        d = load_country("FRATNP")
        results = rolling_origin_eval(
            lambda: LeeCarter(),
            d["log_mx"], d["ages"], d["years"], d["exposures"], d["deaths"],
            origins=[2010], horizons=[5],
        )
        metrics = {r.metric for r in results}
        assert "rmse_e0" in metrics
        assert "rmse_annuity_65" in metrics


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HMD data not downloaded")
class TestScenarios:
    def test_short_history_returns_results(self):
        """Truncated training must still leave room for the forecast horizon."""
        d = load_country("FRATNP")
        results = short_history_eval(
            lambda: LeeCarter(),
            d["log_mx"], d["ages"], d["years"],
            train_lengths=[20, 30, 50], horizons=[5, 10],
            exposures=d["exposures"], deaths=d["deaths"],
        )
        assert len(results) > 0
        metrics = {r.metric for r in results}
        # every requested length must actually be evaluated
        for length in (20, 30, 50):
            assert f"rmse_log_mx_hist{length}" in metrics

    def test_age_groups_no_actuarial_on_subgrids(self):
        """e0/ä65 on a truncated age grid (e.g. 0-19) are meaningless — must be absent."""
        d = load_country("FRATNP")
        results = age_group_eval(
            lambda: LeeCarter(),
            d["log_mx"], d["ages"], d["years"],
            origins=[2013], horizons=[5],
            exposures=d["exposures"], deaths=d["deaths"],
        )
        metrics = {r.metric for r in results}
        assert "rmse_log_mx_young" in metrics
        assert not any("e0" in m or "annuity" in m for m in metrics)


class TestDieboldMariano:
    def test_symmetry(self):
        rng = np.random.default_rng(0)
        e1, e2 = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
        s1, _ = diebold_mariano_test(e1, e2)
        s2, _ = diebold_mariano_test(e2, e1)
        assert np.isclose(s1, -s2, atol=1e-9)

    def test_multistep_less_confident(self):
        """With serially-uncorrelated loss differentials the HLN small-sample factor
        (< 1 for h > 1) shrinks the statistic, so a larger assumed horizon is less
        confident. For autocorrelated differentials the autocovariance term can offset
        this, which is the correct statistical behaviour."""
        rng = np.random.default_rng(1)
        e1 = rng.normal(0.0, 1.0, 60)
        e2 = rng.normal(0.3, 1.0, 60)  # independent of e1
        s1, _ = diebold_mariano_test(e1, e2, horizon=1)
        s10, _ = diebold_mariano_test(e1, e2, horizon=10)
        assert abs(s10) < abs(s1)

    def test_pvalue_range(self):
        rng = np.random.default_rng(2)
        e1, e2 = rng.normal(0, 1, 40), rng.normal(0, 1, 40)
        _, p = diebold_mariano_test(e1, e2)
        assert 0.0 <= p <= 1.0
