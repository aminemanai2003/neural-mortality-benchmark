import numpy as np
import pytest

from mortality.data.loader import load_country
from mortality.models.neural import NEURAL_MODELS

DATA_AVAILABLE = True
try:
    load_country("FRATNP")
except FileNotFoundError:
    DATA_AVAILABLE = False


@pytest.fixture
def france_data():
    return load_country("FRATNP")


@pytest.mark.skipif(not DATA_AVAILABLE, reason="HMD data not downloaded")
class TestNeuralModels:
    @pytest.mark.parametrize("name", [
        "lstm_kt", "gru_kt", "bilstm_kt", "transformer_kt",
        "ffnn_embeddings", "cnn_surface",
    ])
    def test_fit_forecast(self, name, france_data):
        d = france_data
        model = NEURAL_MODELS[name](epochs=5, patience=3, seed=42)
        model.fit(d["log_mx"], d["ages"], d["years"])
        fc = model.forecast(5)
        assert fc.shape[1] == 5
        assert not np.any(np.isnan(fc))

    def test_lstm_simulate_shape(self, france_data):
        d = france_data
        model = NEURAL_MODELS["lstm_kt"](epochs=5, patience=3, seed=42)
        model.fit(d["log_mx"], d["ages"], d["years"])
        sim = model.simulate(5, n_paths=10)
        assert sim.shape == (10, 101, 5)
