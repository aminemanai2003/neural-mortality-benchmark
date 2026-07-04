from mortality.models.neural.cnn_surface import CNNSurface
from mortality.models.neural.ffnn_embeddings import FFNNEmbeddings
from mortality.models.neural.kt_models import NeuralKtModel

NEURAL_MODELS = {
    "lstm_kt": lambda **kw: NeuralKtModel(arch="lstm", **kw),
    "gru_kt": lambda **kw: NeuralKtModel(arch="gru", **kw),
    "bilstm_kt": lambda **kw: NeuralKtModel(arch="bilstm", **kw),
    "transformer_kt": lambda **kw: NeuralKtModel(arch="transformer", **kw),
    "ffnn_embeddings": lambda **kw: FFNNEmbeddings(**kw),
    "cnn_surface": lambda **kw: CNNSurface(**kw),
}

__all__ = ["NeuralKtModel", "FFNNEmbeddings", "CNNSurface", "NEURAL_MODELS"]
