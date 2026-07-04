from mortality.models.hybrid.lc_resnet import LCResNet

HYBRID_MODELS = {
    "lc_resnet": LCResNet,
}

__all__ = ["LCResNet", "HYBRID_MODELS"]
