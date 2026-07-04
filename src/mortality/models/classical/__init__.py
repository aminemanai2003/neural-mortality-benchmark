from mortality.models.classical.baselines import FrozenRates, RandomWalkDrift
from mortality.models.classical.bms import BoothMaindonaldSmith
from mortality.models.classical.cbd import CairnsBlakeDowd
from mortality.models.classical.hyndman_ullah import HyndmanUllah
from mortality.models.classical.lee_carter import LeeCarter
from mortality.models.classical.lee_miller import LeeMiller
from mortality.models.classical.poisson_lc import PoissonLeeCarter

CLASSICAL_MODELS = {
    "lee_carter": LeeCarter,
    "lee_miller": LeeMiller,
    "bms": BoothMaindonaldSmith,
    "poisson_lc": PoissonLeeCarter,
    "cbd": CairnsBlakeDowd,
    "hyndman_ullah": HyndmanUllah,
    "random_walk": RandomWalkDrift,
    "frozen_rates": FrozenRates,
}

__all__ = [
    "LeeCarter",
    "LeeMiller",
    "BoothMaindonaldSmith",
    "PoissonLeeCarter",
    "CairnsBlakeDowd",
    "HyndmanUllah",
    "RandomWalkDrift",
    "FrozenRates",
    "CLASSICAL_MODELS",
]
