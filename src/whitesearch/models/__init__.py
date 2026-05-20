from .base import BaseModel, ParameterSpec  # noqa: F401
from .gr_eternal import GREternalWhiteHole  # noqa: F401
from .bounce import BlackToWhiteBounce  # noqa: F401
from .pbh_tunneling import PBHTunnelingWhiteHole  # noqa: F401
from .alternatives import (  # noqa: F401
    NullHypothesis,
    MagnetarFlare,
    GRBAfterglowFRB,
    StandardBHRingdown,
    BHAccretion,
)

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "gr_eternal": GREternalWhiteHole,
    "bounce": BlackToWhiteBounce,
    "pbh_tunneling": PBHTunnelingWhiteHole,
    "null": NullHypothesis,
    "magnetar": MagnetarFlare,
    "grb_frb": GRBAfterglowFRB,
    "bh_ringdown": StandardBHRingdown,
    "bh_accretion": BHAccretion,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
