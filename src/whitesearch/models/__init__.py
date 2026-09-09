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


#: Which model channels each data channel accepts.  'generic' is the null
#: hypothesis, which fits anywhere; 'radio' is accepted on 'xray' because
#: PBHTunnelingWhiteHole carries the gamma-ray efficiency the X-ray light-curve
#: simulator uses.  Previously this table lived only in cli.py and guarded the
#: fit side; the injection side (dataio.loader._load_mock) had no check at all,
#: which docs/RADIO_PREFLIGHT_AUDIT.md R.12.2 records.
CHANNEL_COMPATIBILITY: dict[str, frozenset[str]] = {
    "gw": frozenset({"gw", "generic"}),
    "radio": frozenset({"radio", "generic"}),
    "xray": frozenset({"xray", "radio", "generic"}),
    "image": frozenset({"image", "generic"}),
}


def check_model_channel(model_name: str, channel: str) -> None:
    """Raise unless ``model_name`` may be used on data channel ``channel``.

    Fail-closed: an unrecognised data channel accepts nothing rather than
    everything, so adding a channel without deciding its compatibility is an
    error rather than a silent pass.
    """
    model_channel = get_model(model_name).channel
    allowed = CHANNEL_COMPATIBILITY.get(channel, frozenset())
    if model_channel not in allowed:
        raise ValueError(
            f"Model {model_name!r} (native channel={model_channel!r}) cannot be "
            f"used on data channel {channel!r}; that channel accepts "
            f"{sorted(allowed) if allowed else 'no model channels (unknown data channel)'}."
        )
