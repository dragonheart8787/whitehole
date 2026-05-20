from .base import BaseSimulator, SimData  # noqa: F401
from .grav_wave import GravitationalWaveSimulator  # noqa: F401
from .em_burst import EMBurstSimulator, XRayLightCurveSimulator  # noqa: F401
from .image_shadow import ImageShadowSimulator  # noqa: F401

SIMULATOR_REGISTRY: dict[str, type[BaseSimulator]] = {
    "gw": GravitationalWaveSimulator,
    "radio": EMBurstSimulator,
    "xray": XRayLightCurveSimulator,
    "image": ImageShadowSimulator,
}


def get_simulator(channel: str, **kwargs) -> BaseSimulator:
    """Instantiate a simulator by channel name."""
    if channel not in SIMULATOR_REGISTRY:
        raise KeyError(f"Unknown channel {channel!r}. Available: {list(SIMULATOR_REGISTRY)}")
    return SIMULATOR_REGISTRY[channel](**kwargs)
