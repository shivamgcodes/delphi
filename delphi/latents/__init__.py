from .cache import LatentCache
from .constructors import (
    constructor,
    neighbour_non_activation_windows,
    pool_max_activation_windows,
    random_non_activating_windows,
)
from .latents import (
    ActivatingExample,
    Example,
    Latent,
    LatentRecord,
    NonActivatingExample,
)
from .loader import LatentDataset
from .samplers import sampler


def __getattr__(name: str):
    if name == "MoELatentCache":
        from delphi.latents.cache_moe import MoELatentCache

        return MoELatentCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LatentCache",
    "MoELatentCache",
    "LatentDataset",
    "Latent",
    "LatentRecord",
    "Example",
    "ActivatingExample",
    "NonActivatingExample",
    "pool_max_activation_windows",
    "random_non_activating_windows",
    "neighbour_non_activation_windows",
    "constructor",
    "sampler",
]
