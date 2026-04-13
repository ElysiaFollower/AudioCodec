"""AudioCodec package."""

from .config import CodecExperimentConfig, load_experiment_config

__all__ = [
    "CodecExperimentConfig",
    "load_experiment_config",
]

try:
    from .models.codec import CodecOutput, ConvRVQCodec, EncodedAudio

    __all__.extend(["CodecOutput", "ConvRVQCodec", "EncodedAudio"])
except ModuleNotFoundError:
    # Keep config-only imports usable before the torch environment is prepared.
    pass
