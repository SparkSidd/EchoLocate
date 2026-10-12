"""
EchoLocate — Speaker Encoder Factory
====================================
Factory function for dynamic, model-agnostic speaker encoder instantiation.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import SpeakerEncoder
from .campp_encoder import CAMPPEncoder
from .ecapa_encoder import ECAPAEncoder
from .eres2netv2_encoder import ERes2NetV2Encoder
from .mock_encoder import MockAcousticEncoder

ENCODER_REGISTRY: Dict[str, Type[SpeakerEncoder]] = {
    "campp": CAMPPEncoder,
    "cam++": CAMPPEncoder,
    "eres2netv2": ERes2NetV2Encoder,
    "eres2net": ERes2NetV2Encoder,
    "ecapa": ECAPAEncoder,
    "ecapa_tdnn": ECAPAEncoder,
    "mock": MockAcousticEncoder,
}


def create_encoder(
    model_name: str = "campp",
    model_path: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> SpeakerEncoder:
    """
    Instantiate a speaker encoder by architecture name.
    
    Args:
        model_name: "campp", "eres2netv2", "ecapa", or "mock"
        model_path: Optional path to saved model weights (.pt / .pth)
        device: "cpu" or "cuda"
        **kwargs: Additional model-specific hyperparameters
    """
    key = model_name.lower().strip()
    if key not in ENCODER_REGISTRY:
        valid = list(ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown speaker encoder '{model_name}'. Valid options: {valid}")

    cls = ENCODER_REGISTRY[key]
    if key == "mock":
        return cls(name="mock", **kwargs)

    # Set default dims to match pre-trained weights if not specified in kwargs
    if key in ["eres2netv2", "eres2net"] and "embedding_dim" not in kwargs:
        kwargs["embedding_dim"] = 512
    elif key in ["campp", "cam++"] and "embedding_dim" not in kwargs:
        kwargs["embedding_dim"] = 192
    elif key in ["ecapa", "ecapa_tdnn"] and "embedding_dim" not in kwargs:
        kwargs["embedding_dim"] = 192

    return cls(model_path=model_path, device=device, **kwargs)
