"""Encoder protocol + factory + null implementation."""
from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

import numpy as np

from ..registry import ENCODERS

logger = logging.getLogger(__name__)


class TrajectoryEncoder(Protocol):
    name: str

    def fit_encode(self, texts: Sequence[str]) -> np.ndarray: ...


class NullEncoder:
    """No-op encoder used when text-channel embeddings are disabled.

    Returns an empty (N, 0) matrix so downstream code can still concat without
    branching on the absence of embeddings.
    """

    name = "none"

    def fit_encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 0), dtype=np.float32)


ENCODERS.register("none")(lambda **_: NullEncoder())


def build_encoder(config: dict[str, Any]) -> TrajectoryEncoder:
    """Resolve `config["type"]` against the encoder registry.

    On failure (e.g. sentence-transformers not installed) the caller decides
    how to fall back; we surface the original error.
    """
    encoder_type = config.get("type", "none")
    factory = ENCODERS.get(encoder_type)
    kwargs = {k: v for k, v in config.items() if k != "type"}
    encoder = factory(**kwargs)
    logger.info("Built encoder: %s (%s)", encoder.name, encoder_type)
    return encoder
