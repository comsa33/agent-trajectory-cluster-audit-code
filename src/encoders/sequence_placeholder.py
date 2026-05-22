"""Placeholder slot for a future sequence-aware trajectory encoder.

The intent is that a transformer / LSTM trained directly on the
(thought, tool_call, observation) token stream will live here. Registering
the name now means experiment configs can reference it the moment an
implementation lands; today it raises a NotImplementedError so misconfigured
experiments fail loudly instead of silently degrading.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..registry import ENCODERS


class SequenceEncoderPlaceholder:
    name = "sequence"

    def __init__(self, **kwargs: object) -> None:
        # Kwargs are accepted but ignored so configs can be authored ahead
        # of the actual implementation.
        self._kwargs = kwargs

    def fit_encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(
            "Sequence encoder is not implemented yet. "
            "Drop in a transformer / LSTM here and wire it through "
            "register('sequence')."
        )


ENCODERS.register("sequence")(SequenceEncoderPlaceholder)
