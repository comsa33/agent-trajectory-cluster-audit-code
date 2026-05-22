"""sentence-transformers encoder with lazy import + graceful failure."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..registry import ENCODERS

logger = logging.getLogger(__name__)


class SentenceTransformerEncoder:
    name = "sentence_transformer"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        **_: object,
    ) -> None:
        self._model_name = model
        self._batch_size = batch_size
        self._model = None  # lazy

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with `uv pip install sentence-transformers` "
                "or switch encoder.type to 'tfidf' / 'none'."
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        logger.info("Loaded SentenceTransformer model: %s", self._model_name)

    def fit_encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._load()
        vectors = self._model.encode(  # type: ignore[union-attr]
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


ENCODERS.register("sentence_transformer")(SentenceTransformerEncoder)
