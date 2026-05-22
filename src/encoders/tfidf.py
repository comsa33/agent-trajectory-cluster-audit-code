"""TF-IDF baseline encoder — zero external dependencies, useful as a control."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..registry import ENCODERS


class TfidfEncoder:
    name = "tfidf"

    def __init__(
        self,
        max_features: int = 4096,
        ngram_range: tuple[int, int] | list[int] = (1, 2),
        **_: object,
    ) -> None:
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)  # type: ignore[assignment]
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
        )

    def fit_encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = self._vectorizer.fit_transform(list(texts))
        # L2-normalize so cosine similarity == dot product downstream.
        matrix = normalize(matrix, norm="l2", axis=1)
        return matrix.toarray().astype(np.float32)


ENCODERS.register("tfidf")(TfidfEncoder)
