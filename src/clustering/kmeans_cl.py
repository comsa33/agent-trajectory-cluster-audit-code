"""Vanilla K-Means clusterer."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..registry import CLUSTERERS


class KMeansClusterer:
    name = "kmeans"

    def __init__(
        self,
        n_clusters: int = 6,
        random_state: int = 42,
        n_init: int = 10,
        **_: object,
    ) -> None:
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._n_init = n_init
        self.extras: dict[str, object] = {}

    def fit_predict(self, matrix: np.ndarray) -> np.ndarray:
        n_samples = matrix.shape[0]
        effective_k = max(2, min(self._n_clusters, n_samples))
        if effective_k != self._n_clusters:
            self.extras["adjusted_k"] = effective_k
        model = KMeans(
            n_clusters=effective_k,
            random_state=self._random_state,
            n_init=self._n_init,
        )
        return model.fit_predict(matrix)


CLUSTERERS.register("kmeans")(KMeansClusterer)
