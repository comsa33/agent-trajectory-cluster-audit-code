"""K-Means with automatic K selection via silhouette over a search range."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..registry import CLUSTERERS

logger = logging.getLogger(__name__)


class AutoKMeansClusterer:
    name = "auto_kmeans"

    def __init__(
        self,
        k_search_range: tuple[int, int] | list[int] = (3, 12),
        random_state: int = 42,
        n_init: int = 10,
        **_: object,
    ) -> None:
        if isinstance(k_search_range, list):
            k_search_range = tuple(k_search_range)  # type: ignore[assignment]
        self._k_min, self._k_max = k_search_range  # type: ignore[misc]
        self._random_state = random_state
        self._n_init = n_init
        self.extras: dict[str, object] = {}

    def fit_predict(self, matrix: np.ndarray) -> np.ndarray:
        n_samples = matrix.shape[0]
        k_max = min(self._k_max, max(2, n_samples - 1))
        k_min = max(2, min(self._k_min, k_max))
        if k_min > k_max:
            k_min = k_max

        best_score = -np.inf
        best_labels = None
        best_k = k_min
        scores: dict[int, float] = {}

        for k in range(k_min, k_max + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self._random_state,
                n_init=self._n_init,
            )
            labels = model.fit_predict(matrix)
            if len(np.unique(labels)) < 2:
                continue
            try:
                score = float(silhouette_score(matrix, labels))
            except ValueError:
                continue
            scores[k] = score
            logger.info("auto_kmeans: k=%d silhouette=%.4f", k, score)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

        if best_labels is None:
            # Fall back to k_min so we always return something usable.
            model = KMeans(
                n_clusters=k_min,
                random_state=self._random_state,
                n_init=self._n_init,
            )
            best_labels = model.fit_predict(matrix)
            best_k = k_min

        self.extras["selected_k"] = best_k
        self.extras["k_silhouette_scores"] = scores
        return best_labels


CLUSTERERS.register("auto_kmeans")(AutoKMeansClusterer)
