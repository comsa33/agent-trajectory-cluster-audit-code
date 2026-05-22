"""HDBSCAN clusterer (optional dependency, density-based)."""
from __future__ import annotations

import logging

import numpy as np

from ..registry import CLUSTERERS

logger = logging.getLogger(__name__)


class HDBSCANClusterer:
    name = "hdbscan"

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        **_: object,
    ) -> None:
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self.extras: dict[str, object] = {}

    def fit_predict(self, matrix: np.ndarray) -> np.ndarray:
        try:
            import hdbscan  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "hdbscan is not installed. Install with `uv pip install hdbscan` "
                "or switch clustering.type to 'kmeans' / 'auto_kmeans'."
            ) from exc

        model = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
        )
        labels = model.fit_predict(matrix)
        n_noise = int(np.sum(labels == -1))
        self.extras["noise_count"] = n_noise
        if n_noise:
            logger.info("HDBSCAN classified %d points as noise (label=-1)", n_noise)
        return labels


CLUSTERERS.register("hdbscan")(HDBSCANClusterer)
