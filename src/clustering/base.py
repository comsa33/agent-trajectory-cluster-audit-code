"""Clustering protocol, matrix assembly, and orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..registry import CLUSTERERS

logger = logging.getLogger(__name__)


class Clusterer(Protocol):
    name: str

    def fit_predict(self, matrix: np.ndarray) -> np.ndarray: ...


@dataclass
class ClusterResult:
    labels: np.ndarray
    matrix: np.ndarray
    coords_2d: np.ndarray
    silhouette: float | None
    calinski_harabasz: float | None
    mode: str
    clusterer_name: str
    n_clusters: int
    extras: dict[str, Any]


def build_feature_matrix(
    features_df: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    matrix = features_df[feature_columns].to_numpy(dtype=np.float64, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(matrix)


def combine_hybrid(feat: np.ndarray, text: np.ndarray) -> np.ndarray:
    feat_norm = np.linalg.norm(feat, axis=1, keepdims=True)
    text_norm = np.linalg.norm(text, axis=1, keepdims=True)
    feat_scaled = feat / np.where(feat_norm == 0, 1.0, feat_norm)
    text_scaled = text / np.where(text_norm == 0, 1.0, text_norm)
    return np.hstack([feat_scaled, text_scaled])


def _safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    if matrix.shape[0] < 3:
        return None
    valid_labels = labels[labels >= 0]
    if len(np.unique(valid_labels)) < 2:
        return None
    mask = labels >= 0
    if mask.sum() < 3:
        return None
    try:
        return float(silhouette_score(matrix[mask], labels[mask]))
    except ValueError as exc:
        logger.warning("Silhouette score failed: %s", exc)
        return None


def _safe_ch_score(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    valid_labels = labels[labels >= 0]
    if len(np.unique(valid_labels)) < 2:
        return None
    mask = labels >= 0
    if mask.sum() < 3:
        return None
    try:
        return float(calinski_harabasz_score(matrix[mask], labels[mask]))
    except ValueError as exc:
        logger.warning("Calinski-Harabasz failed: %s", exc)
        return None


def _project_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] <= 2:
        # Pad to 2 columns for plotting consistency.
        padded = np.zeros((matrix.shape[0], 2), dtype=np.float64)
        padded[:, : matrix.shape[1]] = matrix
        return padded
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], 2), dtype=np.float64)
    return PCA(n_components=2, random_state=42).fit_transform(matrix)


def build_clusterer(config: dict[str, Any], random_state: int) -> Clusterer:
    factory = CLUSTERERS.get(config.get("type", "kmeans"))
    kwargs = {k: v for k, v in config.items() if k not in {"type", "mode"}}
    kwargs.setdefault("random_state", random_state)
    return factory(**kwargs)


def run_clustering(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    text_embeddings: np.ndarray | None,
    cluster_config: dict[str, Any],
    random_state: int,
) -> ClusterResult:
    mode = cluster_config.get("mode", "hybrid")
    feature_matrix = build_feature_matrix(features_df, feature_columns)

    if mode == "feature":
        matrix = feature_matrix
        effective_mode = "feature"
    elif mode == "text":
        if text_embeddings is None or text_embeddings.size == 0:
            logger.warning(
                "Text embeddings unavailable; falling back to feature-only mode."
            )
            matrix = feature_matrix
            effective_mode = "feature"
        else:
            matrix = text_embeddings
            effective_mode = "text"
    elif mode == "hybrid":
        if text_embeddings is None or text_embeddings.size == 0:
            logger.warning(
                "Text embeddings unavailable; hybrid degrades to feature-only."
            )
            matrix = feature_matrix
            effective_mode = "feature"
        else:
            matrix = combine_hybrid(feature_matrix, text_embeddings)
            effective_mode = "hybrid"
    else:
        raise ValueError(f"Unknown clustering mode: {mode}")

    clusterer = build_clusterer(cluster_config, random_state)
    labels = clusterer.fit_predict(matrix)

    silhouette = _safe_silhouette(matrix, labels)
    ch_score = _safe_ch_score(matrix, labels)
    coords_2d = _project_2d(matrix)
    n_clusters = int(len(np.unique(labels[labels >= 0])))

    extras: dict[str, Any] = {}
    if hasattr(clusterer, "extras"):
        extras = dict(clusterer.extras)  # type: ignore[arg-type]

    logger.info(
        "Clustered %d trajectories into %d clusters "
        "(clusterer=%s, mode=%s, silhouette=%s, CH=%s)",
        matrix.shape[0],
        n_clusters,
        clusterer.name,
        effective_mode,
        f"{silhouette:.4f}" if silhouette is not None else "n/a",
        f"{ch_score:.2f}" if ch_score is not None else "n/a",
    )

    return ClusterResult(
        labels=labels,
        matrix=matrix,
        coords_2d=coords_2d,
        silhouette=silhouette,
        calinski_harabasz=ch_score,
        mode=effective_mode,
        clusterer_name=clusterer.name,
        n_clusters=n_clusters,
        extras=extras,
    )
