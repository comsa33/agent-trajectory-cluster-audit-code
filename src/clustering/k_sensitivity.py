"""Sweep K for KMeans and score each K against the configured labels.

Used to ask: when we vary the number of clusters, does the cluster ↔
label_X NMI track an underlying signal, or did the main run merely
bottom out at the K that happens to mirror task_type?
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def _safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2 or matrix.shape[0] < 3:
        return None
    try:
        return float(silhouette_score(matrix, labels))
    except ValueError:
        return None


def _safe_ch(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    try:
        return float(calinski_harabasz_score(matrix, labels))
    except ValueError:
        return None


def k_sensitivity_sweep(
    matrix: np.ndarray,
    enriched_df: pd.DataFrame,
    k_values: list[int],
    label_columns: list[str],
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """For each k, fit KMeans on `matrix`, compute silhouette + per-label
    NMI / ARI, return a long-form table.
    """
    rows: list[dict[str, object]] = []
    n_samples = matrix.shape[0]
    for k in k_values:
        if k < 2 or k >= n_samples:
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(matrix)
        sil = _safe_silhouette(matrix, labels)
        ch = _safe_ch(matrix, labels)

        row: dict[str, object] = {
            "k": int(k),
            "silhouette": sil,
            "calinski_harabasz": ch,
        }
        for col in label_columns:
            if col not in enriched_df.columns:
                continue
            sub = pd.DataFrame({"cid": labels, "lab": enriched_df[col].values}).dropna()
            if sub.empty or sub["lab"].nunique() < 2:
                row[f"nmi__{col}"] = None
                row[f"ari__{col}"] = None
                continue
            codes, _ = pd.factorize(sub["lab"].astype(str))
            cids = sub["cid"].to_numpy()
            row[f"nmi__{col}"] = float(normalized_mutual_info_score(codes, cids))
            row[f"ari__{col}"] = float(adjusted_rand_score(codes, cids))
        rows.append(row)
        logger.info(
            "k_sensitivity: k=%d silhouette=%s",
            k,
            f"{sil:.4f}" if sil is not None else "n/a",
        )
    return pd.DataFrame(rows)
