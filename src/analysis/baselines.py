"""Simple-baseline cluster assignments and head-to-head comparison.

Reviewer demand: prove the full-feature clustering actually beats trivial
heuristics. We compute cluster ids from single columns (quantile bins),
random labels, and oracle labels (the ground-truth column itself), then
score each baseline against the same label columns the main run uses.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .labeled_validation import _cluster_purity  # reuse the same metric impl

logger = logging.getLogger(__name__)


def _quantile_bin(series: pd.Series, k: int) -> np.ndarray:
    # Equal-frequency quantile bins; falls back to fewer bins on ties.
    try:
        binned = pd.qcut(series.astype(float), q=k, labels=False, duplicates="drop")
    except ValueError:
        binned = pd.cut(series.astype(float), bins=k, labels=False)
    return binned.fillna(-1).astype(int).to_numpy()


def _random_labels(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=n)


def _oracle_labels(series: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(series.astype(str))
    return codes


def baseline_cluster_ids(
    features_df: pd.DataFrame,
    spec: str,
    k: int,
    seed: int,
) -> np.ndarray:
    """Resolve a baseline spec string to a cluster-id array."""
    if spec == "random":
        return _random_labels(len(features_df), k, seed)
    if spec.startswith("quantile:"):
        col = spec.split(":", 1)[1]
        if col not in features_df.columns:
            raise KeyError(f"baseline column missing: {col}")
        return _quantile_bin(features_df[col], k)
    if spec.startswith("oracle:"):
        col = spec.split(":", 1)[1]
        if col not in features_df.columns:
            raise KeyError(f"oracle column missing: {col}")
        return _oracle_labels(features_df[col])
    raise ValueError(f"Unknown baseline spec: {spec}")


def _score_against_labels(
    cluster_ids: np.ndarray,
    df: pd.DataFrame,
    label_columns: list[str],
    baseline_name: str,
    n_clusters: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in label_columns:
        if label not in df.columns:
            continue
        sub_mask = df[label].notna() & (cluster_ids >= 0)
        if not sub_mask.any():
            continue
        cluster_sub = cluster_ids[sub_mask.to_numpy()]
        labels_sub = df.loc[sub_mask, label]
        if labels_sub.nunique() < 2:
            continue
        codes, _ = pd.factorize(labels_sub.astype(str))
        rows.append(
            {
                "baseline": baseline_name,
                "n_clusters": int(len(np.unique(cluster_sub[cluster_sub >= 0]))),
                "n_clusters_target": int(n_clusters),
                "label": label,
                "n_labeled": int(len(codes)),
                "purity": _cluster_purity(cluster_sub, codes),
                "nmi": float(normalized_mutual_info_score(codes, cluster_sub)),
                "ari": float(adjusted_rand_score(codes, cluster_sub)),
            }
        )
    return rows


def compare_baselines(
    enriched_df: pd.DataFrame,
    label_columns: list[str],
    baseline_specs: list[str],
    k: int,
    seed: int = 42,
    main_cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """Run every baseline + the main clustering result through the same scoring.

    The full-feature clustering is included as `full_features` so the
    output is a single comparison table.
    """
    rows: list[dict[str, object]] = []

    # Main result row.
    if main_cluster_column in enriched_df.columns:
        rows.extend(
            _score_against_labels(
                enriched_df[main_cluster_column].to_numpy(),
                enriched_df,
                label_columns,
                baseline_name="full_features",
                n_clusters=k,
            )
        )

    for spec in baseline_specs:
        try:
            cluster_ids = baseline_cluster_ids(enriched_df, spec, k=k, seed=seed)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping baseline %s: %s", spec, exc)
            continue
        rows.extend(
            _score_against_labels(
                cluster_ids, enriched_df, label_columns, baseline_name=spec, n_clusters=k
            )
        )

    return pd.DataFrame(rows)
