"""External-label validation: cluster purity, NMI, ARI vs ground-truth columns.

Also provides:
- `evaluate_within_groups`: stratified validation, used to ask "after we
  hold task domain constant, does the cluster ↔ failure_type signal
  survive or was it just a domain confound?"
- `evaluate_multilabel`: per-tag binary validation for label columns that
  carry list-of-tags values (e.g. AgentErrorBench `failure_types`).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

logger = logging.getLogger(__name__)


def _cluster_purity(cluster_ids: np.ndarray, labels: np.ndarray) -> float:
    # Standard clustering purity: for each cluster, take the dominant label
    # and weight by cluster size. Defined for non-noise clusters only.
    valid = cluster_ids >= 0
    if not valid.any():
        return float("nan")
    clusters = cluster_ids[valid]
    labs = labels[valid]
    total = clusters.size
    correct = 0
    for cid in np.unique(clusters):
        mask = clusters == cid
        cluster_labs = labs[mask]
        # mode is the dominant label.
        unique, counts = np.unique(cluster_labs, return_counts=True)
        correct += int(counts.max())
    return float(correct / total)


def _chi_square(cluster_ids: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    contingency = pd.crosstab(pd.Series(cluster_ids), pd.Series(labels)).to_numpy()
    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return float("nan"), float("nan")
    try:
        chi2, p, _, _ = sps.chi2_contingency(contingency)
        return float(chi2), float(p)
    except ValueError as exc:
        logger.warning("chi2 failed: %s", exc)
        return float("nan"), float("nan")


def evaluate_label_columns(
    df: pd.DataFrame,
    label_columns: list[str],
) -> pd.DataFrame:
    """Compare `df.cluster_id` against each `label_columns` entry.

    Rows of NA are dropped per-label so each metric is computed on the
    intersection of labeled trajectories. Categorical labels are factorized
    so NMI / ARI receive integer arrays.
    """
    if "cluster_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster_id' column")

    rows: list[dict[str, object]] = []
    for label in label_columns:
        if label not in df.columns:
            logger.info("label_validation: column %s missing; skipping", label)
            continue

        sub = df[["cluster_id", label]].dropna()
        if sub.empty:
            logger.info("label_validation: no labeled rows for %s; skipping", label)
            continue
        if sub[label].nunique() < 2:
            logger.info(
                "label_validation: only one class for %s; metrics undefined", label
            )
            rows.append(
                {
                    "label": label,
                    "n_labeled": int(len(sub)),
                    "n_classes": int(sub[label].nunique()),
                    "purity": float("nan"),
                    "nmi": float("nan"),
                    "ari": float("nan"),
                    "chi2": float("nan"),
                    "p_value": float("nan"),
                }
            )
            continue

        cluster_ids = sub["cluster_id"].to_numpy()
        # factorize handles strings, ints, and mixed types uniformly.
        label_codes, _ = pd.factorize(sub[label].astype(str))
        purity = _cluster_purity(cluster_ids, label_codes)
        nmi = float(normalized_mutual_info_score(label_codes, cluster_ids))
        ari = float(adjusted_rand_score(label_codes, cluster_ids))
        chi2, p_value = _chi_square(cluster_ids, label_codes)

        rows.append(
            {
                "label": label,
                "n_labeled": int(len(sub)),
                "n_classes": int(sub[label].nunique()),
                "purity": purity,
                "nmi": nmi,
                "ari": ari,
                "chi2": chi2,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows)


def evaluate_within_groups(
    df: pd.DataFrame,
    label_columns: list[str],
    group_column: str,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Run `evaluate_label_columns` separately within each value of
    `group_column`. Used to check whether cluster ↔ label alignment is
    actually a confound carried by the grouping variable (typically
    domain / task_type).
    """
    if group_column not in df.columns:
        logger.info(
            "stratified validation: group column %s missing; skipping",
            group_column,
        )
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for group_value, group_df in df.groupby(group_column):
        if len(group_df) < min_group_size:
            logger.info(
                "stratified validation: group %s=%s has %d < %d rows; skipping",
                group_column,
                group_value,
                len(group_df),
                min_group_size,
            )
            continue
        sub_metrics = evaluate_label_columns(group_df, label_columns)
        if sub_metrics.empty:
            continue
        sub_metrics.insert(0, "group_column", group_column)
        sub_metrics.insert(1, "group_value", group_value)
        rows.append(sub_metrics)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _coerce_tag_set(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None and str(x).strip()]
    if hasattr(value, "tolist"):
        try:
            return _coerce_tag_set(value.tolist())
        except Exception:  # noqa: BLE001
            return []
    if isinstance(value, str):
        # Allow "a|b|c" or "a,b,c" string forms.
        text = value.strip()
        if not text:
            return []
        if "|" in text:
            return [t.strip() for t in text.split("|") if t.strip()]
        if "," in text:
            return [t.strip() for t in text.split(",") if t.strip()]
        return [text]
    return []


def evaluate_multilabel(
    df: pd.DataFrame,
    multilabel_columns: list[str],
    cluster_column: str = "cluster_id",
    min_tag_count: int = 5,
) -> pd.DataFrame:
    """For each list-valued label column, expand into per-tag binary columns
    and compute per-tag purity / NMI / chi^2 vs cluster id. Tags with fewer
    than `min_tag_count` occurrences are skipped.
    """
    if cluster_column not in df.columns:
        raise ValueError(f"DataFrame must contain a {cluster_column!r} column")

    rows: list[dict[str, object]] = []
    for col in multilabel_columns:
        if col not in df.columns:
            logger.info("multilabel: column %s missing; skipping", col)
            continue

        tag_lists = df[col].apply(_coerce_tag_set)
        # Tag occurrence count.
        tag_counts: dict[str, int] = {}
        for tags in tag_lists:
            for t in set(tags):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        for tag, count in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
            if count < min_tag_count:
                continue
            binary = tag_lists.apply(lambda tags: int(tag in tags))
            sub = pd.DataFrame(
                {"cluster_id": df[cluster_column], "tag": binary}
            ).dropna()
            if sub["tag"].nunique() < 2:
                continue
            cluster_ids = sub["cluster_id"].to_numpy()
            tag_codes = sub["tag"].to_numpy()
            chi2, p_value = _chi_square(cluster_ids, tag_codes)
            rows.append(
                {
                    "label": col,
                    "tag": tag,
                    "support": int(count),
                    "purity": _cluster_purity(cluster_ids, tag_codes),
                    "nmi": float(normalized_mutual_info_score(tag_codes, cluster_ids)),
                    "ari": float(adjusted_rand_score(tag_codes, cluster_ids)),
                    "chi2": chi2,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)
