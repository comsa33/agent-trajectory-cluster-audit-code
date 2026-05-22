"""CSV-shaped outputs and per-cluster aggregate statistics."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..analysis import classify_clusters

logger = logging.getLogger(__name__)


_METRIC_CANDIDATES = [
    "accuracy",
    "tokens",
    "time_s",
    "n_iters",
    "empty_observation_ratio",
    "repeated_tool_ratio",
    "repeated_search_keyword_ratio",
    "unique_tool_count",
    "fixation_score",
    "tool_transition_entropy",
    "tool_bigram_diversity",
    "empty_observation_burst",
    "retry_burst_max",
    "retry_burst_ratio",
    "thought_drift_slope",
    "hypothesis_persistence",
    "exploration_index",
    "finalization_lateness",
    "reasoning_depth_growth",
    "search_keyword_diversity",
]


def _aggregate_top_tools(
    counters: Iterable[dict[str, int]], k: int = 3
) -> str:
    bag: Counter[str] = Counter()
    for counter in counters:
        if isinstance(counter, dict):
            bag.update(counter)
    top = bag.most_common(k)
    return "; ".join(f"{name}:{count}" for name, count in top) if top else ""


def _representative_query_ids(group: pd.DataFrame, k: int = 3) -> str:
    if group.empty:
        return ""
    sort_keys: list[str] = []
    if "final_correct" in group.columns:
        sort_keys.append("final_correct")
    if "tokens" in group.columns:
        sort_keys.append("tokens")
    if not sort_keys:
        return ", ".join(str(q) for q in group["query_id"].head(k).tolist())
    sample = (
        group.sort_values(by=sort_keys, ascending=[True] * len(sort_keys))[
            "query_id"
        ]
        .head(k)
        .tolist()
    )
    return ", ".join(str(q) for q in sample)


def build_cluster_summary(
    features_df: pd.DataFrame,
    rule_set: str = "default",
) -> pd.DataFrame:
    # Aggregate per-cluster stats and assign a heuristic descriptive tag.
    # The tag is a one-line cluster summary derived from feature thresholds,
    # NOT a validated failure-type label. See docs/leakage_fix_comparison.md
    # and docs/planning_phenotype.md for the evidence on why we do not claim
    # these tags constitute a failure-mode taxonomy.
    rows: list[dict[str, object]] = []
    for cluster_id, group in features_df.groupby("cluster_id", sort=True):
        stats: dict[str, object] = {
            "cluster_id": int(cluster_id),
            "size": int(len(group)),
            "accuracy": float(group["final_correct"].mean()) if "final_correct" in group else float("nan"),
        }
        for col in _METRIC_CANDIDATES:
            if col in group.columns and col != "accuracy":
                stats[f"avg_{col}"] = float(group[col].mean())
        if "top_tool_counter" in group.columns:
            stats["top_tools"] = _aggregate_top_tools(
                group["top_tool_counter"].tolist()
            )
        else:
            stats["top_tools"] = ""
        stats["representative_query_ids"] = _representative_query_ids(group)
        rows.append(stats)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df["descriptive_cluster_tag"] = classify_clusters(
            summary_df, rule_set=rule_set
        ).values
    return summary_df


def write_outputs(
    features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    coords_2d: np.ndarray,
    output_dir: Path,
    rule_set: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    # Persist core CSV artifacts and return the enriched dataframe + summary.
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched = features_df.copy()
    enriched["cluster_id"] = cluster_labels.astype(int)
    enriched["pca_x"] = coords_2d[:, 0]
    enriched["pca_y"] = coords_2d[:, 1]

    feature_csv = output_dir / "trajectory_features.csv"
    assignments_csv = output_dir / "cluster_assignments.csv"
    summary_csv = output_dir / "cluster_summary.csv"

    feature_view = enriched.drop(columns=["top_tool_counter"], errors="ignore")
    feature_view.to_csv(feature_csv, index=False)

    assignment_cols = [
        "query_id",
        "source_dir",
        "phase",
        "cluster_id",
        "final_correct",
        "tokens",
        "time_s",
        "n_iters",
    ]
    assignment_cols = [c for c in assignment_cols if c in enriched.columns]
    enriched[assignment_cols].to_csv(assignments_csv, index=False)

    summary_df = build_cluster_summary(enriched, rule_set=rule_set)
    summary_df.to_csv(summary_csv, index=False)

    paths = {
        "trajectory_features": feature_csv,
        "cluster_assignments": assignments_csv,
        "cluster_summary": summary_csv,
    }
    logger.info("Wrote core CSV artifacts under %s", output_dir)
    return enriched, summary_df, paths
