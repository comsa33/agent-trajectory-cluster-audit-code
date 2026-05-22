"""Matplotlib plots — scatter, metric bars, pathology distribution."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


def plot_scatter(features_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(features_df["cluster_id"].unique())
    cmap = plt.colormaps.get_cmap("tab10")
    for i, cluster_id in enumerate(clusters):
        subset = features_df[features_df["cluster_id"] == cluster_id]
        label = f"Cluster {cluster_id} (n={len(subset)})"
        if cluster_id == -1:
            label = f"Noise (n={len(subset)})"
        ax.scatter(
            subset["pca_x"],
            subset["pca_y"],
            label=label,
            s=42,
            alpha=0.75,
            color=cmap(i % 10),
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_title("Trajectory Behavioral Clusters (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df.empty:
        logger.warning("Empty cluster summary; skipping metrics plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cluster_ids = summary_df["cluster_id"].astype(str).tolist()

    axes[0].bar(cluster_ids, summary_df.get("accuracy", []), color="#4C72B0")
    axes[0].set_title("Accuracy per Cluster")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(cluster_ids, summary_df.get("avg_tokens", []), color="#DD8452")
    axes[1].set_title("Avg Tokens per Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Tokens")

    axes[2].bar(cluster_ids, summary_df.get("avg_n_iters", []), color="#55A868")
    axes[2].set_title("Avg Iterations per Cluster")
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("Iterations")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pathology_distribution(summary_df: pd.DataFrame, path: Path) -> None:
    # Bar chart of trajectories per descriptive cluster tag (heuristic).
    if summary_df.empty or "descriptive_cluster_tag" not in summary_df:
        return
    counts = (
        summary_df.groupby("descriptive_cluster_tag")["size"]
        .sum()
        .sort_values(ascending=False)
    )
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.tolist(), counts.values, color="#8172B2")
    ax.set_title("Trajectory Count per Descriptive Cluster Tag (heuristic)")
    ax.set_ylabel("Trajectories")
    ax.set_xlabel("Descriptive cluster tag")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
