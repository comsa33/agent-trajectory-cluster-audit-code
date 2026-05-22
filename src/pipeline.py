"""End-to-end orchestration: config -> data -> features -> embed -> cluster -> report."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    cluster_outcome_correlation,
    compare_baselines,
    evaluate_label_columns,
    evaluate_multilabel,
    evaluate_within_groups,
)
from .clustering.k_sensitivity import k_sensitivity_sweep
from .clustering import run_clustering
from .config import ExperimentConfig
from .encoders import build_encoder
from .features import build_feature_dataframe, trajectory_to_text
from .loaders import load_trajectories
from .reporting import (
    plot_metrics_bar,
    plot_pathology_distribution,
    plot_scatter,
    write_outputs,
    write_run_manifest,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output_dir: Path
    metrics: dict[str, Any]
    artifact_paths: dict[str, Path]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_text_embeddings(
    records, encoder_config: dict[str, Any], max_text_chars: int
) -> np.ndarray | None:
    if encoder_config.get("type", "none") == "none":
        return None
    encoder = build_encoder(encoder_config)
    texts = [trajectory_to_text(r, max_chars=max_text_chars) for r in records]
    try:
        return encoder.fit_encode(texts)
    except (RuntimeError, NotImplementedError) as exc:
        logger.warning("Encoder failed (%s); pipeline will degrade.", exc)
        return None


def run_pipeline(config: ExperimentConfig) -> PipelineResult:
    _set_seed(config.seed)

    input_dir = Path(config.data.get("input_dir", "./data/external/aftraj"))
    dataset_type = str(config.data.get("dataset_type", "legacy_flat_json"))
    filter_by = dict(config.data.get("filter_by") or {})
    loader_kwargs = {
        k: v
        for k, v in config.data.items()
        if k not in {"input_dir", "dataset_type", "max_text_chars", "filter_by"}
    }
    records = load_trajectories(input_dir, dataset_type=dataset_type, **loader_kwargs)
    if filter_by:
        before = len(records)
        records = [
            r
            for r in records
            if all(r.labels.get(k) == v for k, v in filter_by.items())
        ]
        logger.info(
            "Applied filter_by=%s: %d -> %d records", filter_by, before, len(records)
        )
    if not records:
        raise RuntimeError(f"No trajectories found under {input_dir}")

    features_df, feature_columns = build_feature_dataframe(
        records, enabled=config.features
    )

    text_embeddings = _load_text_embeddings(
        records,
        config.encoder,
        max_text_chars=int(config.data.get("max_text_chars", 6000)),
    )

    cluster_result = run_clustering(
        features_df=features_df,
        feature_columns=feature_columns,
        text_embeddings=text_embeddings,
        cluster_config=config.clustering,
        random_state=config.seed,
    )

    enriched_df, summary_df, csv_paths = write_outputs(
        features_df=features_df,
        cluster_labels=cluster_result.labels,
        coords_2d=cluster_result.coords_2d,
        output_dir=config.output_dir,
        rule_set=str(config.analysis.get("pathology_rule_set", "default")),
    )

    correlation_df = cluster_outcome_correlation(
        enriched_df,
        targets=list(config.analysis.get("correlation_targets", [])),
    )
    correlation_csv = config.output_dir / "cluster_correlation.csv"
    correlation_df.to_csv(correlation_csv, index=False)

    artifact_paths: dict[str, Path] = dict(csv_paths)
    artifact_paths["cluster_correlation"] = correlation_csv

    label_columns = list(config.analysis.get("label_columns", []))
    if label_columns:
        validation_df = evaluate_label_columns(enriched_df, label_columns)
        validation_csv = config.output_dir / "cluster_label_validation.csv"
        validation_df.to_csv(validation_csv, index=False)
        artifact_paths["cluster_label_validation"] = validation_csv

    # Stratified validation: condition on a domain/task variable so we can
    # ask whether label alignment survives after the confound is held fixed.
    stratify_by = config.analysis.get("stratify_by")
    if label_columns and stratify_by:
        stratified_df = evaluate_within_groups(
            enriched_df,
            label_columns=[c for c in label_columns if c != stratify_by],
            group_column=str(stratify_by),
            min_group_size=int(config.analysis.get("stratify_min_group_size", 30)),
        )
        if not stratified_df.empty:
            stratified_csv = (
                config.output_dir / "cluster_label_validation_stratified.csv"
            )
            stratified_df.to_csv(stratified_csv, index=False)
            artifact_paths["cluster_label_validation_stratified"] = stratified_csv

    # Per-tag multilabel validation: list-valued labels (failure_types, ...)
    # are unfair to evaluate as a single category; expand to binary tags.
    multilabel_columns = list(config.analysis.get("multilabel_columns", []))
    if multilabel_columns:
        multilabel_df = evaluate_multilabel(enriched_df, multilabel_columns)
        if not multilabel_df.empty:
            multilabel_csv = config.output_dir / "cluster_multilabel_validation.csv"
            multilabel_df.to_csv(multilabel_csv, index=False)
            artifact_paths["cluster_multilabel_validation"] = multilabel_csv

    # Baseline comparison: trivial heuristics (single-feature quantile
    # binning, random labels, oracle on the grouping variable) scored
    # against the same label columns as the main result.
    baseline_specs = list(config.analysis.get("baselines", []))
    if baseline_specs and label_columns:
        baseline_df = compare_baselines(
            enriched_df,
            label_columns=label_columns,
            baseline_specs=baseline_specs,
            k=cluster_result.n_clusters,
            seed=config.seed,
        )
        if not baseline_df.empty:
            baseline_csv = config.output_dir / "cluster_baseline_comparison.csv"
            baseline_df.to_csv(baseline_csv, index=False)
            artifact_paths["cluster_baseline_comparison"] = baseline_csv

    # K-sensitivity sweep: check that the picked K is not the only K where
    # cluster ↔ label NMI looks reasonable.
    k_sweep_values = list(config.analysis.get("k_sensitivity", []))
    if k_sweep_values and label_columns:
        sweep_df = k_sensitivity_sweep(
            cluster_result.matrix,
            enriched_df,
            k_values=[int(k) for k in k_sweep_values],
            label_columns=label_columns,
            random_state=config.seed,
        )
        if not sweep_df.empty:
            sweep_csv = config.output_dir / "k_sensitivity.csv"
            sweep_df.to_csv(sweep_csv, index=False)
            artifact_paths["k_sensitivity"] = sweep_csv

    if config.reporting.get("scatter", True):
        path = config.output_dir / "cluster_scatter.png"
        plot_scatter(enriched_df, path)
        artifact_paths["cluster_scatter"] = path
    if config.reporting.get("metrics_bar", True):
        path = config.output_dir / "cluster_metrics.png"
        plot_metrics_bar(summary_df, path)
        artifact_paths["cluster_metrics"] = path
    if config.reporting.get("pathology_distribution", True):
        path = config.output_dir / "pathology_distribution.png"
        plot_pathology_distribution(summary_df, path)
        artifact_paths["pathology_distribution"] = path

    metrics = {
        "n_trajectories": int(len(features_df)),
        "n_clusters": cluster_result.n_clusters,
        "silhouette": cluster_result.silhouette,
        "calinski_harabasz": cluster_result.calinski_harabasz,
        "mode": cluster_result.mode,
        "clusterer": cluster_result.clusterer_name,
        "encoder": config.encoder.get("type", "none"),
        "feature_dim": len(feature_columns),
        "text_dim": int(text_embeddings.shape[1]) if text_embeddings is not None else 0,
        "extras": cluster_result.extras,
    }

    if config.reporting.get("manifest", True):
        manifest_path = write_run_manifest(
            config, config.output_dir, metrics, artifact_paths
        )
        artifact_paths["run_manifest"] = manifest_path

    return PipelineResult(
        output_dir=config.output_dir,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
