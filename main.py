"""Config-driven CLI for the trajectory failure clustering framework."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import load_config
from src.pipeline import run_pipeline

logger = logging.getLogger("trajectory_clustering")

_DEFAULT_BASE = Path("configs/default.yaml")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Behavioral representation learning + failure-mode clustering "
            "framework for LLM agent trajectories."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_BASE,
        help="Experiment YAML (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_BASE,
        help="Base config that the experiment YAML overrides.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Override data.input_dir without editing the YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override experiment.output_dir.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Override clustering.n_clusters.",
    )
    parser.add_argument(
        "--mode",
        choices=("feature", "text", "hybrid"),
        default=None,
        help="Override clustering.mode.",
    )
    parser.add_argument(
        "--encoder",
        choices=("none", "tfidf", "sentence_transformer", "sequence"),
        default=None,
        help="Override encoder.type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment.seed.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    experiment: dict[str, Any] = {}
    if args.output_dir is not None:
        experiment["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        experiment["seed"] = args.seed
    if experiment:
        overrides["experiment"] = experiment

    if args.input_dir is not None:
        overrides["data"] = {"input_dir": str(args.input_dir)}

    cluster: dict[str, Any] = {}
    if args.n_clusters is not None:
        cluster["n_clusters"] = args.n_clusters
    if args.mode is not None:
        cluster["mode"] = args.mode
    if cluster:
        overrides["clustering"] = cluster

    if args.encoder is not None:
        overrides["encoder"] = {"type": args.encoder}

    return overrides


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(args.log_level)

    overrides = _build_overrides(args)
    base_path = args.base if args.base.exists() else None
    config = load_config(args.config, base_path=base_path, overrides=overrides)

    try:
        result = run_pipeline(config)
    except RuntimeError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 2

    logger.info(
        "Experiment %s finished. Output: %s", config.name, result.output_dir
    )
    for name, path in result.artifact_paths.items():
        logger.info("  %s -> %s", name, path)
    if result.metrics.get("silhouette") is not None:
        logger.info(
            "silhouette=%.4f calinski=%s n_clusters=%d",
            result.metrics["silhouette"],
            f"{result.metrics['calinski_harabasz']:.2f}"
            if result.metrics["calinski_harabasz"] is not None
            else "n/a",
            result.metrics["n_clusters"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
