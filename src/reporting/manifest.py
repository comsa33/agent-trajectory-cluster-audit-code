"""Run manifest emitting JSON + YAML side-by-side for reproducibility."""
from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig

logger = logging.getLogger(__name__)


def write_run_manifest(
    config: ExperimentConfig,
    output_dir: Path,
    metrics: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> Path:
    # Write a manifest capturing the resolved config, library versions,
    # and headline metrics so a result directory is self-describing.
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": config.name,
        "config_hash": config.config_hash,
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": config.seed,
        "config": config.raw,
        "metrics": metrics,
        "artifacts": {k: str(v) for k, v in artifact_paths.items()},
    }
    json_path = output_dir / "run_manifest.json"
    yaml_path = output_dir / "run_manifest.yaml"
    json_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    logger.info("Wrote run manifest -> %s", json_path)
    return json_path
