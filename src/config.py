"""Experiment configuration loading and validation."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    # Top-level dataclass mirroring the YAML schema. Keep flexible — components
    # read what they need from `raw`.
    name: str
    seed: int
    output_dir: Path
    data: dict[str, Any]
    features: dict[str, Any]
    encoder: dict[str, Any]
    clustering: dict[str, Any]
    analysis: dict[str, Any]
    reporting: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        # Stable digest of the resolved config — used as a manifest field so
        # that results can be matched back to the exact configuration.
        canonical = json.dumps(self.raw, sort_keys=True, default=str)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _deep_merge(base: dict, override: dict) -> dict:
    # Recursive dict merge so experiment files only override what they change.
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping")
    return data


def load_config(
    config_path: Path,
    base_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    # Resolve base + experiment + CLI overrides into a single ExperimentConfig.
    base = _load_yaml(base_path) if base_path is not None else {}
    experiment = _load_yaml(config_path)
    merged = _deep_merge(base, experiment)
    if overrides:
        merged = _deep_merge(merged, overrides)

    experiment_block = merged.get("experiment", {})
    name = experiment_block.get("name", config_path.stem)
    seed = int(experiment_block.get("seed", 42))
    output_dir = Path(experiment_block.get("output_dir", f"./outputs/{name}"))

    config = ExperimentConfig(
        name=name,
        seed=seed,
        output_dir=output_dir,
        data=merged.get("data", {}),
        features=merged.get("features", {}),
        encoder=merged.get("encoder", {}),
        clustering=merged.get("clustering", {}),
        analysis=merged.get("analysis", {}),
        reporting=merged.get("reporting", {}),
        raw=merged,
    )
    logger.info(
        "Loaded experiment %s (hash=%s) from %s", config.name, config.config_hash, config_path
    )
    return config
