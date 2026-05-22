"""Adapter for the legacy flat-JSON trajectory format.

Schema: each `trajectory_*.json` file contains a top-level dict with
`query_id`, `question`, `gold`, `pred`, `judgment_correct`, `tokens`,
`time_s`, `n_iters`, `n_retrieved_positive`, `n_retrieved_negative`,
`n_added_hints`, `phase`, and a flat `trajectory` block whose keys are
`thought_N`, `tool_name_N`, `tool_args_N`, `observation_N` for step N.

This adapter exists to load small in-tree fixtures and to validate that
schema regressions do not break the pipeline. Public datasets (AFTraj,
AgentErrorBench, ...) live in their own adapters.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from .base import TrajectoryRecord, TrajectoryStep

logger = logging.getLogger(__name__)

# Sidecar files that sit next to trajectories in the legacy export layout.
_SKIP_FILENAMES = {"config.json", "smoke_summary.json"}


def iter_trajectory_paths(input_dir: Path) -> Iterable[Path]:
    # Walk the input directory and yield candidate JSON paths recursively.
    for path in sorted(input_dir.rglob("*.json")):
        if path.name in _SKIP_FILENAMES:
            continue
        yield path


def _reconstruct_steps(trajectory: dict[str, Any]) -> list[TrajectoryStep]:
    # The on-disk format flattens steps into thought_N / tool_name_N / ...
    if not isinstance(trajectory, dict):
        return []

    indices: set[int] = set()
    for key in trajectory:
        if "_" not in key:
            continue
        suffix = key.rsplit("_", 1)[-1]
        if suffix.isdigit():
            indices.add(int(suffix))

    steps: list[TrajectoryStep] = []
    for i in sorted(indices):
        steps.append(
            TrajectoryStep(
                index=i,
                thought=trajectory.get(f"thought_{i}"),
                tool_name=trajectory.get(f"tool_name_{i}"),
                tool_args=trajectory.get(f"tool_args_{i}"),
                observation=trajectory.get(f"observation_{i}"),
            )
        )
    return steps


def _to_record(payload: dict[str, Any], path: Path) -> TrajectoryRecord | None:
    if not isinstance(payload, dict) or "query_id" not in payload:
        return None
    if not isinstance(payload.get("trajectory"), dict):
        return None

    return TrajectoryRecord(
        query_id=str(payload.get("query_id", "")),
        question=payload.get("question"),
        gold=payload.get("gold"),
        pred=payload.get("pred"),
        judgment_correct=bool(payload.get("judgment_correct", False)),
        tokens=int(payload.get("tokens", 0) or 0),
        time_s=float(payload.get("time_s", 0.0) or 0.0),
        n_iters=int(payload.get("n_iters", 0) or 0),
        n_retrieved_positive=int(payload.get("n_retrieved_positive", 0) or 0),
        n_retrieved_negative=int(payload.get("n_retrieved_negative", 0) or 0),
        n_added_hints=int(payload.get("n_added_hints", 0) or 0),
        phase=str(payload.get("phase", "")),
        source_dir=path.parent.name,
        source_path=str(path),
        steps=_reconstruct_steps(payload.get("trajectory", {})),
        labels={
            "phase": str(payload.get("phase", "")),
            "source_dir": path.parent.name,
        },
        raw=payload,
    )


def load_trajectory(path: Path) -> TrajectoryRecord | None:
    # Read a single JSON file into a TrajectoryRecord (or None if invalid).
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse %s: %s", path, exc)
        return None
    return _to_record(payload, path)


def load_legacy_flat_json(
    input_dir: Path,
    include_dirs: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
    **_: object,
) -> list[TrajectoryRecord]:
    # Bulk-load legacy flat-JSON trajectories with optional dir filters.
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    include = set(include_dirs) if include_dirs else None
    exclude = set(exclude_dirs) if exclude_dirs else set()

    records: list[TrajectoryRecord] = []
    for path in iter_trajectory_paths(input_dir):
        parent = path.parent.name
        if include is not None and parent not in include:
            continue
        if parent in exclude:
            continue
        record = load_trajectory(path)
        if record is not None:
            records.append(record)

    logger.info(
        "Loaded %d legacy flat-JSON trajectories from %s", len(records), input_dir
    )
    return records
