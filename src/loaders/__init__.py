"""Trajectory loader package — dataset adapters dispatched by name.

Backward-compat shim: legacy callers import `load_trajectories`,
`TrajectoryRecord`, `TrajectoryStep`, `iter_trajectory_paths`, and
`load_trajectory` from `src.loaders`. Those symbols remain exposed at the
package level. New code passes `dataset_type=` to dispatch to a specific
adapter; the default is the legacy flat-JSON adapter that the in-tree
sample fixture uses.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .base import TrajectoryRecord, TrajectoryStep
from .legacy_flat_json import (
    iter_trajectory_paths,
    load_legacy_flat_json,
    load_trajectory,
)

logger = logging.getLogger(__name__)

# Adapter registry. Each adapter takes (input_dir: Path, **kwargs) -> list[TrajectoryRecord].
DATASET_LOADERS: dict[str, Callable[..., list[TrajectoryRecord]]] = {
    "legacy_flat_json": load_legacy_flat_json,
}

# Aliases preserve historical config strings without polluting the canonical
# name set. Add to `_DATASET_ALIASES` when renaming an adapter.
_DATASET_ALIASES: dict[str, str] = {
    "hgc": "legacy_flat_json",
}


def register_dataset_loader(
    name: str, loader: Callable[..., list[TrajectoryRecord]]
) -> None:
    if name in DATASET_LOADERS:
        raise ValueError(f"Dataset loader already registered: {name}")
    DATASET_LOADERS[name] = loader


def list_dataset_loaders() -> list[str]:
    return sorted(DATASET_LOADERS)


def _resolve(name: str) -> str:
    return _DATASET_ALIASES.get(name, name)


def load_trajectories(
    input_dir: Path | str,
    dataset_type: str = "legacy_flat_json",
    **kwargs: Any,
) -> list[TrajectoryRecord]:
    """Dispatch to the requested adapter.

    Backward compatible: prior callers passed only `input_dir` (and optionally
    `include_dirs` / `exclude_dirs`); the default `dataset_type` and the
    `_DATASET_ALIASES` table preserve their behavior.
    """
    resolved = _resolve(dataset_type)
    if resolved not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset_type {dataset_type!r}. "
            f"Available: {list_dataset_loaders()}"
        )
    loader = DATASET_LOADERS[resolved]
    return loader(Path(input_dir), **kwargs)


# Late imports below register additional adapters without forming circular
# dependencies with the package __init__.
try:
    from . import aftraj as _aftraj  # noqa: F401  -- side-effect: registration
except ImportError as exc:
    logger.debug("AFTraj loader not registered: %s", exc)

try:
    from . import agenterrorbench as _aeb  # noqa: F401  -- side-effect: registration
except ImportError as exc:
    logger.debug("AgentErrorBench loader not registered: %s", exc)

try:
    from . import agentrx as _agentrx  # noqa: F401  -- side-effect: registration
except ImportError as exc:
    logger.debug("AgentRx loader not registered: %s", exc)


__all__ = [
    "TrajectoryRecord",
    "TrajectoryStep",
    "iter_trajectory_paths",
    "load_trajectory",
    "load_trajectories",
    "register_dataset_loader",
    "list_dataset_loaders",
    "DATASET_LOADERS",
]
