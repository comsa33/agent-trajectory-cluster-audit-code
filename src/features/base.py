"""Base classes / utilities shared across feature extractors."""
from __future__ import annotations

import json
from typing import Any, Protocol

from ..loaders import TrajectoryRecord


class FeatureExtractor(Protocol):
    """Pluggable feature extractor interface.

    `numeric_columns` declares the columns that should participate in the
    clustering feature matrix. `extract` returns one row of values keyed by
    those (and optionally additional metadata) columns.
    """

    numeric_columns: list[str]

    def extract(self, record: TrajectoryRecord) -> dict[str, Any]: ...


def is_empty_observation(obs: Any) -> bool:
    if obs is None:
        return True
    if isinstance(obs, (list, tuple, dict, str)) and len(obs) == 0:
        return True
    if isinstance(obs, str) and obs.strip() in {"", "---", "null", "None"}:
        return True
    return False


def observation_length(obs: Any) -> int:
    if obs is None:
        return 0
    if isinstance(obs, str):
        return len(obs)
    try:
        return len(json.dumps(obs, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(obs))


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def extract_keyword(tool_args: Any) -> str | None:
    # Pull a "keyword"/"query"/"search"/"term" out of tool args if present.
    if not isinstance(tool_args, dict):
        return None
    for key in ("keyword", "query", "search", "term"):
        value = tool_args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None
