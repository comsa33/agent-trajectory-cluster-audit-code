"""Aggregate structural features (counts, ratios, summary statistics)."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..loaders import TrajectoryRecord
from .base import (
    extract_keyword,
    is_empty_observation,
    observation_length,
)


class StructuralFeatures:
    # Aggregate scalar features over the full trajectory.

    numeric_columns: list[str] = [
        "n_iters",
        "tokens",
        "time_s",
        "total_tool_calls",
        "unique_tool_count",
        "empty_observation_ratio",
        "avg_observation_length",
        "avg_thought_length",
        "repeated_tool_ratio",
        "repeated_search_keyword_ratio",
        "hint_count",
        "retrieval_positive_count",
        "retrieval_negative_count",
    ]

    def extract(self, record: TrajectoryRecord) -> dict[str, Any]:
        steps = record.steps
        tool_names = [s.tool_name for s in steps if isinstance(s.tool_name, str) and s.tool_name]
        keywords = [k for k in (extract_keyword(s.tool_args) for s in steps) if k]
        thoughts = [s.thought for s in steps if isinstance(s.thought, str)]
        empty_flags = [is_empty_observation(s.observation) for s in steps]
        observation_lens = [observation_length(s.observation) for s in steps]

        total_tool_calls = len(tool_names)
        if total_tool_calls > 0:
            tool_counter = Counter(tool_names)
            repeated = sum(c for c in tool_counter.values() if c > 1)
            repeated_tool_ratio = repeated / total_tool_calls
        else:
            tool_counter = Counter()
            repeated_tool_ratio = 0.0

        if keywords:
            kw_counter = Counter(keywords)
            repeated_kw = sum(c for c in kw_counter.values() if c > 1)
            repeated_search_keyword_ratio = repeated_kw / len(keywords)
        else:
            repeated_search_keyword_ratio = 0.0

        return {
            "n_iters": int(record.n_iters or len(steps)),
            "tokens": int(record.tokens),
            "time_s": float(record.time_s),
            "total_tool_calls": total_tool_calls,
            "unique_tool_count": len(set(tool_names)),
            "empty_observation_ratio": float(np.mean(empty_flags)) if empty_flags else 0.0,
            "avg_observation_length": float(np.mean(observation_lens)) if observation_lens else 0.0,
            "avg_thought_length": float(np.mean([len(t) for t in thoughts])) if thoughts else 0.0,
            "repeated_tool_ratio": float(repeated_tool_ratio),
            "repeated_search_keyword_ratio": float(repeated_search_keyword_ratio),
            "hint_count": int(record.n_added_hints),
            "retrieval_positive_count": int(record.n_retrieved_positive),
            "retrieval_negative_count": int(record.n_retrieved_negative),
            "top_tool_counter": dict(tool_counter),
        }
