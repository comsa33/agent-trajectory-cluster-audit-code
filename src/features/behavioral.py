"""Behavioral / cognitive-level features: exploration vs fixation."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from ..loaders import TrajectoryRecord
from .base import extract_keyword

# Capitalized phrases as a cheap proxy for named entities / hypotheses.
_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


class BehavioralFeatures:
    # Higher-level behavioral signals derived from thought + tool stream.

    numeric_columns: list[str] = [
        "exploration_index",
        "hypothesis_persistence",
        "reasoning_depth_growth",
        "finalization_lateness",
        "fixation_score",
        "search_keyword_diversity",
    ]

    def extract(self, record: TrajectoryRecord) -> dict[str, Any]:
        steps = record.steps
        n_steps = len(steps)
        tool_names = [s.tool_name for s in steps if isinstance(s.tool_name, str)]

        # Exploration: unique tools used / total tool calls. Higher = broader.
        exploration_index = (
            len(set(tool_names)) / len(tool_names) if tool_names else 0.0
        )

        # Hypothesis persistence: max repeat count of any candidate entity in
        # thought texts. Captures the "agent keeps coming back to X" pattern.
        thought_blob = " ".join(
            s.thought for s in steps if isinstance(s.thought, str)
        )
        entities = _ENTITY_PATTERN.findall(thought_blob)
        entity_counter = Counter(entities)
        hypothesis_persistence = (
            max(entity_counter.values()) if entity_counter else 0
        )

        # Reasoning depth growth: linear slope of thought length over step idx.
        reasoning_depth_growth = _length_slope(
            [len(s.thought) if isinstance(s.thought, str) else 0 for s in steps]
        )

        # Finalization lateness: where did the last tool call land relative to
        # total iterations? Late finalization on incorrect answers correlates
        # with over-reasoning rather than premature finalization.
        finalization_lateness = (
            (steps[-1].index + 1) / n_steps if n_steps else 0.0
        )

        # Fixation: keyword-repetition + entity-repetition combined.
        keywords = [k for k in (extract_keyword(s.tool_args) for s in steps) if k]
        if keywords:
            kw_counter = Counter(keywords)
            kw_repeat_ratio = (
                sum(c for c in kw_counter.values() if c > 1) / len(keywords)
            )
            search_keyword_diversity = len(kw_counter) / len(keywords)
        else:
            kw_repeat_ratio = 0.0
            search_keyword_diversity = 0.0

        entity_repeat_ratio = (
            sum(c for c in entity_counter.values() if c > 1) / max(len(entities), 1)
            if entities
            else 0.0
        )
        fixation_score = float(
            0.5 * kw_repeat_ratio + 0.5 * entity_repeat_ratio
        )

        return {
            "exploration_index": float(exploration_index),
            "hypothesis_persistence": int(hypothesis_persistence),
            "reasoning_depth_growth": float(reasoning_depth_growth),
            "finalization_lateness": float(finalization_lateness),
            "fixation_score": fixation_score,
            "search_keyword_diversity": float(search_keyword_diversity),
        }


def _length_slope(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    xs = np.arange(len(arr), dtype=np.float64)
    return float(np.polyfit(xs, arr, 1)[0])
