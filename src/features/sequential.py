"""Sequence-aware features: tool transitions, retry bursts, drift slopes."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from ..loaders import TrajectoryRecord
from .base import extract_keyword, is_empty_observation


def _shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array(counts, dtype=np.float64) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _max_run_length(flags: list[bool]) -> int:
    best = current = 0
    for f in flags:
        current = current + 1 if f else 0
        best = max(best, current)
    return best


def _argument_signature(tool_name: str | None, args: Any) -> str:
    # Hashable canonical form for retry-burst detection.
    try:
        args_str = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        args_str = str(args)
    return f"{tool_name}::{args_str}"


class SequentialFeatures:
    # Derived from the ordering of steps, not just their counts.

    numeric_columns: list[str] = [
        "tool_transition_entropy",
        "tool_bigram_diversity",
        "empty_observation_burst",
        "retry_burst_max",
        "retry_burst_ratio",
        "thought_drift_slope",
        "tool_path_length_norm",
    ]

    def extract(self, record: TrajectoryRecord) -> dict[str, Any]:
        steps = record.steps
        tool_seq = [s.tool_name for s in steps if isinstance(s.tool_name, str) and s.tool_name]

        # Tool transition entropy over the empirical bigram distribution.
        bigram_counts: Counter[tuple[str, str]] = Counter(
            zip(tool_seq, tool_seq[1:])
        )
        tool_transition_entropy = _shannon_entropy(list(bigram_counts.values()))

        if bigram_counts:
            tool_bigram_diversity = len(bigram_counts) / sum(bigram_counts.values())
        else:
            tool_bigram_diversity = 0.0

        empty_flags = [is_empty_observation(s.observation) for s in steps]
        empty_observation_burst = _max_run_length(empty_flags)

        # Retry-burst: identical (tool_name, tool_args) signatures repeated.
        signatures = [
            _argument_signature(s.tool_name, s.tool_args) for s in steps
        ]
        sig_counter = Counter(signatures)
        retry_burst_max = max(sig_counter.values()) if sig_counter else 0
        repeated_sig_count = sum(c - 1 for c in sig_counter.values() if c > 1)
        retry_burst_ratio = (
            repeated_sig_count / len(signatures) if signatures else 0.0
        )

        # Thought drift slope: linear trend of consecutive thought self-similarity.
        thought_drift_slope = _thought_drift_slope(
            [s.thought for s in steps if isinstance(s.thought, str)]
        )

        tool_path_length_norm = (
            len(tool_seq) / max(len(steps), 1) if steps else 0.0
        )

        return {
            "tool_transition_entropy": float(tool_transition_entropy),
            "tool_bigram_diversity": float(tool_bigram_diversity),
            "empty_observation_burst": int(empty_observation_burst),
            "retry_burst_max": int(retry_burst_max),
            "retry_burst_ratio": float(retry_burst_ratio),
            "thought_drift_slope": float(thought_drift_slope),
            "tool_path_length_norm": float(tool_path_length_norm),
        }


def _thought_drift_slope(thoughts: list[str]) -> float:
    # Slope of cosine similarity between consecutive thought BoW vectors.
    # Negative slope -> thoughts diverge over time (exploration);
    # positive slope -> thoughts converge (potential fixation).
    if len(thoughts) < 3:
        return 0.0

    vocab: dict[str, int] = {}
    token_lists: list[list[str]] = []
    for t in thoughts:
        tokens = [w.lower() for w in t.split() if w.isalpha()]
        token_lists.append(tokens)
        for w in tokens:
            vocab.setdefault(w, len(vocab))
    if not vocab:
        return 0.0

    matrix = np.zeros((len(token_lists), len(vocab)), dtype=np.float32)
    for r, tokens in enumerate(token_lists):
        if not tokens:
            continue
        c = Counter(tokens)
        for w, n in c.items():
            matrix[r, vocab[w]] = n
    norms = np.linalg.norm(matrix, axis=1)
    sims: list[float] = []
    for i in range(len(token_lists) - 1):
        a, b = norms[i], norms[i + 1]
        if a == 0 or b == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(matrix[i], matrix[i + 1]) / (a * b)))
    if len(sims) < 2:
        return 0.0
    xs = np.arange(len(sims), dtype=np.float64)
    slope = np.polyfit(xs, np.array(sims, dtype=np.float64), 1)[0]
    return float(slope)
