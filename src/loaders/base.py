"""Canonical TrajectoryStep / TrajectoryRecord shared by all dataset adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryStep:
    # Single agent step in the canonical (thought, tool_call, observation) shape.
    index: int
    thought: str | None
    tool_name: str | None
    tool_args: dict[str, Any] | None
    observation: Any


@dataclass
class TrajectoryRecord:
    # Normalized in-memory representation of one trajectory across any source.
    query_id: str
    question: str | None
    gold: str | None
    pred: str | None
    judgment_correct: bool
    tokens: int
    time_s: float
    n_iters: int
    n_retrieved_positive: int
    n_retrieved_negative: int
    n_added_hints: int
    phase: str
    source_dir: str
    source_path: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    # Dataset-specific labels (mistake_agent, failure_category, root_cause, ...).
    # Carried separately from features so they can act as ground truth in
    # cluster validation without leaking into the feature matrix.
    labels: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.steps)
