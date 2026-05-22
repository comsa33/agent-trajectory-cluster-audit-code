"""Trajectory -> text serialization for downstream encoders."""
from __future__ import annotations

from ..loaders import TrajectoryRecord
from .base import safe_str


def trajectory_to_text(record: TrajectoryRecord, max_chars: int = 6000) -> str:
    # Flatten a TrajectoryRecord into a structured text blob suitable for
    # sentence / TF-IDF / future sequence encoders.
    parts: list[str] = []
    if record.question:
        parts.append(f"QUESTION: {record.question.strip()}")

    for step in record.steps:
        thought = safe_str(step.thought)
        tool_name = safe_str(step.tool_name)
        tool_args = safe_str(step.tool_args)
        observation = safe_str(step.observation)
        if len(observation) > 400:
            observation = observation[:400] + "..."
        parts.append(
            f"STEP {step.index} | THOUGHT: {thought} "
            f"| TOOL: {tool_name}({tool_args}) | OBS: {observation}"
        )

    if record.pred:
        parts.append(f"FINAL_ANSWER: {record.pred.strip()}")

    text = "\n".join(parts)
    return text[:max_chars]
