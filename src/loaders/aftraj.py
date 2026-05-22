"""AFTraj-2K adapter (HF: ZBox008003/AFTraj).

Schema per row of `aftraj_safe.parquet` / `aftraj_unsafe.parquet`:
    conv_id, domain, task, gold_answer, num_turns, tools, turns,
    mistake_step, mistake_agent, mistake_reason, unsafe_source

`turns` is a list of `{role, action, content, thought}`. Roles include
`user`, `environment`, and arbitrary multi-agent names ("MathSolver",
"Verifier", "CodeWriter", ...). Actions are JSON-encoded tool-call lists,
e.g. `[{"name": "compute", "arguments": "..."}]`. Observations come from
following `environment` turns.

Each non-environment, non-user turn becomes a TrajectoryStep. The
following `environment` turn(s) (until the next agent turn) are joined as
the observation for that step.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import register_dataset_loader
from .base import TrajectoryRecord, TrajectoryStep

logger = logging.getLogger(__name__)


def _parse_action(action_raw: Any) -> tuple[str | None, dict[str, Any] | None]:
    # AFTraj actions are JSON-encoded tool-call lists. We collapse the first
    # call into (tool_name, tool_args). Multi-call rows are rare; if we see
    # more we still report only the first to keep the canonical step shape.
    if not action_raw:
        return None, None
    if not isinstance(action_raw, str):
        return None, None
    text = action_raw.strip()
    if not text:
        return None, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:80], None

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        first = parsed[0]
        name = first.get("name")
        args_raw = first.get("arguments")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = None
        return (name if isinstance(name, str) else None), args

    if isinstance(parsed, dict):
        return parsed.get("name"), parsed.get("arguments")
    return None, None


def _turns_to_steps(turns: list[dict[str, Any]]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    i = 0
    n = len(turns)
    step_idx = 0
    while i < n:
        turn = turns[i] if isinstance(turns[i], dict) else {}
        role = (turn.get("role") or "").strip()
        if role in {"user", "environment", ""}:
            i += 1
            continue

        thought = turn.get("thought") or ""
        content = turn.get("content") or ""
        tool_name, tool_args = _parse_action(turn.get("action"))

        # Walk forward, gathering environment observations until the next
        # agent / user turn.
        observation_parts: list[str] = []
        if not tool_name and content:
            # Plain-text agent output behaves as its own observation surface.
            observation_parts.append(content)
        j = i + 1
        while j < n:
            env = turns[j] if isinstance(turns[j], dict) else {}
            if (env.get("role") or "") != "environment":
                break
            env_content = env.get("content")
            if isinstance(env_content, str) and env_content.strip():
                observation_parts.append(env_content)
            j += 1

        observation: Any
        if observation_parts:
            observation = "\n".join(observation_parts)
        else:
            observation = "" if tool_name else None

        steps.append(
            TrajectoryStep(
                index=step_idx,
                thought=thought if isinstance(thought, str) else None,
                # Fall back to the agent role name when the row is a plain
                # message (no tool call). This preserves "who acted" so our
                # tool_transition_entropy / unique_tool_count features stay
                # meaningful for the multi-agent case.
                tool_name=tool_name or role or None,
                tool_args=tool_args,
                observation=observation,
            )
        )
        step_idx += 1
        i = j

    return steps


def _row_to_record(
    row: pd.Series, source_path: Path, split: str
) -> TrajectoryRecord:
    turns_raw = row.get("turns")
    if hasattr(turns_raw, "tolist"):
        turns = list(turns_raw.tolist())
    else:
        turns = list(turns_raw or [])
    steps = _turns_to_steps(turns)

    mistake_step = row.get("mistake_step")
    mistake_agent = row.get("mistake_agent")
    unsafe_source = row.get("unsafe_source")
    domain = row.get("domain")

    is_safe = split == "safe"

    labels: dict[str, Any] = {
        "safe_unsafe": "safe" if is_safe else "unsafe",
        "domain": domain if isinstance(domain, str) else None,
        "mistake_agent": mistake_agent if isinstance(mistake_agent, str) and mistake_agent else None,
        "unsafe_source": unsafe_source if isinstance(unsafe_source, str) and unsafe_source else None,
        "mistake_step": int(mistake_step) if pd.notna(mistake_step) else None,
    }

    return TrajectoryRecord(
        query_id=str(row.get("conv_id", "")),
        question=str(row.get("task", "")) or None,
        gold=str(row.get("gold_answer", "")) or None,
        pred=None,  # AFTraj does not expose a single final-answer field
        judgment_correct=is_safe,
        tokens=0,
        time_s=0.0,
        n_iters=int(row.get("num_turns", len(turns)) or len(turns)),
        n_retrieved_positive=0,
        n_retrieved_negative=0,
        n_added_hints=0,
        phase=split,
        source_dir=f"aftraj_{split}",
        source_path=str(source_path),
        steps=steps,
        labels=labels,
        raw={},  # avoid materializing the full payload; labels carry what we need
    )


def _read_split(path: Path, split: str) -> list[TrajectoryRecord]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    records = [_row_to_record(df.iloc[i], path, split) for i in range(len(df))]
    logger.info("AFTraj %s split: %d records from %s", split, len(records), path)
    return records


def load_aftraj(
    input_dir: Path,
    splits: list[str] | tuple[str, ...] = ("safe", "unsafe"),
    **_: object,
) -> list[TrajectoryRecord]:
    # Bulk-load AFTraj parquet splits. `splits` selects a subset.
    if not input_dir.exists():
        raise FileNotFoundError(f"AFTraj input directory not found: {input_dir}")

    records: list[TrajectoryRecord] = []
    for split in splits:
        path = input_dir / f"aftraj_{split}.parquet"
        records.extend(_read_split(path, split))
    logger.info("Loaded %d AFTraj trajectories from %s (splits=%s)", len(records), input_dir, list(splits))
    return records


register_dataset_loader("aftraj", load_aftraj)
