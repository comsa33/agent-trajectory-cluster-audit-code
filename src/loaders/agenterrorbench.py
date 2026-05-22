"""AgentErrorBench adapter (HF: davide221/agenterrorbench).

Schema per parquet row:
    trajectory_id, task_type, llm_model, critical_failure_step,
    critical_failure_module, failure_types, failure_reasonings,
    failure_modules, num_steps, trajectory_length, full_trajectory,
    step_annotations

`full_trajectory` is a JSON-encoded string carrying
`{"messages": [...], "metadata": {...}}`. Messages alternate
`role=user` (env observation) and `role=assistant` (agent step). Each
assistant message contains a `<plan>...</plan>` block (thought) followed
by an `<action>...</action>` block whose body is either a plain command
(alfworld), a structured tool/parameters block (gaia), or a
`command[args]` form (webshop). We normalize all three into the canonical
(thought, tool_name, tool_args, observation) step shape.

`step_annotations` is a JSON-encoded list of per-step failure records
mapped by 1-indexed step number; the dataset only contains failed
trajectories, so every record has at least one annotation. We expose
the labels needed for cluster validation through `TrajectoryRecord.labels`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import register_dataset_loader
from .base import TrajectoryRecord, TrajectoryStep

logger = logging.getLogger(__name__)

_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
# gaia-style action body: "tool: name\nparameters: {...}"
_GAIA_TOOL_RE = re.compile(
    r"tool\s*:\s*([\w_.\-]+)\s*\nparameters\s*:\s*(\{.*\})",
    re.DOTALL | re.IGNORECASE,
)
# webshop-style: "command[args...]"
_WEBSHOP_RE = re.compile(r"^\s*([a-zA-Z_][\w]*)\s*\[(.*)\]\s*$", re.DOTALL)


def _parse_action_body(body: str) -> tuple[str | None, dict[str, Any] | None]:
    body = body.strip()
    if not body:
        return None, None

    gaia = _GAIA_TOOL_RE.search(body)
    if gaia:
        name = gaia.group(1)
        try:
            args = json.loads(gaia.group(2))
            if not isinstance(args, dict):
                args = {"_raw": args}
        except json.JSONDecodeError:
            args = {"_raw": gaia.group(2).strip()}
        return name, args

    web = _WEBSHOP_RE.match(body)
    if web:
        return web.group(1), {"argument": web.group(2).strip()}

    # Default: alfworld-style "verb object ..." — first token is the verb,
    # the rest is the argument.
    parts = body.split(None, 1)
    if not parts:
        return None, None
    name = parts[0]
    args = {"argument": parts[1].strip()} if len(parts) > 1 else None
    return name, args


def _extract_step(content: str) -> tuple[str | None, str | None, dict[str, Any] | None]:
    plan = _PLAN_RE.search(content)
    action = _ACTION_RE.search(content)
    thought = plan.group(1).strip() if plan else None
    if action:
        tool_name, tool_args = _parse_action_body(action.group(1))
    else:
        # Some assistant turns may be pure final-answer / reflection text.
        tool_name, tool_args = None, None
        if thought is None:
            thought = content.strip()
    return thought, tool_name, tool_args


def _messages_to_steps(messages: list[dict[str, Any]]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    pending_thought: str | None = None
    pending_tool_name: str | None = None
    pending_tool_args: dict[str, Any] | None = None
    pending = False
    step_idx = 1  # 1-indexed to align with step_annotations

    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if role == "assistant":
            if pending:
                # Previous assistant turn produced no observation before the
                # next assistant turn — emit it with empty observation.
                steps.append(
                    TrajectoryStep(
                        index=step_idx,
                        thought=pending_thought,
                        tool_name=pending_tool_name,
                        tool_args=pending_tool_args,
                        observation="",
                    )
                )
                step_idx += 1
            pending_thought, pending_tool_name, pending_tool_args = _extract_step(content)
            pending = True
        elif role == "user" and pending:
            steps.append(
                TrajectoryStep(
                    index=step_idx,
                    thought=pending_thought,
                    tool_name=pending_tool_name,
                    tool_args=pending_tool_args,
                    observation=content,
                )
            )
            step_idx += 1
            pending = False

    if pending:
        steps.append(
            TrajectoryStep(
                index=step_idx,
                thought=pending_thought,
                tool_name=pending_tool_name,
                tool_args=pending_tool_args,
                observation="",
            )
        )
    return steps


def _safe_first(seq: Any) -> Any:
    if seq is None:
        return None
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    if isinstance(seq, (list, tuple)) and seq:
        return seq[0]
    return None


def _row_to_record(row: pd.Series, source_path: Path) -> TrajectoryRecord | None:
    raw_traj = row.get("full_trajectory")
    if not isinstance(raw_traj, str):
        return None
    try:
        payload = json.loads(raw_traj)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse full_trajectory for %s: %s", row.get("trajectory_id"), exc)
        return None

    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        return None

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    steps = _messages_to_steps(messages)

    # First user message is the task statement in all three task families.
    question = None
    for m in messages:
        if (m.get("role") or "").lower() == "user" and isinstance(m.get("content"), str):
            question = m["content"]
            break

    failure_types = list(row.get("failure_types") or [])
    failure_modules = list(row.get("failure_modules") or [])

    labels: dict[str, Any] = {
        "task_type": row.get("task_type"),
        "llm_model": row.get("llm_model"),
        "critical_failure_module": row.get("critical_failure_module"),
        "critical_failure_step": (
            int(row.get("critical_failure_step"))
            if pd.notna(row.get("critical_failure_step"))
            else None
        ),
        "failure_type_first": _safe_first(failure_types),
        "failure_module_first": _safe_first(failure_modules),
        # Full lists preserved so multilabel validation can do per-tag binary
        # purity / NMI rather than only treating the first tag as the label.
        "failure_types_list": failure_types,
        "failure_modules_list": failure_modules,
        "n_failure_types": int(len(failure_types)),
        "won": bool(metadata.get("won", False)),
    }

    return TrajectoryRecord(
        query_id=str(row.get("trajectory_id", "")),
        question=question,
        gold=None,
        pred=None,
        judgment_correct=bool(metadata.get("won", False)),
        tokens=0,
        time_s=0.0,
        n_iters=int(row.get("num_steps", len(steps)) or len(steps)),
        n_retrieved_positive=0,
        n_retrieved_negative=0,
        n_added_hints=0,
        phase=str(row.get("task_type", "")),
        source_dir=f"agenterrorbench_{row.get('task_type','unknown')}",
        source_path=str(source_path),
        steps=steps,
        labels=labels,
        raw={},
    )


def _read_parquet(path: Path) -> list[TrajectoryRecord]:
    df = pd.read_parquet(path)
    records = [_row_to_record(df.iloc[i], path) for i in range(len(df))]
    records = [r for r in records if r is not None]
    logger.info("AgentErrorBench %s: %d records", path.name, len(records))
    return records


def load_agenterrorbench(
    input_dir: Path,
    splits: list[str] | tuple[str, ...] = ("train", "validation", "test"),
    **_: object,
) -> list[TrajectoryRecord]:
    # Load AgentErrorBench parquet splits from the HF snapshot layout:
    #   {input_dir}/data/{split}-00000-of-00001.parquet
    if not input_dir.exists():
        raise FileNotFoundError(f"AgentErrorBench input directory not found: {input_dir}")

    data_dir = input_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Expected parquet folder {data_dir} (HF snapshot uses data/ subdir)"
        )

    records: list[TrajectoryRecord] = []
    for split in splits:
        path = data_dir / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            logger.warning("AgentErrorBench split missing: %s", path)
            continue
        records.extend(_read_parquet(path))

    logger.info(
        "Loaded %d AgentErrorBench trajectories from %s (splits=%s)",
        len(records),
        input_dir,
        list(splits),
    )
    return records


register_dataset_loader("agenterrorbench", load_agenterrorbench)
