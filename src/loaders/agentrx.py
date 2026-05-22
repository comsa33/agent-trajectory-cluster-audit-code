"""AgentRx adapter (HF: microsoft/AgentRx, gated).

The HF snapshot ships two splits, each as paired annotation + raw files:

    tau_retail.jsonl          # 29 annotations, id format: "2", "3", ...
    tau_retail_dataset.jsonl  # 29 raw traces, id format: "tau_retail_2", ...
    magentic_one.jsonl        # 44 annotations, uuid trajectory_id
    magentic_dataset.jsonl    # 58 raw traces, uuid trajectory_id

Records are emitted only for annotated trajectories (inner join on
normalized trajectory_id). Each TrajectoryRecord carries both the
observed-failures list (symptoms) and the single root_cause failure
pulled out separately so cluster-validation analyses can ask whether
clusters track symptoms or root causes.

Raw substep schemas differ between the two splits:

* tau_retail uses OpenAI-style roles (`system`/`user`/`assistant`/`tool`)
  where assistant content is JSON tool calls and `tool` substeps carry
  the result. We pair (assistant, following tool) into a single
  TrajectoryStep with tool_name / tool_args parsed from the JSON.
* magentic uses descriptive roles like `"Orchestrator (thought)"` /
  `"WebSurfer (action)"` / `"human"`. We treat the role string as the
  tool_name and attribute the content to thought or observation based
  on whether the suffix is `(thought)`.

Both paths skip pure user / system / human substeps.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import register_dataset_loader
from .base import TrajectoryRecord, TrajectoryStep

logger = logging.getLogger(__name__)

_USER_ROLES = {"human", "user", "system"}


def _normalize_tau_id(raw_id: str) -> str:
    prefix = "tau_retail_"
    return raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_tau_assistant_call(content: str) -> tuple[str | None, dict[str, Any] | None]:
    # tau assistant message is a JSON list with [{function: {name, arguments}}].
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
        return None, None
    fn = parsed[0].get("function", {})
    name = fn.get("name") if isinstance(fn, dict) else None
    args_raw = fn.get("arguments") if isinstance(fn, dict) else None
    args: dict[str, Any] | None = None
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
    elif isinstance(args_raw, dict):
        args = args_raw
    return (name if isinstance(name, str) else None), args


def _extract_steps_tau(raw_steps: list[dict[str, Any]]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    pending: dict[str, Any] | None = None
    idx = 1
    for step in raw_steps:
        for ss in step.get("substeps", []):
            role = (ss.get("role") or "").strip().lower()
            content = ss.get("content") or ""
            if role in _USER_ROLES:
                continue
            if role == "tool":
                if pending is not None:
                    steps.append(
                        TrajectoryStep(
                            index=idx,
                            thought=pending.get("thought"),
                            tool_name=pending.get("tool_name"),
                            tool_args=pending.get("tool_args"),
                            observation=content,
                        )
                    )
                    idx += 1
                    pending = None
                continue
            if role == "assistant":
                tool_name, tool_args = _parse_tau_assistant_call(content)
                if pending is not None:
                    steps.append(
                        TrajectoryStep(
                            index=idx,
                            thought=pending.get("thought"),
                            tool_name=pending.get("tool_name"),
                            tool_args=pending.get("tool_args"),
                            observation="",
                        )
                    )
                    idx += 1
                    pending = None
                if tool_name is None:
                    # Plain text assistant message — emit immediately.
                    steps.append(
                        TrajectoryStep(
                            index=idx,
                            thought=content,
                            tool_name=None,
                            tool_args=None,
                            observation="",
                        )
                    )
                    idx += 1
                else:
                    pending = {
                        "thought": "",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                    }
                continue
            # Catch-all: treat as plain text agent step.
            steps.append(
                TrajectoryStep(
                    index=idx,
                    thought=content if "thought" in role else None,
                    tool_name=role or None,
                    tool_args=None,
                    observation=content if "thought" not in role else "",
                )
            )
            idx += 1
    if pending is not None:
        steps.append(
            TrajectoryStep(
                index=idx,
                thought=pending.get("thought"),
                tool_name=pending.get("tool_name"),
                tool_args=pending.get("tool_args"),
                observation="",
            )
        )
    return steps


def _extract_steps_magentic(raw_steps: list[dict[str, Any]]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    idx = 1
    for step in raw_steps:
        for ss in step.get("substeps", []):
            role = (ss.get("role") or "").strip()
            role_lc = role.lower()
            content = ss.get("content") or ""
            if role_lc in _USER_ROLES:
                continue
            is_thought = "thought" in role_lc
            steps.append(
                TrajectoryStep(
                    index=idx,
                    thought=content if is_thought else None,
                    tool_name=role or None,
                    tool_args=None,
                    observation="" if is_thought else content,
                )
            )
            idx += 1
    return steps


def _lookup_root_cause_category(
    failures: list[dict[str, Any]], root_cause_failure_id: object
) -> str | None:
    if root_cause_failure_id is None:
        return None
    target = str(root_cause_failure_id)
    for fl in failures:
        if str(fl.get("failure_id")) == target:
            cat = fl.get("failure_category")
            return str(cat) if isinstance(cat, str) and cat else None
    return None


def _build_record(
    annotation: dict[str, Any],
    raw: dict[str, Any],
    framework: str,
    source_path: Path,
) -> TrajectoryRecord:
    raw_steps = raw.get("steps") or []
    if framework == "tau_retail":
        steps = _extract_steps_tau(raw_steps)
    else:
        steps = _extract_steps_magentic(raw_steps)

    failures = annotation.get("failures") or []
    failure_categories = [
        fl.get("failure_category")
        for fl in failures
        if isinstance(fl.get("failure_category"), str)
    ]
    root_cause_failure_id = annotation.get("root_cause_failure_id")
    root_cause_category = _lookup_root_cause_category(failures, root_cause_failure_id)

    labels: dict[str, Any] = {
        "framework": framework,
        "num_failures": int(annotation.get("num_failures", len(failures)) or 0),
        "failure_categories_list": failure_categories,
        "root_cause_category": root_cause_category,
        "root_cause_failure_id": (
            str(root_cause_failure_id) if root_cause_failure_id is not None else None
        ),
        "root_cause_reason": annotation.get("root_cause_reason"),
        "failure_summary": annotation.get("failure_summary"),
    }

    return TrajectoryRecord(
        query_id=str(annotation.get("trajectory_id", "")),
        question=raw.get("instruction"),
        gold=None,
        pred=None,
        # AgentRx ships failed trajectories only.
        judgment_correct=False,
        tokens=0,
        time_s=0.0,
        n_iters=len(steps),
        n_retrieved_positive=0,
        n_retrieved_negative=0,
        n_added_hints=0,
        phase=framework,
        source_dir=f"agentrx_{framework}",
        source_path=str(source_path),
        steps=steps,
        labels=labels,
        raw={},
    )


def _load_split(
    annotation_path: Path,
    raw_path: Path,
    framework: str,
    raw_id_normalizer=lambda x: x,
) -> list[TrajectoryRecord]:
    if not annotation_path.exists() or not raw_path.exists():
        logger.warning(
            "AgentRx split missing files: %s / %s", annotation_path, raw_path
        )
        return []
    annotations = _read_jsonl(annotation_path)
    raws = _read_jsonl(raw_path)
    raw_by_id = {raw_id_normalizer(str(r.get("trajectory_id", ""))): r for r in raws}

    records: list[TrajectoryRecord] = []
    missed = 0
    for ann in annotations:
        ann_id = str(ann.get("trajectory_id", ""))
        raw = raw_by_id.get(ann_id)
        if raw is None:
            missed += 1
            continue
        records.append(_build_record(ann, raw, framework, source_path=annotation_path))
    logger.info(
        "AgentRx %s: %d records (joined), %d annotations missed raw counterpart",
        framework,
        len(records),
        missed,
    )
    return records


def load_agentrx(input_dir: Path, **_: object) -> list[TrajectoryRecord]:
    # Bulk-load both AgentRx splits from the HF snapshot directory.
    if not input_dir.exists():
        raise FileNotFoundError(f"AgentRx input directory not found: {input_dir}")
    records: list[TrajectoryRecord] = []
    records.extend(
        _load_split(
            input_dir / "tau_retail.jsonl",
            input_dir / "tau_retail_dataset.jsonl",
            framework="tau_retail",
            raw_id_normalizer=_normalize_tau_id,
        )
    )
    records.extend(
        _load_split(
            input_dir / "magentic_one.jsonl",
            input_dir / "magentic_dataset.jsonl",
            framework="magentic",
            raw_id_normalizer=lambda x: x,
        )
    )
    logger.info("Loaded %d AgentRx trajectories from %s", len(records), input_dir)
    return records


register_dataset_loader("agentrx", load_agentrx)
