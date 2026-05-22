#!/usr/bin/env python3
"""Pull 2-3 representative trajectories per dataset for the paper appendix.

We pick one trajectory per illustrative pattern:
    * AFTraj agentic: cluster with strongest mistake_agent purity
    * AgentErrorBench gaia: a planning-tagged failure (task-confound example)
    * AgentRx tau_retail: a trajectory from the single tau_retail cluster
      that shows the within-framework collapse

Output: docs/qualitative_examples.md
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.text import trajectory_to_text  # noqa: E402
from src.loaders import load_trajectories  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("qualitative")


def _pick_aftraj_agentic() -> dict:
    df = pd.read_csv("outputs/aftraj_agentic/trajectory_features.csv")
    # Find a cluster where mistake_agent is concentrated, then pick a record from it.
    valid = df[df["label_mistake_agent"].notna()].copy()
    if valid.empty:
        return {}
    cluster_top = (
        valid.groupby("cluster_id")["label_mistake_agent"]
        .agg(lambda s: s.value_counts().iloc[0] / len(s))
        .sort_values(ascending=False)
    )
    best_cluster = int(cluster_top.index[0])
    sub = valid[valid["cluster_id"] == best_cluster].sort_values(
        "retry_burst_max", ascending=False
    )
    return {
        "dataset": "AFTraj",
        "split": "agentic (unsafe)",
        "query_id": sub.iloc[0]["query_id"],
        "cluster_id": best_cluster,
        "purity_in_cluster": float(cluster_top.iloc[0]),
        "labels": {
            "mistake_agent": sub.iloc[0].get("label_mistake_agent"),
            "unsafe_source": sub.iloc[0].get("label_unsafe_source"),
        },
        "features": {
            "n_iters": int(sub.iloc[0]["n_iters"]),
            "retry_burst_max": float(sub.iloc[0]["retry_burst_max"]),
            "tool_transition_entropy": float(sub.iloc[0]["tool_transition_entropy"]),
            "hypothesis_persistence": int(sub.iloc[0]["hypothesis_persistence"]),
        },
    }


def _pick_aeb_planning() -> dict:
    df = pd.read_csv("outputs/agenterrorbench_full/trajectory_features.csv")
    records = load_trajectories(Path("data/external/agenterrorbench"), dataset_type="agenterrorbench")
    by_id = {r.query_id: r for r in records}
    planning_rows = []
    for _, row in df.iterrows():
        rec = by_id.get(str(row["query_id"]))
        if rec is None:
            continue
        if "planning" in {str(x) for x in rec.labels.get("failure_modules_list") or []}:
            planning_rows.append((row, rec))
    if not planning_rows:
        return {}
    # Pick the gaia one with the most failure_types.
    planning_rows.sort(
        key=lambda pair: (pair[0].get("label_task_type") == "gaia", int(pair[0]["n_iters"])),
        reverse=True,
    )
    row, rec = planning_rows[0]
    return {
        "dataset": "AgentErrorBench",
        "split": str(row.get("label_task_type")),
        "query_id": row["query_id"],
        "cluster_id": int(row["cluster_id"]),
        "labels": {
            "task_type": row.get("label_task_type"),
            "llm_model": row.get("label_llm_model"),
            "failure_type_first": row.get("label_failure_type_first"),
            "failure_modules_list": rec.labels.get("failure_modules_list"),
            "critical_failure_module": row.get("label_critical_failure_module"),
        },
        "features": {
            "n_iters": int(row["n_iters"]),
            "total_tool_calls": int(row["total_tool_calls"]),
            "search_keyword_diversity": float(row.get("search_keyword_diversity", 0.0)),
            "retry_burst_max": float(row["retry_burst_max"]),
        },
    }


def _pick_agentrx_tau() -> dict:
    df = pd.read_csv("outputs/agentrx_full/trajectory_features.csv")
    records = load_trajectories(Path("data/external/agentrx"), dataset_type="agentrx")
    by_id = {r.query_id: r for r in records}
    tau = df[df["source_dir"] == "agentrx_tau_retail"]
    if tau.empty:
        return {}
    # Pick the median-iters trajectory from the tau_retail cluster.
    row = tau.sort_values("n_iters").iloc[len(tau) // 2]
    rec = by_id.get(str(row["query_id"]))
    return {
        "dataset": "AgentRx",
        "split": "tau_retail",
        "query_id": row["query_id"],
        "cluster_id": int(row["cluster_id"]),
        "labels": {
            "framework": "tau_retail",
            "root_cause_category": rec.labels.get("root_cause_category") if rec else None,
            "num_failures": rec.labels.get("num_failures") if rec else None,
            "failure_categories_list": rec.labels.get("failure_categories_list") if rec else [],
        },
        "features": {
            "n_iters": int(row["n_iters"]),
            "total_tool_calls": int(row["total_tool_calls"]),
            "tool_transition_entropy": float(row["tool_transition_entropy"]),
            "hypothesis_persistence": int(row["hypothesis_persistence"]),
        },
    }


def _trajectory_snippet(dataset: str, query_id: str, max_chars: int = 600) -> str:
    if dataset == "AFTraj":
        records = load_trajectories(Path("data/external/aftraj"), dataset_type="aftraj")
    elif dataset == "AgentErrorBench":
        records = load_trajectories(Path("data/external/agenterrorbench"), dataset_type="agenterrorbench")
    else:
        records = load_trajectories(Path("data/external/agentrx"), dataset_type="agentrx")
    for r in records:
        if str(r.query_id) == str(query_id):
            text = trajectory_to_text(r, max_chars=max_chars * 3)
            # Trim to a few representative steps.
            lines = text.split("\n")
            head = lines[:4]
            tail = lines[-2:]
            stitched = "\n".join(head + ["    ... <truncated> ..."] + tail)
            return stitched[: max_chars * 3]
    return "(record not found)"


def main() -> int:
    out_path = Path("docs/qualitative_examples.md")
    examples = [
        _pick_aftraj_agentic(),
        _pick_aeb_planning(),
        _pick_agentrx_tau(),
    ]
    examples = [ex for ex in examples if ex]
    sections = ["# Qualitative trajectory examples", ""]
    sections.append(
        "Three trajectories chosen to illustrate the headline patterns in"
        " the empirical evidence. Each example shows the dataset, the"
        " cluster the trajectory landed in, the ground-truth labels"
        " carried by the loader, and a short trajectory excerpt"
        " (truncated for readability)."
    )
    sections.append("")
    for ex in examples:
        snippet = _trajectory_snippet(ex["dataset"], ex["query_id"])
        sections.append(f"## {ex['dataset']} ({ex['split']}) — `{ex['query_id']}`")
        sections.append("")
        sections.append(f"- cluster_id: **{ex['cluster_id']}**")
        if "purity_in_cluster" in ex:
            sections.append(f"- mistake_agent purity within cluster: {ex['purity_in_cluster']:.2f}")
        sections.append("- ground-truth labels:")
        for k, v in ex["labels"].items():
            sections.append(f"    - {k}: `{v}`")
        sections.append("- selected features:")
        for k, v in ex["features"].items():
            sections.append(f"    - {k}: `{v}`")
        sections.append("")
        sections.append("```text")
        sections.append(snippet)
        sections.append("```")
        sections.append("")
    out_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("wrote %s", out_path)
    print(json.dumps(examples, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
