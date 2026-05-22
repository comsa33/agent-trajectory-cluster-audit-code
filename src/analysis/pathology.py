"""Heuristic descriptive cluster tags.

The rules in this module are NOT a validated failure-mode taxonomy. Each
rule is a short feature-threshold predicate that produces a one-line
descriptive label for a cluster (`retry-storm`, `tool-looping`,
`retrieval-collapse`, ...). The labels are useful for log-skim
descriptions and for the descriptive_cluster_tag column in
cluster_summary.csv, but the empirical evidence in
`docs/leakage_fix_comparison.md`, `docs/planning_phenotype.md`, and
`docs/agentrx_root_cause.md` does NOT support treating these tags as
discovered failure types. Paper-facing claims should therefore stay
restricted to "behavioral execution phenotypes" framing rather than
"automatic failure-mode discovery".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..registry import PATHOLOGY_RULE_SETS


@dataclass(frozen=True)
class PathologyRule:
    """Named predicate over per-cluster aggregate statistics.

    The `name` is a heuristic descriptor (e.g. `retry-storm`), not a
    validated failure category. See the module docstring for scope.
    """

    name: str
    description: str
    predicate: Callable[[dict[str, float]], bool]


def _has(stats: dict[str, float], key: str, default: float = 0.0) -> float:
    value = stats.get(key, default)
    return float(value) if value is not None else default


def default_pathology_rules() -> list[PathologyRule]:
    # Order matters — first matching rule wins. Move the most specific failure
    # patterns to the top so they take precedence over broad fallbacks.
    return [
        PathologyRule(
            "retrieval-collapse",
            "High empty-observation ratio with high iteration count.",
            lambda s: _has(s, "avg_empty_observation_ratio") >= 0.5
            and _has(s, "avg_n_iters") >= 8,
        ),
        PathologyRule(
            "retry-storm",
            "Identical (tool, args) signatures repeated within a short window.",
            lambda s: _has(s, "avg_retry_burst_max") >= 4
            or _has(s, "avg_retry_burst_ratio") >= 0.4,
        ),
        PathologyRule(
            "hypothesis-fixation",
            "Repeated keywords / entities and elevated error rate.",
            lambda s: (
                _has(s, "avg_repeated_search_keyword_ratio") >= 0.4
                or _has(s, "avg_fixation_score") >= 0.4
            )
            and (1.0 - _has(s, "accuracy")) >= 0.5,
        ),
        PathologyRule(
            "semantic-anchoring-collapse",
            "Hypothesis persistence with positive thought-similarity drift.",
            lambda s: _has(s, "avg_hypothesis_persistence") >= 5
            and _has(s, "avg_thought_drift_slope") > 0.0,
        ),
        PathologyRule(
            "tool-looping",
            "Low tool diversity and repeated tool invocations.",
            lambda s: _has(s, "avg_unique_tool_count") <= 2
            and _has(s, "avg_repeated_tool_ratio") >= 0.5,
        ),
        PathologyRule(
            "over-reasoning",
            "High token + latency budget burned without proportional accuracy.",
            lambda s: _has(s, "avg_tokens") >= 30000
            and _has(s, "avg_time_s") >= 25.0,
        ),
        PathologyRule(
            "premature-finalization",
            "Very few iterations and elevated error rate.",
            lambda s: _has(s, "avg_n_iters") <= 3
            and (1.0 - _has(s, "accuracy")) >= 0.5,
        ),
        PathologyRule(
            "successful-resolution",
            "High accuracy control cluster.",
            lambda s: _has(s, "accuracy") >= 0.7,
        ),
        PathologyRule(
            "mixed-pattern",
            "No specific pathology threshold met.",
            lambda _s: True,
        ),
    ]


PATHOLOGY_RULE_SETS.register("default")(default_pathology_rules)


def _classify_one(stats: dict[str, float], rules: list[PathologyRule]) -> str:
    for rule in rules:
        if rule.predicate(stats):
            return rule.name
    return "mixed-pattern"


def classify_clusters(
    cluster_stats_df: pd.DataFrame,
    rule_set: str = "default",
) -> pd.Series:
    rules_factory = PATHOLOGY_RULE_SETS.get(rule_set)
    rules = rules_factory()
    labels = cluster_stats_df.apply(
        lambda row: _classify_one(row.to_dict(), rules), axis=1
    )
    return labels
