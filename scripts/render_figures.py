#!/usr/bin/env python3
"""Render the headline figures for the JIPS manuscript.

F1 — three-panel headline 'apparent alignment vs within-domain collapse'
F2 — AFTraj baseline parity for label_mistake_agent

Outputs land under paper/figures/ (created if missing) so the LaTeX
source can include them with a fixed relative path.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("render_figures")

FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _read_label_nmi(csv_path: Path, label: str) -> tuple[float, float | None]:
    """Return (nmi, ari) for a given label row from a label_validation csv."""
    if not csv_path.exists():
        return float("nan"), None
    df = pd.read_csv(csv_path)
    row = df[df["label"] == label]
    if row.empty:
        return float("nan"), None
    return float(row.iloc[0]["nmi"]), float(row.iloc[0].get("ari", float("nan")))


def _seed_band(seed_summary: pd.DataFrame, experiment: str, label: str) -> float:
    row = seed_summary[
        (seed_summary["experiment"] == experiment)
        & (seed_summary["label"] == label)
    ]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["nmi_std"])


def render_f1() -> Path:
    seed_summary = pd.read_csv("outputs/seed_variance/summary.csv")

    # --- Panel A: AFTraj ----------------------------------------------------
    aftraj_groups = ["overall", "math", "coding", "agentic"]
    aftraj_paths = [
        Path("outputs/aftraj_full/cluster_label_validation.csv"),
        Path("outputs/aftraj_math/cluster_label_validation.csv"),
        Path("outputs/aftraj_coding/cluster_label_validation.csv"),
        Path("outputs/aftraj_agentic/cluster_label_validation.csv"),
    ]
    aftraj_nmi = [_read_label_nmi(p, "label_mistake_agent")[0] for p in aftraj_paths]
    aftraj_seed = [
        _seed_band(seed_summary, exp, "label_mistake_agent")
        for exp in ["aftraj_full", "aftraj_math", "aftraj_coding", "aftraj_agentic"]
    ]
    aftraj_verdict = ["overall", "collapse", "survive", "survive"]

    # --- Panel B: AgentErrorBench ------------------------------------------
    aeb_groups = ["overall", "alfworld", "gaia", "webshop"]
    aeb_path = Path("outputs/agenterrorbench_full/cluster_label_validation.csv")
    aeb_strat = Path("outputs/agenterrorbench_full/cluster_label_validation_stratified.csv")
    aeb_overall_nmi = _read_label_nmi(aeb_path, "label_failure_type_first")[0]
    strat_df = pd.read_csv(aeb_strat)

    def _strat(group_value: str) -> float:
        r = strat_df[
            (strat_df["group_value"] == group_value)
            & (strat_df["label"] == "label_failure_type_first")
        ]
        return float(r.iloc[0]["nmi"]) if len(r) else float("nan")

    aeb_nmi = [
        aeb_overall_nmi,
        _strat("alfworld"),
        _strat("gaia"),
        _strat("webshop"),
    ]
    aeb_seed = [
        _seed_band(seed_summary, "agenterrorbench_full", "label_failure_type_first")
    ] + [0.0, 0.0, 0.0]
    aeb_verdict = ["overall", "collapse", "weak", "weak"]

    # --- Panel C: AgentRx --------------------------------------------------
    agx_groups = ["overall", "magentic", "tau_retail"]
    agx_path = Path("outputs/agentrx_full/cluster_label_validation.csv")
    agx_strat_path = Path("outputs/agentrx_full/cluster_label_validation_stratified.csv")
    agx_overall = _read_label_nmi(agx_path, "label_root_cause_category")[0]
    agx_strat = pd.read_csv(agx_strat_path)

    def _agx_strat(fw: str) -> float:
        r = agx_strat[
            (agx_strat["group_value"] == fw)
            & (agx_strat["label"] == "label_root_cause_category")
        ]
        return float(r.iloc[0]["nmi"]) if len(r) else float("nan")

    agx_nmi = [agx_overall, _agx_strat("magentic"), _agx_strat("tau_retail")]
    agx_seed = [
        _seed_band(seed_summary, "agentrx_full", "label_root_cause_category"),
        0.0,
        0.0,
    ]
    agx_verdict = ["overall", "weak", "collapse"]

    # --- Compose figure ----------------------------------------------------
    # Publication-style: verdicts color-coded; no annotation arrows; no
    # figure-level suptitle; tau_retail single-cluster cell is flagged
    # with an asterisk that the LaTeX caption explains.
    verdict_color = {
        "overall":  "#7F7F7F",  # gray
        "survive":  "#2CA02C",  # green
        "weak":     "#DD8452",  # orange
        "collapse": "#C44E52",  # red
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def _bar_panel(ax, labels, values, errors, verdicts, title,
                   star_indices=()):
        x = np.arange(len(labels))
        colors = [verdict_color[v] for v in verdicts]
        ax.bar(
            x, values, yerr=errors, capsize=3,
            color=colors, edgecolor="white", linewidth=0.5,
        )
        for i, v in enumerate(values):
            if not np.isnan(v):
                tag = f"{v:.2f}*" if i in star_indices else f"{v:.2f}"
                ax.text(i, v + 0.02, tag, ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=9)
        ax.set_ylim(0, max(0.85, np.nanmax(values) * 1.25))
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("NMI", fontsize=9)
        ax.axhline(0.0, color="black", linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    _bar_panel(
        axes[0], aftraj_groups, aftraj_nmi, aftraj_seed, aftraj_verdict,
        "AFTraj: mistake-agent alignment",
    )
    _bar_panel(
        axes[1], aeb_groups, aeb_nmi, aeb_seed, aeb_verdict,
        "AgentErrorBench: failure-type alignment",
    )
    _bar_panel(
        axes[2], agx_groups, agx_nmi, agx_seed, agx_verdict,
        "AgentRx: root-cause alignment",
        star_indices=(2,),  # tau_retail single-cluster collapse
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=verdict_color[k])
        for k in ("overall", "survive", "weak", "collapse")
    ]
    legend_labels = [
        "overall (un-stratified)",
        "survives stratification",
        "weak within-stratum",
        "collapses (≈ 0)",
    ]
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=4, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()
    out = FIG_DIR / "f1_apparent_vs_within_domain.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def render_f2() -> Path:
    csv = Path("outputs/aftraj_full/cluster_baseline_comparison.csv")
    df = pd.read_csv(csv)
    sub = df[df["label"] == "label_mistake_agent"].copy()
    order = [
        "oracle:label_domain",
        "full_features",
        "quantile:tool_transition_entropy",
        "quantile:total_tool_calls",
        "quantile:n_iters",
        "quantile:retry_burst_max",
        "random",
    ]
    sub = sub.set_index("baseline").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [
        "#8172B2" if name == "full_features"
        else "#C44E52" if name.startswith("oracle:")
        else "#A0A0A0" if name == "random"
        else "#4C72B0"
        for name in sub["baseline"]
    ]
    bars = ax.barh(sub["baseline"], sub["nmi"], color=colors, edgecolor="white")
    for bar, value in zip(bars, sub["nmi"]):
        ax.text(
            value + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}", va="center", fontsize=9,
        )
    ax.set_xlabel("NMI vs label_mistake_agent (AFTraj overall)")
    ax.set_title("Baseline parity: full pipeline barely beats single features,\nloses to domain-oracle")
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "f2_aftraj_baseline_parity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)
    return out


def main() -> int:
    render_f1()
    render_f2()
    return 0


if __name__ == "__main__":
    sys.exit(main())
