"""Statistical correlation between cluster assignment and outcome variables."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)


def _eta_squared(groups: list[np.ndarray]) -> float:
    # Effect-size companion to a one-way ANOVA F-test.
    grand = np.concatenate(groups)
    grand_mean = grand.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((grand - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


def _cramers_v(contingency: np.ndarray, chi2: float) -> float:
    n = contingency.sum()
    if n == 0:
        return 0.0
    r, c = contingency.shape
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def cluster_outcome_correlation(
    df: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    # For each target column, test whether cluster assignment is associated
    # with the outcome. Continuous -> ANOVA + eta^2, binary -> chi^2 + Cramer V.
    rows: list[dict[str, object]] = []
    if "cluster_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster_id' column")

    cluster_ids = sorted(df["cluster_id"].unique())
    if len(cluster_ids) < 2:
        logger.warning("Fewer than 2 clusters; skipping correlation tests.")
        return pd.DataFrame(columns=["target", "test", "statistic", "p_value", "effect_size"])

    for target in targets:
        if target not in df.columns:
            logger.info("Correlation skipped: target %s not in dataframe", target)
            continue
        series = df[target]

        is_binary = series.dropna().nunique() <= 2 and set(
            series.dropna().unique()
        ).issubset({0, 1, True, False})

        if is_binary:
            contingency = pd.crosstab(df["cluster_id"], series).to_numpy()
            if contingency.size == 0:
                continue
            try:
                chi2, p_value, _dof, _ = sps.chi2_contingency(contingency)
                effect = _cramers_v(contingency, chi2)
            except ValueError as exc:
                logger.warning("chi2 failed for %s: %s", target, exc)
                continue
            rows.append(
                {
                    "target": target,
                    "test": "chi_square",
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "effect_size": effect,
                    "effect_size_metric": "cramers_v",
                }
            )
        else:
            groups = [
                series[df["cluster_id"] == cid].dropna().to_numpy(dtype=np.float64)
                for cid in cluster_ids
            ]
            groups = [g for g in groups if g.size > 1]
            if len(groups) < 2:
                continue
            try:
                f_stat, p_value = sps.f_oneway(*groups)
                effect = _eta_squared(groups)
            except (ValueError, TypeError) as exc:
                logger.warning("ANOVA failed for %s: %s", target, exc)
                continue
            rows.append(
                {
                    "target": target,
                    "test": "anova_f",
                    "statistic": float(f_stat),
                    "p_value": float(p_value),
                    "effect_size": float(effect),
                    "effect_size_metric": "eta_squared",
                }
            )

    return pd.DataFrame(rows)
