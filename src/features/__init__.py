"""Feature extraction package — structural, sequential, and behavioral."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..loaders import TrajectoryRecord
from ..registry import FEATURE_EXTRACTORS
from .behavioral import BehavioralFeatures
from .sequential import SequentialFeatures
from .structural import StructuralFeatures
from .text import trajectory_to_text  # noqa: F401  -- re-export

logger = logging.getLogger(__name__)


# Register the three default extractors so configs can refer to them by name.
FEATURE_EXTRACTORS.register("structural")(StructuralFeatures)
FEATURE_EXTRACTORS.register("sequential")(SequentialFeatures)
FEATURE_EXTRACTORS.register("behavioral")(BehavioralFeatures)


def build_feature_dataframe(
    records: list[TrajectoryRecord],
    enabled: dict[str, bool] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (dataframe, numeric_feature_columns).

    `enabled` selects which extractors to apply. Identifier columns
    (query_id / source_dir / phase) are always present.
    """
    if not records:
        raise ValueError("No trajectories were loaded; cannot extract features.")

    enabled = enabled or {"structural": True, "sequential": True, "behavioral": True}
    extractors = [
        FEATURE_EXTRACTORS.get(name)()
        for name, on in enabled.items()
        if on
    ]

    rows: list[dict[str, Any]] = []
    feature_columns: list[str] = []
    seen_cols: set[str] = set()
    for extractor in extractors:
        for col in extractor.numeric_columns:
            if col not in seen_cols:
                feature_columns.append(col)
                seen_cols.add(col)

    for record in records:
        row: dict[str, Any] = {
            "query_id": record.query_id,
            "source_dir": record.source_dir,
            "source_path": record.source_path,
            "phase": record.phase,
            "final_correct": int(record.judgment_correct),
        }
        for extractor in extractors:
            row.update(extractor.extract(record))
        # Carry dataset-specific labels through as `label_*` columns. They
        # are NOT included in `feature_columns` so the clustering matrix
        # stays unaffected, but downstream label-validation can read them.
        for key, value in record.labels.items():
            col = f"label_{key}"
            if col not in row:
                row[col] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    # `final_correct` is the supervised outcome and must NOT participate in
    # the clustering feature matrix (otherwise cluster <-> safe/unsafe label
    # validation is contaminated by the outcome itself). It stays in the
    # dataframe so downstream correlation / outcome reporting can read it,
    # but it is excluded from `feature_columns`.
    if "final_correct" in feature_columns:
        feature_columns = [c for c in feature_columns if c != "final_correct"]

    logger.info(
        "Extracted %d feature columns for %d trajectories (extractors=%s)",
        len(feature_columns),
        len(df),
        [type(e).__name__ for e in extractors],
    )
    return df, feature_columns
