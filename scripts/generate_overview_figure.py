"""Generate the methodology overview schematic.

Six boxes wired left-to-right:
  Trajectory logs -> Canonical schema -> Feature stack -> Clustering
  -> Four validation checks -> Claim boundary.

Output: a single PNG that goes next to main_sn.tex in the SN-CS venue
folder (per SN user-manual: "Do not place your image files in
subfolders"), and the same PNG also lands in paper/figures/ for the
JIPS legacy source.

Run:
  uv run python scripts/generate_overview_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def make_overview_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 3.8), dpi=200)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 36)
    ax.axis("off")

    box_w, box_h = 17.5, 15.0
    y_main = 11
    gap = 2.0
    n = 6
    total_w = n * box_w + (n - 1) * gap
    x0 = (120 - total_w) / 2

    boxes = [
        {
            "title": "Trajectory logs",
            "lines": [
                "raw per-step",
                "Thought / Action /",
                "Observation",
            ],
            "color": "#e7eef7",
            "edge": "#3b6694",
        },
        {
            "title": "Canonical\nschema",
            "lines": [
                "TrajectoryRecord",
                "TrajectoryStep",
                "labels: dict",
            ],
            "color": "#e7eef7",
            "edge": "#3b6694",
        },
        {
            "title": "Feature stack",
            "lines": [
                "26 features",
                "structural / sequential /",
                "behavioral",
            ],
            "color": "#e7eef7",
            "edge": "#3b6694",
        },
        {
            "title": "Clustering",
            "lines": [
                "K-Means,",
                "silhouette-selected K",
                "(pluggable encoder)",
            ],
            "color": "#e7eef7",
            "edge": "#3b6694",
        },
        {
            "title": "Four validation\nchecks",
            "lines": [
                "1. leakage audit",
                "2. stratified label",
                "3. per-tag multilabel",
                "4. matched-K baseline",
            ],
            "color": "#fdecd4",
            "edge": "#b97a25",
        },
        {
            "title": "Claim boundary",
            "lines": [
                "report only what",
                "survives all four",
                "checks; flag confound",
            ],
            "color": "#dceadc",
            "edge": "#3f7a3f",
        },
    ]

    centers = []
    for i, b in enumerate(boxes):
        x = x0 + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y_main),
            box_w,
            box_h,
            boxstyle="round,pad=0.5,rounding_size=1.4",
            linewidth=1.4,
            edgecolor=b["edge"],
            facecolor=b["color"],
        )
        ax.add_patch(rect)

        cx = x + box_w / 2
        cy = y_main + box_h - 2.5
        ax.text(
            cx,
            cy,
            b["title"],
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#1c1c1c",
        )

        line_y = y_main + box_h - 7.5
        for line in b["lines"]:
            ax.text(
                cx,
                line_y,
                line,
                ha="center",
                va="top",
                fontsize=8.3,
                color="#2a2a2a",
            )
            line_y -= 1.9

        centers.append((cx, y_main + box_h / 2))

    arrow_kw = dict(
        arrowstyle="-|>,head_length=0.7,head_width=0.45",
        linewidth=1.5,
        color="#444444",
        shrinkA=4,
        shrinkB=4,
    )
    for i in range(n - 1):
        (x1, y1), (x2, y2) = centers[i], centers[i + 1]
        ax.annotate(
            "",
            xy=(x2 - box_w / 2 + 0.2, y2),
            xytext=(x1 + box_w / 2 - 0.2, y1),
            arrowprops=arrow_kw,
        )

    pipeline_y = y_main + box_h + 2.5
    pipeline_label_x = x0 + (box_w * 4 + gap * 3) / 2
    ax.text(
        pipeline_label_x,
        pipeline_y,
        "unsupervised pipeline",
        ha="center",
        va="bottom",
        fontsize=10.5,
        style="italic",
        color="#3b6694",
    )

    validation_x_start = x0 + 4 * (box_w + gap)
    validation_x_end = x0 + 5 * (box_w + gap) + box_w
    ax.text(
        (validation_x_start + validation_x_end) / 2,
        pipeline_y,
        "audit pass",
        ha="center",
        va="bottom",
        fontsize=10.5,
        style="italic",
        color="#b97a25",
    )

    footer_y = y_main - 5.0
    ax.text(
        60,
        footer_y,
        "any cluster--label number that does not survive the four checks "
        "is reported as a confound, not as a finding",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=(
            "paper/venues/sn-computer-science/f0_methodology_overview.png"
        ),
        help="output PNG path",
    )
    parser.add_argument(
        "--also-copy-to",
        default="paper/figures/f0_methodology_overview.png",
        help=(
            "second output path (kept alongside the JIPS-source figures "
            "so paper/main.tex can pick it up later if desired)"
        ),
    )
    args = parser.parse_args()

    out_primary = Path(args.out)
    make_overview_figure(out_primary)
    print(f"wrote {out_primary}")

    out_copy = Path(args.also_copy_to)
    if out_copy != out_primary:
        out_copy.parent.mkdir(parents=True, exist_ok=True)
        out_copy.write_bytes(out_primary.read_bytes())
        print(f"wrote {out_copy}")


if __name__ == "__main__":
    main()
