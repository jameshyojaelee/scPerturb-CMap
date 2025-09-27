"""Generate Fig 1 system diagram for scPerturb-CMap using matplotlib."""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, center, width, height, text, fontsize=10):
    """Add a rounded box with centered text."""
    x = center[0] - width / 2
    y = center[1] - height / 2
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def add_arrow(ax, start, end, label=None, text_offset=(0.0, 0.0)):
    """Draw a directional arrow with an optional label."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.4, color="black"),
    )
    if label:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.85, bottom=0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Fig 1. scPerturb-CMap pipeline", fontsize=14, fontweight="bold")

    # Define the boxes (center_x, center_y, width, height, text)
    boxes = [
        ((0.14, 0.50), 0.22, 0.20, "Target JSON\n(genes, weights, metadata)"),
        ((0.42, 0.50), 0.26, 0.22, "L1000 Level 5\nLong table\n(landmarks + annotations)"),
        ((0.62, 0.72), 0.26, 0.20, "Baseline scoring\nCosine invert +\nGSEA blend (z-score)"),
        ((0.62, 0.28), 0.26, 0.20, "DualEncoder metric\nTrain inversion pairs\nNT-Xent or Triplet"),
        ((0.84, 0.50), 0.22, 0.22, "Blended score & ranking\nRanked compounds\nMoA enrichment"),
    ]

    for (center, width, height, text) in boxes:
        add_box(ax, center, width, height, text)

    # Arrow coordinates
    target_right = (0.14 + 0.11, 0.50)
    l1000_left = (0.42 - 0.13, 0.50)
    l1000_top = (0.42, 0.50 + 0.11)
    l1000_bottom = (0.42, 0.50 - 0.11)
    baseline_left = (0.62 - 0.13, 0.72)
    baseline_right = (0.62 + 0.13, 0.72)
    metric_left = (0.62 - 0.13, 0.28)
    metric_right = (0.62 + 0.13, 0.28)
    blended_left_top = (0.84 - 0.11, 0.58)
    blended_left_bottom = (0.84 - 0.11, 0.42)

    add_arrow(ax, target_right, l1000_left, label="Align genes", text_offset=(0.0, 0.04))
    add_arrow(ax, l1000_top, baseline_left, label="Into baseline", text_offset=(0.0, 0.05))
    add_arrow(ax, l1000_bottom, metric_left, label="Into metric", text_offset=(0.0, -0.05))
    add_arrow(ax, baseline_right, blended_left_top, label="Blend", text_offset=(0.02, 0.03))
    add_arrow(ax, metric_right, blended_left_bottom, label="Blend", text_offset=(0.02, -0.03))

    ax.text(
        0.5,
        0.06,
        (
            "scPerturb-CMap: target JSON aligns to L1000 Level 5. "
            "Baseline and DualEncoder scores are blended to rank compounds."
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    output_dir = os.path.join("figs")
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "fig1_system_diagram.png")
    svg_path = os.path.join(output_dir, "fig1_system_diagram.svg")
    for path in (png_path, svg_path):
        fig.savefig(path, dpi=300, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
