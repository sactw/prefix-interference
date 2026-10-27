"""Plot saved matched-text history effects; performs no model evaluation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = [("original8", "Original eight reports"), ("replication4", "Four additional reports")]
    models = [
        ("mamba", "Mamba-2-130M", "#ac443d"),
        ("gdn", "Gated DeltaNet", "#287a67"),
        ("rwkv", "RWKV-7-421M", "#4165a0"),
    ]
    effects = [
        ("N", "Net benefit N\nshort - full"),
        ("B", "Content contrast B\nforeign - full"),
        ("P", "Prefix penalty P\nforeign - short"),
    ]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 3, figsize=(7, 5.8))
    for row in (2, 3):
        for col in range(3):
            axes[row, col].sharex(axes[row - 2, col])
    for cohort, (group, label) in enumerate(groups):
        report = json.loads((args.reports / f"{group}_RESULTS.json").read_text())
        for view, selected in enumerate((models, models[1:])):
            row = 2 * cohort + view
            ys = list(range(len(selected) - 1, -1, -1))
            for col, (effect, title) in enumerate(effects):
                ax = axes[row, col]
                ax.set_axisbelow(True)
                ax.axvline(0, color="0.25", lw=1.1, ls=(0, (4, 3)), zorder=2)
                for y, (key, name, color) in zip(ys, selected):
                    value = report["models"][key]["primary_byte_weighted"][effect]
                    mean, (low, high) = value["mean"], value["ci95"]
                    ax.errorbar(
                        mean,
                        y,
                        xerr=[[mean - low], [high - mean]],
                        fmt="o",
                        color=color,
                        ms=4.5,
                        capsize=3,
                        lw=1.4,
                        zorder=3,
                    )
                ax.set_yticks(ys, [name for _, name, _ in selected], fontsize=9)
                ax.set_ylim(-0.55, len(selected) - 0.45)
                ax.grid(axis="x", color="#e2e2e2", lw=0.5)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(axis="y", length=0, labelleft=col == 0)
                ax.xaxis.set_major_locator(
                    MaxNLocator(3, steps=[1, 2, 5, 10]) if view else MaxNLocator(4)
                )
                ax.tick_params(axis="x", labelsize=9)
                if view:
                    ax.set_facecolor("#f5f7f9")
                if row == 0:
                    ax.set_title(title, fontsize=10, pad=9)
    fig.suptitle("Matched-text context effects", fontsize=13)
    fig.text(0.58, 0.055, "Difference (nats per UTF-8 target byte)", ha="center", fontsize=10)
    fig.text(
        0.5,
        0.012,
        "Shaded zoom rows exclude Mamba. Each column shares scales across cohorts within each view.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.985), h_pad=2.4, w_pad=1.4)
    for row, heading in enumerate(
        (
            "Original 8: overview",
            "Original 8: GDN/RWKV zoom",
            "Additional 4: overview",
            "Additional 4: GDN/RWKV zoom",
        )
    ):
        fig.text(0.015, axes[row, 0].get_position().y1 + 0.008, heading, fontsize=9, va="bottom")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
