"""Plot the fixed-target length ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_ladder(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["lengths"]
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharex=True)
    titles = (
        ("N", "Net benefit N\nshort - full", "#2166ac"),
        ("B", "Content contrast B\nforeign - full", "#b35806"),
        ("P", "Prefix penalty P\nforeign - short", "#1b7837"),
    )
    xs = [row["serialized_full_tokens"] for row in rows]
    for ax, (effect, title, color) in zip(axes, titles):
        means = [row["effects"][effect]["mean"] for row in rows]
        ax.set_axisbelow(True)
        ax.axhline(0, color="0.25", lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.plot(xs, means, "o-", color=color, lw=1.2, markersize=5, zorder=4)
        for x, row in zip(xs, rows):
            interval = row["effects"][effect]["ci95"]
            if interval is not None:
                ax.vlines(x, *interval, color=color, lw=1.3, zorder=3)
                ax.hlines(interval, x / 1.04, x * 1.04, color=color, lw=1.3, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [f"{x:,}" for x in xs])
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel("Serialized full input (tokens)")
        ax.grid(axis="y", color="0.9", lw=0.5)
    axes[0].set_ylabel("Difference (nats / target)")
    count = report["coverage"]["paired_documents"]
    fig.suptitle("Mamba-2-130M: identical targets and recent context", fontsize=13)
    fig.text(
        0.5,
        0.02,
        f"{count} documents; recent R=512; targets Y=1,024; only older prefix length changes.\n"
        "Bars: pointwise 95% paired document-bootstrap intervals; fixed donor assignment.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(ext) for ext in (".png", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            [
                str(p)
                for p in plot_ladder(
                    json.loads(args.report.read_text(encoding="utf-8")), args.output
                )
            ]
        )
    )


if __name__ == "__main__":
    main()
