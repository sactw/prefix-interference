"""Plot measured multi-donor readouts, without model or GPU dependencies.

Example: python -m long_context_diagnostics.plot_panel summary.json
  --output figures/overall --label "Exploratory panel"
Each endpoint gets a separate row because changing E changes scored targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EFFECTS = (
    ("N", "Net benefit N", "short - full"),
    ("content", "Content contrast B", "foreign - full"),
    ("prefix_penalty", "Prefix penalty P", "foreign - short"),
)


def _tokens(value):
    return f"{value // 1024}K" if value and value % 1024 == 0 else str(value)


def plot_panel(report, output, label="Context effects", group="overall"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = report["panels"]
    if not panels:
        raise ValueError("No measured panels to plot")
    models = {json.dumps(panel["configuration"]["model"], sort_keys=True) for panel in panels}
    if len(models) != 1:
        raise ValueError(
            "Plot one model/configuration at a time; do not combine raw token CEs across models"
        )
    blocks = {panel["configuration"]["targets_per_block"] for panel in panels}
    modes = {panel["configuration"].get("execution") for panel in panels}
    if len(blocks) != 1 or len(modes) != 1:
        raise ValueError("Select one target block size and execution mode for this figure")
    by_end = {}
    for panel in panels:
        config = panel["configuration"]
        row = by_end.setdefault(config["end"], {})
        if config["recent"] in row:
            raise ValueError(
                "Multiple estimates for the same E/C; select one boundary configuration"
            )
        row[config["recent"]] = panel if group == "overall" else panel["groups"][group]
    ends = sorted(by_end)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(len(ends), 3, figsize=(10.8, 1.3 + 2.65 * len(ends)), squeeze=False)
    colors = ("#2166ac", "#b35806", "#1b7837")
    for r, end in enumerate(ends):
        recent_sizes = sorted(by_end[end])
        labels = [_tokens(c) for c in recent_sizes]
        for column, (effect, title, contrast) in enumerate(EFFECTS):
            ax = axes[r, column]
            ax.set_axisbelow(True)
            ax.axhline(0, color="0.25", linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
            for x, recent in enumerate(recent_sizes):
                estimate = by_end[end][recent]["effects"][effect]
                mean, interval = estimate["mean"], estimate.get("ci95")
                if mean is None:
                    continue
                color = colors[column]
                # Draw percentile endpoints directly: a mean need not lie
                # inside a percentile interval for a skewed bootstrap sample.
                if interval is not None:
                    low, high = interval
                    ax.vlines(x, low, high, color=color, linewidth=1.4, zorder=3)
                    ax.hlines(
                        (low, high), x - 0.055, x + 0.055, color=color, linewidth=1.4, zorder=3
                    )
                ax.plot(x, mean, "o", color=color, markersize=5, zorder=4)
            ax.set_xticks(range(len(recent_sizes)), labels)
            ax.set_xlim(-0.4, len(recent_sizes) - 0.6)
            ax.set_xlabel("Recent prefix C")
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(axis="y", color="0.9", linewidth=0.5)
            if r == 0:
                ax.set_title(f"{title}\n{contrast}", fontsize=12)
            if column == 0:
                ax.set_ylabel(f"Endpoint E={_tokens(end)}\nDifference (nats / target)")
    model = panels[0]["configuration"]["model"]["model_id"]
    display_model = {"state-spaces/mamba2-130m": "Mamba-2-130M"}.get(model, model)
    display_label = label[:1].lower() + label[1:]
    fig.suptitle(f"{display_model}: {display_label}", fontsize=13)
    counts = {
        (cell["documents"], cell["targets"]) for row in by_end.values() for cell in row.values()
    }
    if len(counts) == 1:
        documents, targets = next(iter(counts))
        count_note = f"{documents} documents; {targets:,} distinct targets per setting after donor averaging."
    else:
        count_note = "Document and target counts vary by setting; see the source summary."
    fig.text(
        0.5,
        0.02,
        f"{count_note} Group: {group}; T={next(iter(blocks)):,}.\n"
        "Bars: pointwise 95% document-bootstrap intervals. Different endpoint rows score different target blocks.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.97), h_pad=2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output.with_suffix(suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="output filename stem")
    parser.add_argument("--label", default="Context effects")
    parser.add_argument(
        "--group", default="overall", help="overall, far4, far8, never4, recent4, etc."
    )
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    paths = plot_panel(report, args.output, args.label, args.group)
    print(json.dumps({"figures": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
