"""Commands for model-independent fixtures and saved paired-loss readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .measurement import CONDITIONS, build_histories, load_pair, save_pair
from .readout import classify_repeats, summarize


def write_report(paths, output: Path, bootstrap_samples: int, seed: int) -> dict:
    documents = [load_pair(path) for path in paths]
    result = summarize(documents, bootstrap_samples=bootstrap_samples, seed=seed)
    result["records"] = [
        {"file": Path(path).name, "document_id": doc["document_id"], "metadata": doc["metadata"]}
        for path, doc in zip(paths, documents)
    ]
    kinds = {doc["metadata"].get("kind", "unspecified") for doc in documents}
    result["kind"] = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    brief = {
        "kind": result["kind"],
        "documents": result["documents"],
        "targets": result["targets"],
        "effects": result["effects"],
        "report": str(output),
    }
    print(json.dumps(brief, indent=2, allow_nan=False))
    return result


def fixture(output: Path, bootstrap_samples: int, seed: int) -> dict:
    """Write hand-made losses that exercise the readout; no model is run."""
    output.mkdir(parents=True, exist_ok=True)
    ids = np.array(
        [7, 8, 9, 10, 99, 7, 8, 9, 10, 21, 22, 23, 24, 7, 8, 9, 10, 21, 22, 23, 24, 55, 56, 57]
    )
    paths = []
    for i, net in enumerate((0.02, -0.01, 0.05)):
        plan = build_histories(
            ids, ids + 100, end=24, recent_tokens=8, target_tokens=8, prefix_id=0
        )
        base = np.linspace(1.0, 2.0, 8, dtype=np.float32) + i / 10
        losses = {"full": base, "short": base + net, "foreign": base + 0.06}
        arms = {
            name: {
                "target_ids": plan.inputs[name][plan.target_starts[name] :],
                "target_positions": plan.target_positions,
                "losses": losses[name],
            }
            for name in CONDITIONS
        }
        bins = {
            str(n): classify_repeats(ids, plan.target_positions, plan.retained_start, n=n)
            for n in (4, 8)
        }
        path = output / f"fixture-{i}.npz"
        save_pair(
            path,
            document_id=f"constructed-{i}",
            arms=arms,
            repeat_bins=bins,
            metadata={
                "kind": "constructed_fixture",
                "history": plan.describe(),
                "note": "Hand-constructed losses; no language model was executed.",
            },
        )
        paths.append(path)
    return write_report(paths, output / "fixture_summary.json", bootstrap_samples, seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture_parser = sub.add_parser("fixture", help="light CPU-only constructed loss fixture")
    fixture_parser.add_argument("--output", type=Path, required=True)
    report_parser = sub.add_parser("report", help="read paired NPZ blocks (no model imports)")
    report_parser.add_argument("records", type=Path, nargs="+")
    report_parser.add_argument("--output", type=Path, required=True)
    for command in (fixture_parser, report_parser):
        command.add_argument("--bootstrap-samples", type=int, default=1000)
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "fixture":
        fixture(args.output, args.bootstrap_samples, args.seed)
    else:
        write_report(args.records, args.output, args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
