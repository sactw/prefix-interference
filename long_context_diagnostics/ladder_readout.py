"""Paired fixed-target, fixed-recent-prefix length ladder; CPU only.

Select one assigned donor, average targets within each document, and report
equal-weight document means. Every length and change uses the same bootstrap
draws. Intervals are pointwise and conditional on the selected donor/texts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .measurement import CONDITIONS, load_pair, validate_arms
from .panel_readout import panel_identity


EFFECTS = ("N", "B", "P")


def _fixed_identity(row):
    identity = panel_identity(row)
    # This is the deliberately varied factor; repeat bins may also change
    # when a formerly invisible occurrence enters the longer full history.
    identity.pop("full_history_start")
    return identity


def _estimate(values, draws):
    if not len(values):
        return {"mean": None, "ci95": None}
    sampled = values[draws].mean(axis=1) if len(draws) else np.empty(0)
    return {
        "mean": float(np.mean(values)),
        "ci95": np.percentile(sampled, [2.5, 97.5]).tolist() if len(sampled) else None,
    }


def summarize_ladder(
    rows,
    *,
    donor_id=None,
    fars=None,
    baseline_far=511,
    bootstrap_samples=2000,
    seed=20261024,
    expected_documents=8,
):
    """Compare the documents that have every requested length.

    Mismatched targets, texts or settings raise an error; code revisions may
    differ. Missing cells are reported and never imputed.
    """
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    donors = sorted({row["metadata"]["donor_id"] for row in rows})
    if donor_id is None:
        if len(donors) != 1:
            raise ValueError("select one donor with --donor-id")
        donor_id = donors[0]
    selected = [row for row in rows if row["metadata"]["donor_id"] == donor_id]
    lengths = sorted(
        set(
            fars
            if fars is not None
            else [int(row["metadata"]["history"]["far_prefix_tokens"]) for row in selected]
        )
    )
    if not lengths or baseline_far not in lengths or min(lengths) < 0:
        raise ValueError("fars must include the nonnegative baseline_far")
    cells, identity = {}, None
    for row in selected:
        history = row["metadata"]["history"]
        far = int(history["far_prefix_tokens"])
        if far not in lengths:
            continue
        doc = row["document_id"]
        if (doc, far) in cells:
            raise ValueError(f"{doc}/F{far}: duplicate cell; select one attempt")
        current = _fixed_identity(row)
        if identity is not None and current != identity:
            raise ValueError(f"{doc}/F{far}: model, execution or fixed E/R/T differs")
        identity = current
        validate_arms(
            {
                key: {
                    "target_ids": row["target_ids"],
                    "target_positions": row["target_positions"],
                    "losses": row[key],
                }
                for key in CONDITIONS
            }
        )
        end, recent, targets = (identity[key] for key in ("end", "recent", "targets_per_block"))
        if not np.array_equal(row["target_positions"], np.arange(end - targets, end)):
            raise ValueError(f"{doc}/F{far}: absolute target positions disagree with E/T")
        start = int(history.get("full_history_start", 0))
        if start < 0 or start + far + recent + targets != end:
            raise ValueError(f"{doc}/F{far}: full history start disagrees with F/R/T")
        if history.get("retained_start", end - targets - recent) != end - targets - recent:
            raise ValueError(f"{doc}/F{far}: retained start differs")
        cells[doc, far] = row

    observed = sorted({doc for doc, _ in cells})
    complete = [doc for doc in observed if all((doc, far) in cells for far in lengths)]
    # Check every observed pair, even when a third length is still pending.
    for doc in observed:
        group = [cells[doc, far] for far in lengths if (doc, far) in cells]
        first = group[0]
        for row in group[1:]:
            for key in ("target_ids", "target_positions"):
                if not np.array_equal(first[key], row[key]):
                    raise ValueError(f"{doc}: {key} differ across lengths")
            for key in ("text_sha256", "donor_text_sha256"):
                if first["metadata"].get(key) != row["metadata"].get(key):
                    raise ValueError(f"{doc}: {key} differs across lengths")

    draws = (
        np.random.default_rng(seed).integers(
            0, len(complete), size=(bootstrap_samples, len(complete))
        )
        if complete
        else np.empty((0, 0), dtype=int)
    )
    ce = np.asarray(
        [
            [
                [np.mean(cells[doc, far][key], dtype=np.float64) for key in CONDITIONS]
                for doc in complete
            ]
            for far in lengths
        ]
    )
    ce = ce.reshape(len(lengths), len(complete), len(CONDITIONS))
    effects = np.stack(
        (ce[:, :, 1] - ce[:, :, 0], ce[:, :, 2] - ce[:, :, 0], ce[:, :, 2] - ce[:, :, 1]), axis=-1
    )
    baseline = lengths.index(baseline_far)
    changes = effects - effects[baseline]
    result = {
        "schema_version": 1,
        "kind": "fixed_target_length_ladder",
        "aggregation": "macro_document_mean_of_per_target_nats",
        "configuration": identity,
        "donor_id": donor_id,
        "baseline_far": baseline_far,
        "definitions": {"N": "short-full", "B": "foreign-full", "P": "foreign-short"},
        "bootstrap": {
            "method": "paired_document_percentile",
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence": 0.95,
            "same_draws_all_lengths_and_changes": True,
            "scope": "pointwise; conditional on selected documents and donor assignment",
        },
        "coverage": {
            "expected_documents": expected_documents,
            "paired_documents": len(complete),
            "selected_records": len(cells),
            "excluded_other_donor_records": len(rows) - len(selected),
            "records_per_far": {str(f): sum((d, f) in cells for d in observed) for f in lengths},
            "unpaired_missing_fars": {
                d: [f for f in lengths if (d, f) not in cells]
                for d in observed
                if d not in complete
            },
            "complete_counts": len(complete) == expected_documents
            if expected_documents is not None
            else None,
            "note": "Estimates use the common complete document set; no missing cells imputed.",
        },
        "distinct_targets": sum(len(cells[doc, baseline_far]["target_ids"]) for doc in complete),
        "identity_checks": {
            "target_ids_equal_across_lengths": True,
            "absolute_positions_equal_across_lengths": True,
            "model_and_fixed_history_equal": True,
        },
        "lengths": [],
        "per_document": [],
    }
    max_short = 0.0
    for doc_index, doc in enumerate(complete):
        short = np.stack([cells[doc, f]["short"].astype(np.float64) for f in lengths])
        spread = float(np.ptp(short, axis=0).max())
        max_short = max(max_short, spread)
        row = cells[doc, baseline_far]
        result["per_document"].append(
            {
                "document_id": doc,
                "targets": len(row["target_ids"]),
                "target_position_first": int(row["target_positions"][0]),
                "target_position_last": int(row["target_positions"][-1]),
                "max_short_per_target_range": spread,
                "lengths": [
                    {
                        "far": f,
                        "ce": dict(zip(CONDITIONS, ce[i, doc_index].tolist())),
                        "effects": dict(zip(EFFECTS, effects[i, doc_index].tolist())),
                        "change_vs_baseline": dict(zip(EFFECTS, changes[i, doc_index].tolist())),
                    }
                    for i, f in enumerate(lengths)
                ],
            }
        )
    result["max_short_per_target_range"] = max_short if complete else None
    for i, far in enumerate(lengths):
        result["lengths"].append(
            {
                "far": far,
                "serialized_full_tokens": far
                + identity["recent"]
                + identity["targets_per_block"]
                + int(identity["boundary_token"] is not None)
                if identity
                else None,
                "ce": {key: _estimate(ce[i, :, j], draws) for j, key in enumerate(CONDITIONS)},
                "effects": {
                    key: _estimate(effects[i, :, j], draws) for j, key in enumerate(EFFECTS)
                },
                "change_vs_baseline": {
                    key: _estimate(changes[i, :, j], draws) for j, key in enumerate(EFFECTS)
                },
                "short_mean_change_vs_baseline": _estimate(ce[i, :, 1] - ce[baseline, :, 1], draws),
            }
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path, help="NPZ files or directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--donor-id", help="select the fixed donor by saved donor_id, not folder name"
    )
    parser.add_argument("--fars", nargs="+", type=int, default=[511, 6655, 31232])
    parser.add_argument("--baseline-far", type=int, default=511)
    parser.add_argument("--expected-documents", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args(argv)
    paths = sorted(
        {
            path.resolve()
            for item in args.records
            for path in (item.rglob("*.npz") if item.is_dir() else [item])
        }
    )
    report = summarize_ladder(
        [load_pair(path) for path in paths],
        donor_id=args.donor_id,
        fars=args.fars,
        baseline_far=args.baseline_far,
        expected_documents=args.expected_documents,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "coverage": report["coverage"],
                "lengths": [
                    {
                        "far": row["far"],
                        "effects": row["effects"],
                        "change_vs_baseline": row["change_vs_baseline"],
                    }
                    for row in report["lengths"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
