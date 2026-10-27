"""CPU readout of multi-donor panels; donor repeats are not extra targets.

Each model/E/C/T panel is reported separately. Average each target's losses
over its observed assigned donors, then bootstrap documents with multiplicity.
Intervals condition on those donors; donor sensitivity is reported separately.
Different E values score different targets and are never a paired length curve.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .measurement import CONDITIONS, load_pair
from .readout import EFFECTS, summarize


MODEL_FIELDS = (
    "model_id",
    "revision",
    "tokenizer_id",
    "tokenizer_revision",
    "dtype",
    "backend",
    "triton_fp32_dot_precision",
    "triton_fp32_dot_precision_requested",
    "fla_fused_solve_precision",
)


def panel_identity(row):
    """Model and history settings that define a panel; code revisions are ignored."""
    metadata = row["metadata"]
    history, model = metadata["history"], metadata["model"]
    return {
        "model": {key: model.get(key) for key in MODEL_FIELDS},
        "execution": metadata.get("selected_execution", metadata.get("execution")),
        "end": int(history["end_exclusive"]),
        "recent": int(history["recent_prefix_tokens"]),
        "targets_per_block": int(history["target_tokens"]),
        "boundary_token": history.get("prefix_id"),
        "full_history_start": int(history.get("full_history_start", 0)),
        "repeat_label_scope": history.get("repeat_label_scope", "full_document_prefix"),
    }


def _sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _means(row, mask):
    if not np.any(mask):
        return {key: None for key in EFFECTS}
    full, short, foreign = (np.asarray(row[key], dtype=np.float64)[mask] for key in CONDITIONS)
    return dict(
        zip(
            EFFECTS,
            map(float, (np.mean(short - full), np.mean(foreign - full), np.mean(foreign - short))),
        )
    )


def average_donors(rows):
    """Average donor records per document without multiplying target counts.

    Full and short losses can differ slightly between repeated runs; they are
    averaged and their spread is reported. Records with different targets or
    text raise an error.
    """
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    averaged, details = [], {}
    for document_id in sorted(by_document):
        group = sorted(by_document[document_id], key=lambda row: row["metadata"]["donor_id"])
        first = group[0]
        donor_ids = [row["metadata"]["donor_id"] for row in group]
        if len(donor_ids) != len(set(donor_ids)):
            raise ValueError(f"{document_id}: duplicate donor record; select one attempt per cell")
        for row in group[1:]:
            for key in ("target_ids", "target_positions"):
                if not np.array_equal(first[key], row[key]):
                    raise ValueError(f"{document_id}: donor records have different {key}")
            if set(first["repeat_bins"]) != set(row["repeat_bins"]) or any(
                not np.array_equal(labels, row["repeat_bins"][n])
                for n, labels in first["repeat_bins"].items()
            ):
                raise ValueError(f"{document_id}: donor records have different repeat labels")
            if first["metadata"].get("text_sha256") != row["metadata"].get("text_sha256"):
                raise ValueError(f"{document_id}: donor records describe different target text")
        block = {
            key: first[key]
            for key in ("document_id", "target_ids", "target_positions", "repeat_bins")
        }
        block.update(
            {
                key: np.mean([np.asarray(row[key], dtype=np.float64) for row in group], axis=0)
                for key in CONDITIONS
            }
        )
        averaged.append(block)
        details[document_id] = {
            "donor_ids": donor_ids,
            "targets": len(first["target_ids"]),
            "reference_repeat_spread": {
                key: {
                    "maximum_per_target_range": float(
                        np.ptp(
                            np.stack([row[key] for row in group]).astype(np.float64), axis=0
                        ).max()
                    ),
                    "mean_ce_range": float(
                        np.ptp([np.mean(row[key], dtype=np.float64) for row in group])
                    ),
                }
                for key in ("full", "short")
            },
        }
    return averaged, by_document, details


def _masks(row):
    yield "overall", np.ones(len(row["target_ids"]), dtype=bool)
    for n, labels in sorted(row["repeat_bins"].items()):
        labels = np.asarray(labels, dtype=str)
        yield f"far{n}", np.char.startswith(labels, "far:")
        for label in ("never", "recent", "boundary", "incomplete_ngram"):
            yield f"{label}{n}", labels == label


def donor_sensitivity(averaged, by_document):
    """Range of mean effects over every choice of one donor per document.

    This is a sensitivity range, not a confidence interval; donors are not
    treated as independent documents.
    """
    per_document, accumulators = {}, {}
    for row in averaged:
        document_id = row["document_id"]
        per_document[document_id] = {}
        for group_name, mask in _masks(row):
            count = int(mask.sum())
            values = {
                item["metadata"]["donor_id"]: _means(item, mask)
                for item in by_document[document_id]
            }
            entry = {"targets": count, "per_donor": values, "effects": {}}
            acc = accumulators.setdefault(
                group_name,
                {
                    "targets": 0,
                    "documents": 0,
                    "multiple_donor_documents": 0,
                    "sums": {key: np.zeros(2) for key in EFFECTS},
                },
            )
            acc["targets"] += count
            acc["documents"] += int(count > 0)
            acc["multiple_donor_documents"] += int(count > 0 and len(values) > 1)
            for effect in EFFECTS:
                observed = [item[effect] for item in values.values() if item[effect] is not None]
                bounds = [min(observed), max(observed)] if observed else None
                entry["effects"][effect] = {
                    "mean_range": bounds,
                    "sign_stable": len({_sign(value) for value in observed}) == 1
                    if observed and len(values) > 1
                    else None,
                }
                if bounds is not None:
                    acc["sums"][effect] += np.asarray(bounds) * count
            per_document[document_id][group_name] = entry
    panel = {}
    for name, acc in accumulators.items():
        count = acc["targets"]
        panel[name] = {
            key: acc[key] for key in ("targets", "documents", "multiple_donor_documents")
        }
        panel[name]["effects"] = {}
        for key, sums in acc["sums"].items():
            bounds = (sums / count).tolist() if count else None
            panel[name]["effects"][key] = {
                "mean_range": bounds,
                "sign_stable": (_sign(bounds[0]) == _sign(bounds[1]))
                if bounds and acc["multiple_donor_documents"]
                else None,
            }
    return {
        "scope": "observed donor alternatives; not confidence intervals",
        "panel": panel,
        "per_document": per_document,
    }


def summarize_panel(
    rows, bootstrap_samples=2000, seed=20261020, expected_documents=None, expected_donors=None
):
    """Summarize each observed panel and report its coverage."""
    panels = defaultdict(list)
    for row in rows:
        identity = panel_identity(row)
        panels[json.dumps(identity, sort_keys=True)].append(row)
    output = {
        "schema_version": 1,
        "kind": "public_model_multi_donor_panel",
        "note": "Separate E/C/T panels; E changes targets. Document bootstrap conditions on assigned donors.",
        "panels": [],
    }
    for key, panel_rows in sorted(panels.items()):
        identity = json.loads(key)
        for row in panel_rows:
            positions = np.asarray(row["target_positions"])
            expected = np.arange(identity["end"] - identity["targets_per_block"], identity["end"])
            if not np.array_equal(positions, expected):
                raise ValueError(
                    f"{row['document_id']}: target positions do not match E/T metadata"
                )
        averaged, by_document, details = average_donors(panel_rows)
        report = summarize(averaged, bootstrap_samples, seed)
        report["configuration"] = identity
        report["coverage"] = {
            "records": len(panel_rows),
            "unique_target_documents": len(averaged),
            "expected_documents": expected_documents,
            "expected_donors_per_document": expected_donors,
            "donors_per_document": {key: len(value["donor_ids"]) for key, value in details.items()},
            "complete_counts": (
                len(averaged) == expected_documents
                and all(len(value["donor_ids"]) == expected_donors for value in details.values())
            )
            if expected_documents is not None and expected_donors is not None
            else None,
            "note": "Observed records only; missing results are not imputed.",
        }
        report["donor_aggregation"] = (
            "equal donor average within each document/target, then target-weighted document bootstrap"
        )
        report["documents_detail"] = details
        # Collapse far-distance bins but keep every target and document, so
        # documents without far repeats stay in the bootstrap population.
        collapsed = [
            {
                **row,
                "repeat_bins": {
                    n: np.where(
                        np.char.startswith(np.asarray(labels, dtype=str), "far:"), "far", labels
                    )
                    for n, labels in row["repeat_bins"].items()
                },
            }
            for row in averaged
        ]
        coarse = summarize(collapsed, bootstrap_samples, seed)
        report["groups"] = {}
        for n, bins in coarse["repeat_bins"].items():
            for label in ("far", "never", "recent", "boundary", "incomplete_ngram"):
                # A wholly absent far group has the same empty statistics as
                # an unobserved fine far bin, already retained by summarize().
                value = bins.get(label, bins["far:(0,1024]"])
                report["groups"][f"{label}{n}"] = {**value, "bootstrap": coarse["bootstrap"]}
        report["donor_sensitivity"] = donor_sensitivity(averaged, by_document)
        output["panels"].append(report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records", type=Path, nargs="+", help="NPZ files or directories searched recursively"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--expected-documents", type=int)
    parser.add_argument("--expected-donors", type=int)
    args = parser.parse_args(argv)
    paths = sorted(
        {
            path.resolve()
            for item in args.records
            for path in (item.rglob("*.npz") if item.is_dir() else [item])
        }
    )
    result = summarize_panel(
        [load_pair(path) for path in paths],
        args.bootstrap_samples,
        args.seed,
        args.expected_documents,
        args.expected_donors,
    )
    result["files"] = [str(path) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "files": len(paths),
                "panels": [
                    {
                        "configuration": panel["configuration"],
                        "coverage": panel["coverage"],
                        "effects": panel["effects"],
                    }
                    for panel in result["panels"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
