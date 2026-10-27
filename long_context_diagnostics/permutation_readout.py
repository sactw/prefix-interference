"""Paired original/permuted-prefix effects, conditional on one fixed shuffle.

Overall intervals resample documents with common draws for both variants and
their changes. Original far-repeat strata are descriptive counts and means; their
membership is fixed by the original labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .measurement import CONDITIONS, load_pair, validate_arms
from .panel_readout import panel_identity


VARIANTS = ("original", "permuted")
EFFECTS = ("N", "B", "P")
PERMUTATION_FIELDS = (
    "name",
    "seed",
    "rng",
    "block_tokens",
    "far_tokens",
    "blocks",
    "shuffled_blocks",
    "keep_tail_blocks",
    "anchored_tail_tokens",
    "order",
    "same_order_for_actual_and_foreign",
)


def _effects(ce):
    return np.stack((ce[:, 1] - ce[:, 0], ce[:, 2] - ce[:, 0], ce[:, 2] - ce[:, 1]), axis=1)


def _estimate(values, draws):
    result = {"mean": float(values.mean()) if len(values) else None}
    if draws is not None:
        result["ci95"] = (
            np.percentile(values[draws].mean(axis=1), [2.5, 97.5]).tolist()
            if len(values) and len(draws)
            else None
        )
    return result


def _comparison(original, permuted, draws):
    before, after = _effects(original), _effects(permuted)
    return {
        "original": {
            "ce": {k: _estimate(original[:, i], draws) for i, k in enumerate(CONDITIONS)},
            "effects": {k: _estimate(before[:, i], draws) for i, k in enumerate(EFFECTS)},
        },
        "permuted": {
            "ce": {k: _estimate(permuted[:, i], draws) for i, k in enumerate(CONDITIONS)},
            "effects": {k: _estimate(after[:, i], draws) for i, k in enumerate(EFFECTS)},
        },
        "delta_ce": {
            k: _estimate(permuted[:, i] - original[:, i], draws) for i, k in enumerate(CONDITIONS)
        },
        "delta_effects": {
            k: _estimate(after[:, i] - before[:, i], draws) for i, k in enumerate(EFFECTS)
        },
    }


def _signature(row):
    info = row["metadata"]["intervention"]
    return {key: info[key] for key in PERMUTATION_FIELDS}


def _validate_pair(doc, original, permuted):
    if panel_identity(original) != panel_identity(permuted):
        raise ValueError(f"{doc}: model or history configuration differs")
    for key in ("target_ids", "target_positions"):
        if not np.array_equal(original[key], permuted[key]):
            raise ValueError(f"{doc}: {key} differ")
    for key in ("paired_attempt", "text_sha256", "donor_text_sha256"):
        if original["metadata"].get(key) != permuted["metadata"].get(key):
            raise ValueError(f"{doc}: {key} differs")
    if _signature(original) != _signature(permuted):
        raise ValueError(f"{doc}: permutation differs between paired records")
    if set(original["repeat_bins"]) != set(permuted["repeat_bins"]) or any(
        not np.array_equal(labels, permuted["repeat_bins"][n])
        for n, labels in original["repeat_bins"].items()
    ):
        raise ValueError(f"{doc}: saved original repeat labels differ")
    for row in (original, permuted):
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
        info = row["metadata"]["intervention"]
        if not info["same_order_for_actual_and_foreign"] or not info["short_input_equal"]:
            raise ValueError(f"{doc}: recorded prefix/short matching checks failed")
        for condition in ("full", "foreign"):
            checks = info["verification"][condition]
            if not all(
                checks.get(key) is True
                for key in (
                    "token_multiset_equal",
                    "boundary_equal",
                    "recent_and_targets_equal",
                    "anchored_tail_equal",
                )
            ):
                raise ValueError(f"{doc}: recorded {condition} input matching checks failed")


def summarize_permutation(
    rows, *, donor_id=None, bootstrap_samples=2000, seed=20261020, expected_documents=8
):
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    donors = sorted({row["metadata"]["donor_id"] for row in rows})
    if donor_id is None:
        if len(donors) != 1:
            raise ValueError("select one donor with --donor-id")
        donor_id = donors[0]
    selected = [row for row in rows if row["metadata"]["donor_id"] == donor_id]
    cells = {}
    for row in selected:
        variant = row["metadata"]["intervention"]["variant"]
        doc = row["document_id"]
        if variant not in VARIANTS:
            raise ValueError(f"{doc}: unknown variant {variant}")
        if (doc, variant) in cells:
            raise ValueError(f"{doc}/{variant}: duplicate attempt; select one paired attempt")
        cells[doc, variant] = row
    observed = sorted({doc for doc, _ in cells})
    documents = [doc for doc in observed if all((doc, v) in cells for v in VARIANTS)]
    identity, permutation = None, None
    for doc in documents:
        original, permuted = (cells[doc, v] for v in VARIANTS)
        _validate_pair(doc, original, permuted)
        current, shuffle = panel_identity(original), _signature(original)
        if identity is not None and (current != identity or shuffle != permutation):
            raise ValueError(f"{doc}: panel model/history or fixed permutation differs")
        identity, permutation = current, shuffle
    ce = {
        v: np.asarray(
            [[np.mean(cells[doc, v][k], dtype=np.float64) for k in CONDITIONS] for doc in documents]
        ).reshape(-1, 3)
        for v in VARIANTS
    }
    draws = (
        np.random.default_rng(seed).integers(
            0, len(documents), size=(bootstrap_samples, len(documents))
        )
        if documents
        else np.empty((0, 0), dtype=int)
    )
    result = {
        "schema_version": 1,
        "kind": "paired_old_prefix_permutation",
        "configuration": identity,
        "permutation": permutation,
        "donor_id": donor_id,
        "aggregation": "macro_document_mean_of_per_target_nats",
        "definitions": {
            "N": "short-full",
            "B": "foreign-full",
            "P": "foreign-short",
            "delta": "permuted-original; positive delta CE is worse",
        },
        "bootstrap": {
            "method": "paired_document_percentile",
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence": 0.95,
            "same_draws_all_estimates": True,
            "scope": "pointwise; conditional on these texts, donor assignment and one permutation",
        },
        "coverage": {
            "expected_documents": expected_documents,
            "paired_documents": len(documents),
            "selected_records": len(selected),
            "excluded_other_donor_records": len(rows) - len(selected),
            "missing_variants": {
                d: [v for v in VARIANTS if (d, v) not in cells]
                for d in observed
                if d not in documents
            },
            "complete_counts": len(documents) == expected_documents
            if expected_documents is not None
            else None,
            "note": "Only complete original/permuted pairs contribute; no missing cells imputed.",
        },
        "distinct_targets": sum(len(cells[d, "original"]["target_ids"]) for d in documents),
        "identity_checks": {
            "target_ids_and_positions_equal": bool(documents),
            "original_repeat_labels_preserved": bool(documents),
            "recorded_input_multisets_boundary_recent_targets_and_tail_equal": bool(documents),
        },
        **_comparison(ce["original"], ce["permuted"], draws),
        "per_document": [],
        "original_far_strata": {},
    }
    short_spreads = []
    for i, doc in enumerate(documents):
        original, permuted = (cells[doc, v] for v in VARIANTS)
        spread = float(np.max(np.abs(original["short"].astype(np.float64) - permuted["short"])))
        short_spreads.append(spread)
        survival = original["metadata"]["intervention"].get("far_repeat_survival", {})
        result["per_document"].append(
            {
                "document_id": doc,
                "paired_attempt": original["metadata"].get("paired_attempt"),
                "targets": len(original["target_ids"]),
                "max_abs_short_target_difference": spread,
                "repeat_survival": {
                    n: {
                        k: value
                        for k, value in item.items()
                        if k
                        in (
                            "original_far_targets",
                            "original_far_still_matching_permuted_far",
                            "original_far_labels_after_permutation",
                        )
                    }
                    for n, item in survival.items()
                },
                **_comparison(ce["original"][i : i + 1], ce["permuted"][i : i + 1], None),
            }
        )
    result["max_abs_short_target_difference"] = max(short_spreads) if short_spreads else None
    for n in ("4", "8"):
        if not documents or any(
            n not in cells[doc, "original"]["repeat_bins"] for doc in documents
        ):
            result["original_far_strata"][f"far{n}"] = {"status": "labels_unavailable"}
            continue
        masks = {
            doc: np.char.startswith(cells[doc, "original"]["repeat_bins"][n].astype(str), "far:")
            for doc in documents
        }
        contributing = [doc for doc in documents if np.any(masks[doc])]
        group = {
            v: np.asarray(
                [
                    [np.mean(cells[doc, v][k][masks[doc]], dtype=np.float64) for k in CONDITIONS]
                    for doc in contributing
                ]
            ).reshape(-1, 3)
            for v in VARIANTS
        }
        result["original_far_strata"][f"far{n}"] = {
            "status": "descriptive_only",
            "documents": len(contributing),
            "targets": sum(int(mask.sum()) for mask in masks.values()),
            "targets_per_document": {doc: int(mask.sum()) for doc, mask in masks.items()},
            "aggregation": "macro over within-stratum document means; zero-target documents excluded",
            "note": "Original labels fix membership; sparse exploratory means, no confidence intervals.",
            **_comparison(group["original"], group["permuted"], None),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records", nargs="+", type=Path, help="NPZ files or recursively searched directories"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--donor-id")
    parser.add_argument("--expected-documents", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    paths = sorted(
        {
            p.resolve()
            for item in args.records
            for p in (item.rglob("*.npz") if item.is_dir() else [item])
        }
    )
    report = summarize_permutation(
        [load_pair(p) for p in paths],
        donor_id=args.donor_id,
        expected_documents=args.expected_documents,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "coverage",
                    "delta_ce",
                    "delta_effects",
                    "max_abs_short_target_difference",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
