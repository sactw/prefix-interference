"""Paired document readout for model comparisons on identical target text.

Effects are total target NLL per target UTF-8 byte; native-token means are not
compared across models. Records may come from different code revisions.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .measurement import CONDITIONS, load_pair


MODELS = ("mamba", "gdn")
AVAILABLE_MODELS = (*MODELS, "rwkv")
METRICS = (*CONDITIONS, "N", "B", "P")
SUMMARY_KEYS = (*CONDITIONS, "N", "content", "prefix_penalty")
SHARED_FIELDS = (
    "text_sha256",
    "donor_text_sha256",
    "target_text_sha256",
    "source_char_spans",
    "requested_source_char_spans",
    "donor_id",
    "target_utf8_bytes",
    "target_characters",
    "foreign_policy",
    "normalization",
    "reference_span",
)


def _record_values(record):
    metadata = record["metadata"]
    history = metadata["history"]
    if history.get("protocol") != "matched_text_span_v1":
        raise ValueError("expected matched_text_span_v1; native-token panels are separate")
    name = history["model_name"]
    if name not in AVAILABLE_MODELS or history["document_id"] != record["document_id"]:
        raise ValueError("record model/document identity is inconsistent")
    for field in SHARED_FIELDS:
        if field not in history:
            raise ValueError(f"missing matched source field: {field}")
    size = history["target_utf8_bytes"]
    count = len(record["target_ids"])
    if not isinstance(size, int) or size <= 0 or count != history["target_tokens"]:
        raise ValueError("invalid byte or native-target denominator")
    target_span = history["source_char_spans"]["target"]
    if target_span[1] - target_span[0] != history["target_characters"]:
        raise ValueError("target character count differs from its source span")
    full_end = history["documentary_input_tokens"]["full"]
    if not np.array_equal(record["target_positions"], np.arange(full_end - count, full_end)):
        raise ValueError("native target positions differ from this model's full-input suffix")
    losses = [np.asarray(record[key], dtype=np.float64) for key in CONDITIONS]
    if any(x.shape != (count,) or not np.isfinite(x).all() or (x < 0).any() for x in losses):
        raise ValueError("expected finite nonnegative native-target losses")
    full, short, foreign = (float(x.sum()) for x in losses)
    totals = np.array([full, short, foreign, short - full, foreign - full, foreign - short])
    saved = metadata["matched_text"]
    expected_counts = {
        "tokens": count,
        "characters": history["target_characters"],
        "utf8_bytes": size,
    }
    if saved["counts"] != expected_counts:
        raise ValueError("saved matched-text denominators differ from source metadata")
    for key, value in zip(SUMMARY_KEYS, totals):
        if not np.isclose(saved["total_nll_nats"][key], value, rtol=1e-10, atol=1e-8):
            raise ValueError("saved total NLL differs from per-target loss arrays")
        if not np.isclose(saved["nats_per_utf8_byte"][key], value / size, rtol=1e-10, atol=1e-8):
            raise ValueError("saved byte-normalized NLL differs from loss arrays")
    return name, history, totals, size


def _configuration(record):
    """Model and precision settings that must match within a panel."""
    metadata, history = record["metadata"], record["metadata"]["history"]
    model, reference = metadata["model"], history["model_reference"]
    for key in ("model_id", "revision", "tokenizer_id", "tokenizer_revision"):
        if model[key] != reference[key]:
            raise ValueError(f"measured model and plan disagree on {key}")
    precision_keys = (
        "dtype",
        "loss_dtype",
        "backend",
        "torch_matmul_tf32",
        "torch_cudnn_tf32",
        "triton_fp32_dot_precision",
        "triton_fp32_dot_precision_requested",
    )
    solve = model.get("fla_fused_solve_precision")
    return {
        "model_reference": reference,
        "tokenizer_json_sha256": history["tokenizer_json_sha256"],
        "precision": {key: model.get(key) for key in precision_keys},
        "fla_solve_effective": solve.get("effective") if isinstance(solve, dict) else solve,
        "boundary_token": history["prefix_id"],
        "execution": metadata.get("selected_execution"),
    }


def _estimates(means, draws, *, independent_documents, metrics=METRICS):
    return {
        key: {
            "mean": float(means[j]),
            "ci95": [float(x) for x in np.quantile(draws[:, j], [0.025, 0.975])]
            if independent_documents > 1
            else None,
        }
        for j, key in enumerate(metrics)
    }


def summarize_matched_panel(records, *, bootstrap_samples=2000, seed=20261020, models=MODELS):
    """Pair models by source text and resample identical document multiplicities."""
    if not records or bootstrap_samples < 1:
        raise ValueError("need paired records and positive bootstrap_samples")
    requested = tuple(models)
    if (
        len(requested) < 2
        or len(set(requested)) != len(requested)
        or not set(requested) <= set(AVAILABLE_MODELS)
    ):
        raise ValueError("models must name at least two distinct available models")
    # A stable direction for each contrast, independent of CLI argument order.
    model_names = tuple(name for name in AVAILABLE_MODELS if name in requested)
    by_document, configs = {}, {}
    for record in records:
        name, history, totals, size = _record_values(record)
        doc_id = record["document_id"]
        pair = by_document.setdefault(doc_id, {})
        if name in pair:
            raise ValueError(f"duplicate model/document record: {name}/{doc_id}")
        configuration = _configuration(record)
        if name in configs and configs[name] != configuration:
            raise ValueError(
                f"mixed model revision/precision settings for {name}; report separately"
            )
        configs[name] = configuration
        pair[name] = {
            "history": history,
            "totals": totals,
            "bytes": size,
            "tokens": len(record["target_ids"]),
        }
    for doc_id, pair in by_document.items():
        if set(pair) != set(model_names):
            raise ValueError(f"missing matched model for {doc_id}; missing results are not imputed")
        first = pair[model_names[0]]["history"]
        for name in model_names[1:]:
            for key in SHARED_FIELDS:
                if first[key] != pair[name]["history"][key]:
                    raise ValueError(f"model pair has different {key}: {doc_id}")
    documents = sorted(by_document)
    count = len(documents)
    byte_counts = np.array(
        [by_document[doc][model_names[0]]["bytes"] for doc in documents], dtype=np.float64
    )
    totals = {
        name: np.stack([by_document[doc][name]["totals"] for doc in documents])
        for name in model_names
    }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, count, size=(bootstrap_samples, count))
    multiplicities = np.stack([np.bincount(row, minlength=count) for row in sampled])
    denominators = multiplicities @ byte_counts
    means = {name: totals[name].sum(axis=0) / byte_counts.sum() for name in model_names}
    draws = {name: (multiplicities @ totals[name]) / denominators[:, None] for name in model_names}
    document_rates = {name: totals[name] / byte_counts[:, None] for name in model_names}
    macro_draws = {name: (multiplicities @ document_rates[name]) / count for name in model_names}
    model_reports = {
        name: {
            "configuration": configs[name],
            "native_target_tokens": sum(by_document[doc][name]["tokens"] for doc in documents),
            "primary_byte_weighted": _estimates(
                means[name], draws[name], independent_documents=count
            ),
            "secondary_equal_document": _estimates(
                document_rates[name].mean(axis=0), macro_draws[name], independent_documents=count
            ),
        }
        for name in model_names
    }
    differences, document_differences = {}, {}
    for first, second in combinations(model_names, 2):
        direction = f"{second}_minus_{first}"
        delta = means[second] - means[first]
        delta_draws = draws[second] - draws[first]
        macro_delta = document_rates[second] - document_rates[first]
        document_differences[direction] = macro_delta[:, 3:]
        differences[direction] = {
            "direction": direction,
            "scope": "paired within-model history effects, not architecture ranking",
            "primary_byte_weighted": _estimates(
                delta[3:], delta_draws[:, 3:], independent_documents=count, metrics=("N", "B", "P")
            ),
            "secondary_equal_document": _estimates(
                macro_delta.mean(axis=0)[3:],
                (macro_draws[second] - macro_draws[first])[:, 3:],
                independent_documents=count,
                metrics=("N", "B", "P"),
            ),
        }
    # Keep the original two-model fields unchanged, including in an expanded panel.
    primary_direction = (
        "gdn_minus_mamba" if "gdn_minus_mamba" in differences else next(iter(differences))
    )
    rows = []
    for i, doc in enumerate(documents):
        history = by_document[doc][model_names[0]]["history"]
        row = {
            "document_id": doc,
            "donor_id": history["donor_id"],
            "target_utf8_bytes": int(byte_counts[i]),
            "target_text_sha256": history["target_text_sha256"],
            "source_char_spans": history["source_char_spans"],
            "nats_per_utf8_byte": {
                name: dict(zip(METRICS, document_rates[name][i].tolist())) for name in model_names
            },
            f"{primary_direction}_effects": dict(
                zip(("N", "B", "P"), document_differences[primary_direction][i].tolist())
            ),
            "native_lengths": {
                name: by_document[doc][name]["history"]["documentary_input_tokens"]
                for name in model_names
            },
        }
        if len(model_names) > 2:
            row["pairwise_model_effects"] = {
                key: dict(zip(("N", "B", "P"), values[i].tolist()))
                for key, values in document_differences.items()
            }
        rows.append(row)
    report = {
        "schema_version": 1,
        "kind": "matched_text_model_panel",
        "documents": count,
        "records": len(records),
        "target_utf8_bytes": int(byte_counts.sum()),
        "definitions": {"N": "short-full", "B": "foreign-full", "P": "foreign-short"},
        "aggregation": "sum target NLL / sum identical target UTF8 bytes; equal-document secondary",
        "bootstrap": {
            "method": "paired_document_cluster_percentile_ratio",
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence": 0.95,
            "note": "Pointwise intervals conditional on selected texts/donors; no population interval for one document.",
        },
        "models": model_reports,
        "paired_model_difference": differences[primary_direction],
        "per_document": rows,
        "limitations": [
            "Canonical native-token likelihood per source byte is not a marginal over tokenizations.",
            "Native conditioning lengths differ; this character-matched panel is separate from native-token panels.",
            "The checkpoints differ in size and training; paired differences do not establish an architectural cause.",
            "Uncertainty resamples target documents with fixed donor assignments, not training seeds or donor populations.",
        ],
    }
    if len(model_names) > 2:
        report["pairwise_model_differences"] = differences
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS, default=MODELS)
    args = parser.parse_args(argv)
    report = summarize_matched_panel(
        [load_pair(path) for path in args.files],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        models=args.models,
    )
    report["files"] = [str(path) for path in args.files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "documents": report["documents"],
                "models": {
                    key: value["primary_byte_weighted"] for key, value in report["models"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
