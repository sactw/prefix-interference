"""Readout of the Mamba-2 old-state attenuation experiment on the eight original reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .score_state_attenuation import difference, parity

ALPHAS = ("1", "0.5", "0")
METRICS = (
    "full",
    "foreign",
    "short",
    "N",
    "content",
    "prefix_penalty",
    "delta_full_vs_alpha1",
    "delta_foreign_vs_alpha1",
)
MODEL_KEYS = (
    "model_id",
    "revision",
    "tokenizer_id",
    "tokenizer_revision",
    "dtype",
    "backend",
    "versions",
    "torch_matmul_tf32",
    "triton_fp32_dot_precision",
)


def read(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files if key != "metadata_json"}
        metadata = json.loads(str(saved["metadata_json"].item()))
    return arrays, metadata


def summarize(folder, expected=8, expected_ids=None, samples=2000, seed=20261020):
    records, incomplete, warnings, excluded = {}, [], [], []
    for path in sorted(folder.glob("*.npz")):
        if not path.stem.endswith(("_full", "_foreign")):
            continue
        try:
            arrays, meta = read(path)
            doc = meta["document_id"]
            if expected_ids is not None and doc not in expected_ids:
                excluded.append(str(path))
                continue
            history = path.stem.rsplit("_", 1)[1]
            if meta["status"] != "completed":
                incomplete.append(
                    {"file": path.name, "status": meta["status"], "error": meta.get("error")}
                )
                continue
            key = (doc, history)
            if key in records:
                raise ValueError(f"duplicate completed history {key}")
            records[key] = (arrays, meta, path)
        except (ValueError, KeyError, OSError) as error:
            incomplete.append(
                {"file": path.name, "status": "unreadable_or_invalid", "error": str(error)}
            )

    docs = sorted(set(expected_ids or ()) | {doc for doc, _ in records})
    rows, per_document, cohort = [], [], None
    for doc in docs:
        missing = [key for key in ("full", "foreign") if (doc, key) not in records]
        if missing:
            incomplete.append({"document_id": doc, "missing_histories": missing})
            continue
        a, am, ap = records[(doc, "full")]
        b, bm, bp = records[(doc, "foreign")]
        try:
            if am["history"] != bm["history"] or am["intervention"] != bm["intervention"]:
                raise ValueError("paired histories/interventions differ")
            model = {key: am["model"][key] for key in MODEL_KEYS}
            if model != {key: bm["model"][key] for key in MODEL_KEYS}:
                raise ValueError("paired model settings differ")
            signature = {
                "model": model,
                "donor_id": am["history"]["donor_id"],
                "source_panel_sha256": am["history"]["source_panel_sha256"],
            }
            if cohort is not None and signature != cohort:
                raise ValueError("another model/donor/source panel must be analyzed separately")
            for key in (
                "target_ids",
                "target_positions",
                "reference_short_losses",
                "reference_full_losses",
                "reference_foreign_losses",
            ):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"paired concrete arrays differ: {key}")
            target_count = len(a["target_ids"])
            short = a["fresh_short_losses"].astype(np.float64)
            needed = [short, b["fresh_short_losses"]]
            needed += [
                values[f"{name}_alpha{alpha}_losses"]
                for values, name in ((a, "full"), (b, "foreign"))
                for alpha in ALPHAS
            ]
            if any(v.shape != (target_count,) or not np.isfinite(v).all() for v in needed):
                raise ValueError("missing, nonfinite or misaligned target losses")
            cohort = signature
            checks = {
                "full": parity(a["full_alpha1_losses"], a["reference_full_losses"]),
                "foreign": parity(b["foreign_alpha1_losses"], b["reference_foreign_losses"]),
                "fresh_short_full": parity(short, a["reference_short_losses"]),
                "fresh_short_foreign": parity(b["fresh_short_losses"], b["reference_short_losses"]),
            }
            for key, comparison in checks.items():
                if not comparison["within_provisional_tolerances"]:
                    warnings.append({"document_id": doc, "comparison": key, **comparison})
            matrix, alpha_rows = [], {}
            report_error = 0.0
            for alpha in ALPHAS:
                full = a[f"full_alpha{alpha}_losses"].astype(np.float64)
                foreign = b[f"foreign_alpha{alpha}_losses"].astype(np.float64)
                values = (
                    full,
                    foreign,
                    short,
                    short - full,
                    foreign - full,
                    foreign - short,
                    full - a["full_alpha1_losses"],
                    foreign - b["foreign_alpha1_losses"],
                )
                means = [float(value.mean()) for value in values]
                matrix.append(means)
                alpha_rows[alpha] = dict(zip(METRICS, means))
                for condition, meta, ce in (("full", am, means[0]), ("foreign", bm, means[1])):
                    reported = meta["measurements"][f"{condition}_alpha{alpha}"][
                        "ce_nats_per_token"
                    ]
                    report_error = max(report_error, abs(ce - reported))
            rows.append(matrix)
            measurements = list(am["measurements"].values()) + list(bm["measurements"].values())
            per_document.append(
                {
                    "document_id": doc,
                    "files": [ap.name, bp.name],
                    "target_tokens": target_count,
                    "alphas": alpha_rows,
                    "parity": checks,
                    "fresh_short_cross_job": difference(short, b["fresh_short_losses"]),
                    "raw_mean_vs_logged_ce_max_abs": report_error,
                    "intervention": am["intervention"],
                    "measured_seconds_including_load": sum(m["seconds"] for m in measurements)
                    + am["hardware"]["load_seconds"]
                    + bm["hardware"]["load_seconds"],
                    "peak_allocated_bytes": max(m["peak_allocated_bytes"] for m in measurements),
                    "peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in measurements),
                }
            )
        except (KeyError, ValueError) as error:
            incomplete.append({"document_id": doc, "status": "invalid_pair", "error": str(error)})

    aggregate = {}
    if rows:
        matrix = np.asarray(rows, dtype=np.float64)
        point = matrix.mean(axis=0)
        intervals = None
        if len(rows) > 1 and samples > 0:
            draws = np.random.default_rng(seed).integers(0, len(rows), size=(samples, len(rows)))
            intervals = np.quantile(matrix[draws].mean(axis=1), [0.025, 0.975], axis=0)
        for i, alpha in enumerate(ALPHAS):
            aggregate[alpha] = {
                metric: {
                    "mean": float(point[i, j]),
                    "ci95": None if intervals is None else intervals[:, i, j].tolist(),
                }
                for j, metric in enumerate(METRICS)
            }
    status = "complete_descriptive" if len(rows) == expected and not incomplete else "partial"
    return {
        "kind": "paired_old_state_attenuation",
        "status": status,
        "complete_documents": len(rows),
        "expected_documents": expected,
        "missing_document_count": max(0, expected - len(rows)),
        "targets": sum(row["target_tokens"] for row in per_document),
        "missing_failed_or_invalid": incomplete,
        "numerical_warnings": warnings,
        "excluded_files": excluded,
        "cohort": cohort,
        "aggregate_nats_per_native_token": aggregate,
        "analysis": {
            "weighting": "macro mean of per-document target means",
            "bootstrap_samples": samples,
            "seed": seed,
            "resampling_unit": "document",
            "paired": "same bootstrap multiplicities across every alpha, history and contrast",
            "intervals": "pointwise percentile 95%",
            "short": "fresh fused short from each full-history job; the foreign-job short is a numerical check",
            "parity": "mean absolute <=1e-4 and max absolute <=1e-3 (diagnostic thresholds)",
        },
        "measured_seconds_including_load": sum(
            row["measured_seconds_including_load"] for row in per_document
        ),
        "per_document": per_document,
        "limits": [
            "The IEA pair was observed before the test was extended to the other seven reports.",
            "One fixed donor and one checkpoint; document intervals are conditional, not donor-population or training-seed uncertainty.",
            "Alpha zero retains convolution caches and is not the fresh short history.",
            "Attenuation changes downstream activations; it localizes an old-state contribution without proving a finite-capacity mechanism.",
            "Improved CE does not establish recovered distant-content benefit; inspect content and net effects separately.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "state_attenuation" / "losses",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int, default=8)
    parser.add_argument(
        "--expected-plan", type=Path, help="optional matched-panel PLAN.json with cases/document_id"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    expected_ids = None
    if args.expected_plan:
        expected_ids = [
            case["document_id"]
            for case in json.loads(args.expected_plan.read_text(encoding="utf-8"))["cases"]
        ]
    report = summarize(
        args.input, args.expected_documents, expected_ids, args.bootstrap_samples, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "complete_documents": report["complete_documents"],
                "numerical_warnings": len(report["numerical_warnings"]),
            }
        )
    )


if __name__ == "__main__":
    main()
