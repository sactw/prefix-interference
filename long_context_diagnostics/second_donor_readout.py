"""Second-donor comparison and four-report replication (Mamba-2 and Gated DeltaNet)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .measurement import CONDITIONS, load_pair
from .matched_panel_readout import summarize_matched_panel


STUDY = Path(__file__).resolve().parents[1] / "results"
MODELS = ("mamba", "gdn")
SEED, BOOTSTRAPS = 20261020, 2000


def write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_completed(folder):
    by_document = {}
    for path in sorted(folder.glob("*.npz")):
        record = load_pair(path)
        name = record["metadata"]["history"]["model_name"]
        by_document.setdefault(record["document_id"], {})[name] = (path, record)
    complete = {doc: pair for doc, pair in by_document.items() if set(pair) == set(MODELS)}
    pending = {
        doc: sorted(set(MODELS) - set(pair))
        for doc, pair in by_document.items()
        if doc not in complete
    }
    return complete, pending


def totals(record):
    return np.array([np.asarray(record[name], dtype=np.float64).sum() for name in CONDITIONS])


def effects(values):
    full, short, foreign = values.T
    return np.stack((short - full, foreign - full, foreign - short), axis=-1)


def independent_panel_check(pairs, report):
    """Recompute means and percentile draws directly from saved loss arrays."""
    docs = sorted(pairs)
    sizes = np.array(
        [pairs[d]["mamba"][1]["metadata"]["history"]["target_utf8_bytes"] for d in docs]
    )
    draw = np.random.default_rng(SEED).integers(0, len(docs), (BOOTSTRAPS, len(docs)))
    warm_max = 0.0
    for doc in docs:
        first, second = [pairs[doc][name][1]["metadata"]["history"] for name in MODELS]
        for key in (
            "target_text_sha256",
            "source_char_spans",
            "target_utf8_bytes",
            "text_sha256",
            "donor_text_sha256",
        ):
            assert first[key] == second[key], (doc, key)
    for name in MODELS:
        values = np.stack([totals(pairs[d][name][1]) for d in docs])
        delta = effects(values)
        mean = delta.sum(axis=0) / sizes.sum()
        sampled = delta[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None]
        for i, key in enumerate(("N", "B", "P")):
            expected = report["models"][name]["primary_byte_weighted"][key]
            assert np.isclose(mean[i], expected["mean"], rtol=0, atol=1e-12)
            if len(docs) > 1:
                assert np.allclose(
                    np.quantile(sampled[:, i], [0.025, 0.975]), expected["ci95"], rtol=0, atol=1e-12
                )
        for doc in docs:
            record = pairs[doc][name][1]
            assert all(np.isfinite(record[key]).all() for key in CONDITIONS)
            for measurement in record["metadata"]["measurements"].values():
                warm_max = max(warm_max, measurement.get("repeat_max_abs", 0.0))
    return {
        "direct_loss_array_means_and_bootstrap_match": True,
        "paired_source_spans_hashes_bytes_equal": True,
        "all_target_losses_finite": True,
        "maximum_recorded_warm_repeat_error": warm_max,
        "documents": len(docs),
        "target_utf8_bytes": int(sizes.sum()),
    }


def panel_readout(name, expected_documents, *, study=STUDY, output=None):
    folder = study / name
    pairs, pending = read_completed(folder / "losses")
    if not pairs:
        return pairs, {"status": "awaiting_completed_pairs", "pending": pending}
    paths = [pairs[d][m][0] for d in sorted(pairs) for m in MODELS]
    records = [pairs[d][m][1] for d in sorted(pairs) for m in MODELS]
    report = summarize_matched_panel(records, bootstrap_samples=BOOTSTRAPS, seed=SEED)
    report["files"] = [p.relative_to(study).as_posix() for p in paths]
    report["coverage"] = {
        "expected_documents": expected_documents,
        "paired_documents": len(pairs),
        "pending_model_pairs": pending,
        "complete": len(pairs) == expected_documents and not pending,
    }
    report["independent_array_verification"] = independent_panel_check(pairs, report)
    if name == "replication":
        report["replication_scope"] = (
            "Four newly selected reports, kept separate from the original eight. Intervals over four documents have limited resolution."
        )
    write_json((output if output is not None else study) / name / "RESULTS.json", report)
    return pairs, report


def donor_comparison(original, alternate):
    """Resample documents once, carrying both donor evaluations together."""
    docs = sorted(alternate)
    sizes = np.array(
        [alternate[d]["mamba"][1]["metadata"]["history"]["target_utf8_bytes"] for d in docs]
    )
    draws = np.random.default_rng(SEED).integers(0, len(docs), (BOOTSTRAPS, len(docs)))
    report = {
        "kind": "paired_donor2_minus_donor1",
        "documents": len(docs),
        "expected_documents": 8,
        "complete": len(docs) == 8,
        "target_utf8_bytes": int(sizes.sum()),
        "aggregation": "sum paired target NLL differences / sum common target UTF8 bytes",
        "bootstrap": {
            "samples": BOOTSTRAPS,
            "seed": SEED,
            "unit": "target document",
            "scope": "Both donors and models remain in the same document draw; eight documents, not sixteen independent observations.",
            "confidence": 0.95,
            "pointwise": True,
        },
        "models": {},
        "per_document": [],
    }
    per_doc = {
        d: {"document_id": d, "target_utf8_bytes": int(sizes[i]), "models": {}}
        for i, d in enumerate(docs)
    }
    for name in MODELS:
        differences, maxima = [], {"full": 0.0, "short": 0.0}
        donor_ids = None
        for doc in docs:
            first, second = original[doc][name][1], alternate[doc][name][1]
            h1, h2 = first["metadata"]["history"], second["metadata"]["history"]
            for key in (
                "text_sha256",
                "target_text_sha256",
                "source_char_spans",
                "target_utf8_bytes",
                "reference_span",
                "model_reference",
                "tokenizer_json_sha256",
                "prefix_id",
            ):
                assert h1[key] == h2[key], (doc, name, key)
            np.testing.assert_array_equal(first["target_ids"], second["target_ids"])
            np.testing.assert_array_equal(first["target_positions"], second["target_positions"])
            for condition in ("full", "short"):
                assert (
                    h1["documentary_input_tokens"][condition]
                    == h2["documentary_input_tokens"][condition]
                )
                maxima[condition] = max(
                    maxima[condition],
                    float(
                        np.max(
                            np.abs(
                                np.asarray(second[condition], dtype=np.float64) - first[condition]
                            )
                        )
                    ),
                )
            donor_ids = [h1["donor_id"], h2["donor_id"]]
            delta = totals(second) - totals(first)
            differences.append(delta)
            all_delta = np.concatenate((delta, effects(delta[None])[0])) / h1["target_utf8_bytes"]
            per_doc[doc]["models"][name] = dict(
                zip((*CONDITIONS, "N", "B", "P"), all_delta.tolist())
            )
        delta_ce = np.stack(differences)
        values = np.concatenate((delta_ce, effects(delta_ce)), axis=1)
        mean = values.sum(axis=0) / sizes.sum()
        sampled = values[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
        report["models"][name] = {
            "donor_order": donor_ids,
            "target_ids_positions_and_source_spans_equal": True,
            "max_abs_cross_process_nll_difference_per_target": maxima,
            "paired_nats_per_utf8_byte": {
                key: {
                    "mean": float(mean[i]),
                    "ci95": np.quantile(sampled[:, i], [0.025, 0.975]).tolist()
                    if len(docs) > 1
                    else None,
                }
                for i, key in enumerate((*CONDITIONS, "N", "B", "P"))
            },
        }
    report["per_document"] = list(per_doc.values())
    report["interpretation"] = (
        "The foreign text changes; full/short inputs do not. Delta N estimates repeat execution drift, while delta B and delta P measure the paired sensitivity to this second fixed donor."
    )
    return report


def reproduce(study=STUDY, output=None):
    study = Path(study)
    output = study if output is None else Path(output)
    original, _ = read_completed(study / "matched_panel/losses")
    donor2, second = panel_readout("second_donor", 8, study=study, output=output)
    comparison = None
    if donor2:
        comparison = donor_comparison(original, donor2)
        write_json(output / "second_donor/DONOR_COMPARISON.json", comparison)
    fresh, fresh_report = panel_readout("replication", 4, study=study, output=output)
    return {"donor2": second, "donor_comparison": comparison, "replication4": fresh_report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, help="separate output root; defaults to study root")
    args = parser.parse_args(argv)
    reports = reproduce(args.study_root, args.output)
    second, fresh_report = reports["donor2"], reports["replication4"]
    for name, report in (("donor2", second), ("fresh_reports", fresh_report)):
        print(
            json.dumps(
                {
                    "panel": name,
                    "documents": report.get("documents", 0),
                    "coverage": report.get("coverage"),
                    "effects": {
                        model: {
                            key: values["primary_byte_weighted"][key] for key in ("N", "B", "P")
                        }
                        for model, values in report.get("models", {}).items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
