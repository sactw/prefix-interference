"""Compare fused and recurrent Mamba-2 likelihoods, timing and memory usage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .__main__ import write_report
from .mamba_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DEFAULT_TOKENIZER_REVISION,
    MambaAdapter,
)
from .measurement import CONDITIONS, build_histories, save_pair
from .readout import classify_repeats


def timed(adapter, method, ids, start, chunk_size):
    torch = adapter._torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    losses = method(ids, start, chunk_size=chunk_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    return losses, {
        "seconds": elapsed,
        "input_tokens": len(ids),
        "processed_tokens": len(ids) - 1,
        "scored_targets": len(losses),
        "processed_tokens_per_second": (len(ids) - 1) / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "reserved_memory_note": "Includes allocator cache retained from earlier modes in this process.",
    }


def record_path(output, row, donor, args):
    """Return a deterministic filename from model, text and context settings."""
    science = {
        "document_id": row["document_id"],
        "text": row["text"],
        "donor_id": donor["document_id"],
        "donor_text": donor["text"],
        "model": args.model_id,
        "weights": args.revision,
        "tokenizer": args.tokenizer_revision,
        "E": args.end,
        "C": args.recent,
        "T": args.targets,
    }
    suffix = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()[:20]
    return output / f"block-{suffix}.npz"


def compare_modes(full, recurrent, mean_tolerance, max_tolerance):
    """Compare token losses and mean paired effects against supplied tolerances."""
    errors = {}
    for name in CONDITIONS:
        delta = np.asarray(recurrent[name], dtype=np.float64) - full[name]
        errors[name] = {
            "mean_abs": float(np.abs(delta).mean()),
            "max_abs": float(np.abs(delta).max()),
        }

    def effects(values):
        return {
            "N": values["short"].astype(np.float64) - values["full"],
            "content": values["foreign"].astype(np.float64) - values["full"],
            "prefix_penalty": values["foreign"].astype(np.float64) - values["short"],
        }

    a, b = effects(full), effects(recurrent)
    effect_errors = {key: float(abs(a[key].mean() - b[key].mean())) for key in a}
    within = all(
        v["mean_abs"] <= mean_tolerance and v["max_abs"] <= max_tolerance for v in errors.values()
    ) and all(v <= mean_tolerance for v in effect_errors.values())
    return {
        "per_condition_nll": errors,
        "mean_effect_absolute_difference": effect_errors,
        "within_provisional_tolerances": within,
        "tolerances": {"mean_abs_nll_and_effect": mean_tolerance, "max_abs_nll": max_tolerance},
        "note": "Engineering diagnostic; these tolerances are provisional, not a scientific significance test.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents", type=Path, required=True, help="prepared natural-text JSONL; see README"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--end", type=int, default=2048)
    parser.add_argument("--recent", type=int, default=512)
    parser.add_argument("--targets", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="optional PyTorch allocator budget in decimal GB",
    )
    parser.add_argument(
        "--document-id", action="append", help="retry only these documents; completed blocks reused"
    )
    parser.add_argument(
        "--code-revision", default="unrecorded", help="source revision recorded in result metadata"
    )
    parser.add_argument("--mean-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-tolerance", type=float, default=1e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.documents.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["document_id"]: row for row in rows}
    if len(by_id) != len(rows):
        parser.error("document_id must be unique in the input JSONL")
    all_targets = [row for row in rows if row["role"] == "target"]
    targets = [
        row for row in all_targets if not args.document_id or row["document_id"] in args.document_id
    ]
    if not targets:
        parser.error("no selected target documents")
    for row in all_targets:
        donor = by_id.get(row.get("donor_id"))
        if donor is None or donor["role"] != "donor" or donor["document_id"] == row["document_id"]:
            parser.error("each target needs an explicitly separate donor-role document")
        if row["text"] == donor["text"]:
            parser.error("donor text is identical to target text")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        row["document_id"]: record_path(args.output, row, by_id[row["donor_id"]], args)
        for row in all_targets
    }
    pending = [row for row in targets if not paths[row["document_id"]].exists()]
    for row in targets:
        if row not in pending:
            print(
                f"Reusing completed block for {row['document_id']}; its original metadata is retained.",
                flush=True,
            )
    if not pending:
        write_report(
            [path for path in paths.values() if path.exists()],
            args.output / "summary.json",
            args.bootstrap_samples,
            args.seed,
        )
        return

    total_start = time.perf_counter()
    print(json.dumps({"event": "model_load_started", "model": args.model_id}), flush=True)
    adapter = MambaAdapter(
        args.model_id, args.revision, args.tokenizer_revision, max_memory_gb=args.max_memory_gb
    )
    adapter._torch.cuda.synchronize()
    load_seconds = time.perf_counter() - total_start
    hardware = {
        "gpu": adapter._torch.cuda.get_device_name(),
        "load_seconds": load_seconds,
        "code_revision": args.code_revision,
    }
    # Each document is tokenized once and shared by all interventions.
    tokens = {
        key: np.asarray(adapter.encode(row["text"]), dtype=np.int64) for key, row in by_id.items()
    }
    print(
        json.dumps(
            {
                "event": "tokenized",
                "load_seconds": load_seconds,
                "document_tokens": {key: len(value) for key, value in tokens.items()},
            }
        ),
        flush=True,
    )
    selected = None
    comparison = None
    first_plan = build_histories(
        tokens[pending[0]["document_id"]],
        tokens[pending[0]["donor_id"]],
        end=args.end,
        recent_tokens=args.recent,
        target_tokens=args.targets,
        prefix_id=adapter.prefix_id,
    )
    warm_ids = first_plan.inputs["short"][: min(65, len(first_plan.inputs["short"]))]
    cold_setup = {}
    for name, method in (("full", adapter.score_full),):
        _, cold_setup[name] = timed(adapter, method, warm_ids, 1, min(args.chunk_size, 16))
    hardware["cold_kernel_setup"] = cold_setup
    mode_methods = {"full": adapter.score_full, "recurrent": adapter.score}

    for row in pending:
        doc_id = row["document_id"]
        ids = tokens[doc_id]
        plan = build_histories(
            ids,
            tokens[row["donor_id"]],
            end=args.end,
            recent_tokens=args.recent,
            target_tokens=args.targets,
            prefix_id=adapter.prefix_id,
        )
        bins = {
            str(n): classify_repeats(ids, plan.target_positions, plan.retained_start, n=n)
            for n in (4, 8)
        }
        measurements = {}
        losses_by_mode = {}
        for mode in ("full", "recurrent") if selected is None else (selected,):
            measurements[mode], losses_by_mode[mode] = {}, {}
            try:
                for condition in CONDITIONS:
                    print(
                        json.dumps(
                            {
                                "event": "score_started",
                                "document_id": doc_id,
                                "mode": mode,
                                "condition": condition,
                            }
                        ),
                        flush=True,
                    )
                    if selected is None:
                        _, first_pass = timed(
                            adapter,
                            mode_methods[mode],
                            plan.inputs[condition],
                            plan.target_starts[condition],
                            args.chunk_size,
                        )
                    loss, metric = timed(
                        adapter,
                        mode_methods[mode],
                        plan.inputs[condition],
                        plan.target_starts[condition],
                        args.chunk_size,
                    )
                    if selected is None:
                        metric["first_pass"] = first_pass
                    losses_by_mode[mode][condition] = loss
                    measurements[mode][condition] = metric
                    print(
                        json.dumps(
                            {
                                "event": "score_finished",
                                "document_id": doc_id,
                                "mode": mode,
                                "condition": condition,
                                **metric,
                            }
                        ),
                        flush=True,
                    )
                # Preserve a complete triplet before trying the other backend.
                save_pair(
                    args.output / "backend_checks" / mode / paths[doc_id].name,
                    document_id=doc_id,
                    arms={
                        name: {
                            "target_ids": plan.inputs[name][plan.target_starts[name] :],
                            "target_positions": plan.target_positions,
                            "losses": losses_by_mode[mode][name],
                        }
                        for name in CONDITIONS
                    },
                    repeat_bins=bins,
                    metadata={
                        "kind": "public_model_backend_check",
                        "model": adapter.metadata,
                        "hardware": hardware,
                        "history": plan.describe(),
                        "execution": mode,
                        "measurements": measurements[mode],
                        "source": row.get("source"),
                        "donor_id": row["donor_id"],
                    },
                )
            except (RuntimeError, ValueError, AssertionError) as error:
                if mode != "recurrent" or selected is not None:
                    raise
                # Keep the complete fused-reference triplet when native cached
                # execution exposes a backend problem. Repair/retry only that path.
                measurements[mode]["error"] = f"{type(error).__name__}: {error}"
                losses_by_mode.pop(mode)
        if selected is None:
            comparison = (
                compare_modes(
                    losses_by_mode["full"],
                    losses_by_mode["recurrent"],
                    args.mean_tolerance,
                    args.max_tolerance,
                )
                if "recurrent" in losses_by_mode
                else {
                    "within_provisional_tolerances": False,
                    "recurrent_error": measurements["recurrent"]["error"],
                    "note": "Cached path unavailable; reference losses retained. Parity is unmeasured.",
                }
            )
            times = {
                mode: sum(measurements[mode][name]["seconds"] for name in CONDITIONS)
                for mode in losses_by_mode
            }
            eligible = (
                ("full", "recurrent") if comparison["within_provisional_tolerances"] else ("full",)
            )
            selected = min(eligible, key=times.get)
            comparison["selected_execution"] = selected
            comparison["measured_seconds_three_conditions"] = times
            comparison["timing_scope"] = "one warm pass per history after one identical first pass"
            comparison["tested_document_id"] = doc_id
            comparison["scope"] = (
                "first unscored document; its selected execution path is reused for the rest"
            )
            print(json.dumps({"backend_comparison": comparison}), flush=True)
        arms = {
            name: {
                "target_ids": plan.inputs[name][plan.target_starts[name] :],
                "target_positions": plan.target_positions,
                "losses": losses_by_mode[selected][name],
            }
            for name in CONDITIONS
        }
        metadata = {
            "kind": "public_model_development_smoke",
            "model": adapter.metadata,
            "hardware": hardware,
            "selected_execution": selected,
            "backend_comparison": comparison,
            "measurements": measurements,
            "history": plan.describe(),
            "chunk_size": args.chunk_size,
            "donor_id": row["donor_id"],
            "source": row.get("source"),
            "donor_source": by_id[row["donor_id"]].get("source"),
            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "donor_text_sha256": hashlib.sha256(
                by_id[row["donor_id"]]["text"].encode()
            ).hexdigest(),
            "elapsed_since_model_load_start_seconds": time.perf_counter() - total_start,
        }
        save_pair(paths[doc_id], document_id=doc_id, arms=arms, repeat_bins=bins, metadata=metadata)
        print(json.dumps({"saved_document": doc_id, "execution": selected}), flush=True)
    write_report(
        [path for path in paths.values() if path.exists()],
        args.output / "summary.json",
        args.bootstrap_samples,
        args.seed,
    )


if __name__ == "__main__":
    main()
