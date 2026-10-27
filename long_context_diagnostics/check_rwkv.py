"""Numerical parity and timing checks for the official RWKV-7 CUDA path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .rwkv_adapter import (
    DEFAULT_FILENAME,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DEFAULT_TOKENIZER_REVISION,
    RWKVAdapter,
)
from .check_mamba import timed
from .measurement import CONDITIONS, build_histories, save_pair


def _comparison(reference, candidate, mean_tolerance, max_tolerance):
    delta = np.abs(
        np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    )
    finite = bool(np.isfinite(delta).all())
    mean_error, max_error = float(delta.mean()), float(delta.max())
    return {
        "mean_abs": mean_error,
        "max_abs": max_error,
        "passed": bool(finite and mean_error <= mean_tolerance and max_error <= max_tolerance),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--donor-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--end", type=int, default=2048)
    parser.add_argument("--recent", type=int, default=512)
    parser.add_argument("--targets", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--alternate-chunk-size", type=int, default=127)
    parser.add_argument(
        "--reference-mode",
        choices=("full_sequence", "single_prefill"),
        default="full_sequence",
        help="full_sequence for short smoke; single_prefill avoids all-prefix vocabulary logits",
    )
    parser.add_argument("--single-step-tokens", type=int, default=128)
    parser.add_argument("--max-memory-gb", type=float, default=14)
    parser.add_argument("--mean-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--cuda-off", action="store_true", help="separate-process official eager reference"
    )
    parser.add_argument("--code-revision", default="unrecorded")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "smoke.json"
    report = {
        "kind": "public_rwkv7_backend_smoke",
        "status": "started",
        "code_revision": args.code_revision,
        "measurements": {},
        "comparisons": {},
        "tolerances": {"mean_abs": args.mean_tolerance, "max_abs": args.max_tolerance},
    }

    def write():
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    write()
    try:
        rows = {
            row["document_id"]: row
            for line in args.source_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
        begin = time.perf_counter()
        adapter = RWKVAdapter(
            model_id=args.model_id,
            revision=args.revision,
            tokenizer_revision=args.tokenizer_revision,
            filename=args.filename,
            tokenizer_path=args.tokenizer_path,
            model_path=args.model_path,
            max_memory_gb=args.max_memory_gb,
            prefill_chunk_size=args.prefill_chunk_size,
            cuda_on=not args.cuda_off,
        )
        adapter._torch.cuda.synchronize()
        report.update(
            model=adapter.metadata,
            hardware={
                "gpu": adapter._torch.cuda.get_device_name(),
                "load_seconds": time.perf_counter() - begin,
            },
        )
        document = np.asarray(adapter.encode(rows[args.document_id]["text"]), dtype=np.int64)
        donor = np.asarray(adapter.encode(rows[args.donor_id]["text"]), dtype=np.int64)
        plan = build_histories(
            document,
            donor,
            end=args.end,
            recent_tokens=args.recent,
            target_tokens=args.targets,
            prefix_id=adapter.prefix_id,
        )
        report["history"] = {
            "document_id": args.document_id,
            "donor_id": args.donor_id,
            "end": args.end,
            "recent": args.recent,
            "targets": args.targets,
        }
        losses_by_mode = {}
        reference_name, reference_method = (
            ("full_reference", adapter.score_reference)
            if args.reference_mode == "full_sequence"
            else ("single_prefill_reference", adapter.score_prefill_reference)
        )
        report["reference"] = {
            "mode": args.reference_mode,
            "measurement_name": reference_name,
            "description": (
                "Single native full-output sequence, including every prefix vocabulary logit"
                if args.reference_mode == "full_sequence"
                else "One native prefix call with final output only, followed by bounded target chunks"
            ),
            "target_chunk_size": args.chunk_size,
        }
        modes = (
            (reference_name, reference_method, args.chunk_size),
            ("fused_prefill", adapter.score_full, args.chunk_size),
            ("alternate_chunks", adapter.score, args.alternate_chunk_size),
        )
        for mode, method, size in modes:
            losses_by_mode[mode], report["measurements"][mode] = {}, {}
            for condition in CONDITIONS:
                ids, start = plan.inputs[condition], plan.target_starts[condition]
                _, cold = timed(adapter, method, ids, start, size)
                losses, warm = timed(adapter, method, ids, start, size)
                losses_by_mode[mode][condition] = losses
                report["measurements"][mode][condition] = {
                    "cold": cold,
                    "warm": warm,
                    "mean_ce": float(losses.mean()),
                }
                print(
                    json.dumps(
                        {
                            "event": "rwkv_smoke_cell",
                            "mode": mode,
                            "condition": condition,
                            **report["measurements"][mode][condition],
                        }
                    ),
                    flush=True,
                )
                write()
            save_pair(
                args.output / f"{mode}.npz",
                document_id=args.document_id,
                arms={
                    condition: {
                        "target_ids": plan.target_ids,
                        "target_positions": plan.target_positions,
                        "losses": losses_by_mode[mode][condition],
                    }
                    for condition in CONDITIONS
                },
                repeat_bins={},
                metadata={
                    "model": adapter.metadata,
                    "history": report["history"],
                    "kind": "public_rwkv_backend_check",
                    "mode": mode,
                },
            )
        reference = losses_by_mode[reference_name]
        for mode in ("fused_prefill", "alternate_chunks"):
            report["comparisons"][mode] = {
                condition: _comparison(
                    reference[condition],
                    losses_by_mode[mode][condition],
                    args.mean_tolerance,
                    args.max_tolerance,
                )
                for condition in CONDITIONS
            }

        # The one-token API has a separate TorchScript recurrence rather than
        # the sequence CUDA loop. Use a short real prefix so this independent
        # recurrence check stays short.
        short_ids = plan.inputs["full"][: max(2, args.single_step_tokens + 1)]
        short_start = max(1, len(short_ids) - min(args.targets, 32))
        one, one_time = timed(adapter, adapter.score_one, short_ids, short_start, 1)
        native, native_time = timed(
            adapter, adapter.score_reference, short_ids, short_start, args.chunk_size
        )
        report["single_step"] = {
            "input_tokens": len(short_ids),
            "targets": len(one),
            "one": one_time,
            "sequence": native_time,
            **_comparison(native, one, args.mean_tolerance, args.max_tolerance),
        }
        np.savez_compressed(
            args.output / "single_step.npz",
            one=one,
            sequence=native,
            input_ids=short_ids,
            target_start=short_start,
        )
        passed = report["single_step"]["passed"] and all(
            cell["passed"] for modes in report["comparisons"].values() for cell in modes.values()
        )
        report.update(status="completed", passed=bool(passed))
        write()
        print(
            json.dumps(
                {"event": "rwkv_smoke_finished", "passed": bool(passed), "report": str(report_path)}
            ),
            flush=True,
        )
        return 0 if passed else 2
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
