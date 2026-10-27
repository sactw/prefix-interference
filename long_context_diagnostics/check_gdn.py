"""Compare full, cached-chunk and recurrent GDN inference on identical targets.

Records token-level NLL differences, paired-effect differences, warm timings
and memory usage. Completed execution modes are saved independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .gdn_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DEFAULT_TOKENIZER_REVISION,
    GDNAdapter,
)
from .check_mamba import compare_modes, timed
from .measurement import CONDITIONS, build_histories, save_pair
from .readout import classify_repeats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--donor-id", help="overrides the target row's donor_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--solve-precision",
        choices=("native", "ieee"),
        default="native",
        help="optional process-local IEEE override for FLA's fused triangular solve",
    )
    parser.add_argument("--end", type=int, default=2048)
    parser.add_argument("--recent", type=int, default=512)
    parser.add_argument("--targets", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=float, default=14)
    parser.add_argument("--code-revision", default="unrecorded")
    parser.add_argument("--mean-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-tolerance", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if not (0 < args.max_memory_gb <= 14 and args.chunk_size > 0):
        parser.error("require 0 < max-memory-gb <= 14 and positive chunk size")
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    rows = [
        json.loads(line)
        for line in args.documents.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["document_id"]: row for row in rows}
    row = by_id[args.document_id]
    donor = by_id[args.donor_id or row["donor_id"]]
    if donor["document_id"] == row["document_id"] or donor["text"] == row["text"]:
        parser.error("select a distinct foreign document")
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "model_load_started", "model": args.model_id}), flush=True)
    begin = time.perf_counter()
    adapter = GDNAdapter(
        args.model_id,
        args.revision,
        args.tokenizer_revision,
        dtype=args.dtype,
        max_memory_gb=args.max_memory_gb,
        solve_precision=args.solve_precision,
    )
    adapter._torch.cuda.synchronize()
    hardware = {
        "gpu": adapter._torch.cuda.get_device_name(),
        "load_seconds": time.perf_counter() - begin,
        "code_revision": args.code_revision,
    }
    tokens = np.asarray(adapter.encode(row["text"]), dtype=np.int64)
    donor_tokens = np.asarray(adapter.encode(donor["text"]), dtype=np.int64)
    plan = build_histories(
        tokens,
        donor_tokens,
        end=args.end,
        recent_tokens=args.recent,
        target_tokens=args.targets,
        prefix_id=adapter.prefix_id,
    )
    bins = {
        str(n): classify_repeats(tokens, plan.target_positions, plan.retained_start, n=n)
        for n in (4, 8)
    }
    report = {
        "kind": "public_gdn_backend_smoke",
        "model": adapter.metadata,
        "hardware": hardware,
        "history": plan.describe(),
        "document_id": row["document_id"],
        "donor_id": donor["document_id"],
        "modes": {},
        "comparisons": {},
        "note": "Public BF16 weights; FP32 complete-vocabulary CE. Agreement thresholds are engineering diagnostics.",
    }
    all_losses = {}

    def write_report():
        (args.output / "smoke.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    for mode, method in (
        ("full", adapter.score_full),
        ("cached_chunk", adapter.score),
        ("fused_recurrent", adapter.score_recurrent),
    ):
        mode_report = report["modes"][mode] = {"measurements": {}, "status": "started"}
        losses = {}
        try:
            for condition in CONDITIONS:
                print(
                    json.dumps({"event": "score_started", "mode": mode, "condition": condition}),
                    flush=True,
                )
                cold, first = timed(
                    adapter,
                    method,
                    plan.inputs[condition],
                    plan.target_starts[condition],
                    args.chunk_size,
                )
                loss, metric = timed(
                    adapter,
                    method,
                    plan.inputs[condition],
                    plan.target_starts[condition],
                    args.chunk_size,
                )
                if not np.isfinite(loss).all():
                    raise ValueError(f"nonfinite NLL in {mode}/{condition}")
                losses[condition] = loss
                metric.update(
                    first_pass=first,
                    repeat_max_abs=float(np.max(np.abs(loss.astype(np.float64) - cold))),
                    mean_ce=float(loss.astype(np.float64).mean()),
                )
                mode_report["measurements"][condition] = metric
                print(
                    json.dumps(
                        {"event": "score_finished", "mode": mode, "condition": condition, **metric}
                    ),
                    flush=True,
                )
            save_pair(
                args.output / f"{mode}.npz",
                document_id=row["document_id"],
                arms={
                    name: {
                        "target_ids": plan.inputs[name][plan.target_starts[name] :],
                        "target_positions": plan.target_positions,
                        "losses": losses[name],
                    }
                    for name in CONDITIONS
                },
                repeat_bins=bins,
                metadata={
                    "kind": "public_gdn_backend_check",
                    "model": adapter.metadata,
                    "hardware": hardware,
                    "history": plan.describe(),
                    "execution": mode,
                    "measurements": mode_report["measurements"],
                    "source": row.get("source"),
                    "donor_source": donor.get("source"),
                    "donor_id": donor["document_id"],
                    "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    "donor_text_sha256": hashlib.sha256(donor["text"].encode()).hexdigest(),
                },
            )
            all_losses[mode] = losses
            mode_report["status"] = "completed"
            if mode != "full":
                report["comparisons"][mode] = compare_modes(
                    all_losses["full"], losses, args.mean_tolerance, args.max_tolerance
                )
        except Exception as error:
            mode_report.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            write_report()
            if mode == "full":
                raise
        write_report()
    # Record both BF16 execution paths without selecting a backend automatically.
    report["status"] = (
        "completed"
        if all(item["status"] == "completed" for item in report["modes"].values())
        else "partial"
    )
    write_report()
    print(
        json.dumps(
            {
                "event": "smoke_finished",
                "report": str(args.output / "smoke.json"),
                "status": report["status"],
                "comparisons": report["comparisons"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
