"""Score one pre-aligned text block with the checkpoint named in its plan.

Plans contain native token IDs and source-span metadata. Each process loads
one model and writes a separate matched-text result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

import numpy as np

from .check_mamba import timed
from .matched_text import load_text_plan, save_matched_text, summarize_matched_losses
from .measurement import CONDITIONS


def scoring_command(root, *, name, plan_path, result_path, revision, python_executable=None):
    """Return the Linux command that scores one plan offline in its model environment.

    ``root`` holds the Hugging Face cache, Triton kernel cache and temporary files.
    """
    root = PurePosixPath(root)
    python = python_executable or sys.executable
    return [
        "env",
        "-u",
        "PYTHONPATH",
        "-u",
        "PYTHONHOME",
        "-u",
        "VIRTUAL_ENV",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        f"HF_HOME={root / 'hf'}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "TOKENIZERS_PARALLELISM=false",
        "TRITON_F32_DEFAULT=ieee",
        f"TRITON_CACHE_DIR={root / 'cache' / f'triton-{name}'}",
        "OMP_NUM_THREADS=2",
        "MKL_NUM_THREADS=2",
        "OPENBLAS_NUM_THREADS=2",
        "MAX_JOBS=2",
        "TORCHINDUCTOR_COMPILE_THREADS=2",
        f"TMPDIR={root / 'tmp'}",
        str(python),
        "-u",
        "-m",
        "long_context_diagnostics.score_matched_text",
        "--plan",
        str(plan_path),
        "--output",
        str(result_path),
        "--solve-precision",
        "native",
        "--repeat",
        "--max-memory-gb",
        "14",
        "--code-revision",
        revision,
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="separate matched-text result NPZ"
    )
    parser.add_argument("--max-memory-gb", type=float, default=14)
    parser.add_argument("--chunk-size", type=int, default=256, help="target logits only")
    parser.add_argument("--solve-precision", choices=("native", "ieee"), default="native")
    parser.add_argument("--repeat", action="store_true", help="one warm repeat per history")
    parser.add_argument("--code-revision", default="unrecorded", help="provenance only")
    args = parser.parse_args(argv)
    if not 0 < args.max_memory_gb <= 14 or args.chunk_size < 1:
        parser.error("require 0 < max-memory-gb <= 14 and positive chunk size")
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    plan = load_text_plan(args.plan)
    name = plan.metadata["model_name"]
    if name not in ("mamba", "gdn", "rwkv"):
        parser.error("plan model_name must be mamba, gdn or rwkv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "public_model_matched_text_block",
        "status": "started",
        "document_id": plan.metadata["document_id"],
        "donor_id": plan.metadata["donor_id"],
        "history": plan.metadata,
        "code_revision": args.code_revision,
        "selected_execution": "full",
        "max_memory_gb": args.max_memory_gb,
        "measurements": {},
    }
    report_path = args.output.with_suffix(".json")

    def write_report():
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    write_report()
    try:
        begin = time.perf_counter()
        reference = plan.metadata["model_reference"]
        model_args = {key: reference[key] for key in ("model_id", "revision", "tokenizer_revision")}
        if name == "mamba":
            from .mamba_adapter import MambaAdapter

            adapter = MambaAdapter(**model_args, dtype="float32", max_memory_gb=args.max_memory_gb)
        elif name == "gdn":
            from .gdn_adapter import GDNAdapter

            adapter = GDNAdapter(
                **model_args,
                dtype="float32",
                max_memory_gb=args.max_memory_gb,
                solve_precision=args.solve_precision,
            )
        else:
            from .rwkv_adapter import RWKVAdapter

            adapter = RWKVAdapter(**model_args, dtype="float32", max_memory_gb=args.max_memory_gb)
        adapter._torch.cuda.synchronize()
        report.update(
            model=adapter.metadata,
            selected_execution=adapter.metadata.get("selected_execution", "full"),
            hardware={
                "gpu": adapter._torch.cuda.get_device_name(),
                "load_seconds": time.perf_counter() - begin,
            },
        )
        losses = {}
        for condition in CONDITIONS:
            print(
                json.dumps({"event": "score_started", "model": name, "condition": condition}),
                flush=True,
            )
            loss, metric = timed(
                adapter,
                adapter.score_full,
                plan.inputs[condition],
                plan.target_starts[condition],
                args.chunk_size,
            )
            metric["timing_scope"] = "single pass; includes first-use kernel compilation if any"
            if args.repeat:
                cold, first = loss, metric
                loss, metric = timed(
                    adapter,
                    adapter.score_full,
                    plan.inputs[condition],
                    plan.target_starts[condition],
                    args.chunk_size,
                )
                metric.update(
                    first_pass=first,
                    repeat_max_abs=float(np.max(np.abs(loss.astype(np.float64) - cold))),
                    timing_scope="warm repeat after one identical first pass",
                )
            losses[condition], report["measurements"][condition] = loss, metric
            write_report()
            print(
                json.dumps(
                    {"event": "score_finished", "model": name, "condition": condition, **metric}
                ),
                flush=True,
            )
        report["matched_text"] = summarize_matched_losses(plan, losses)
        save_matched_text(
            args.output,
            document_id=plan.metadata["document_id"],
            plan=plan,
            losses=losses,
            metadata={
                key: value for key, value in report.items() if key not in ("history", "status")
            },
        )
        report["status"] = "completed"
        write_report()
        print(
            json.dumps(
                {
                    "event": "matched_text_finished",
                    "output": str(args.output),
                    "matched_text": report["matched_text"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_report()
        raise


if __name__ == "__main__":
    main()
