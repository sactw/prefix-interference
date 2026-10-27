"""Score Mamba-2 target blocks at selected endpoints using the native fused scan.

The default endpoint is 32K. Hidden activation memory grows with input length;
completed triplets are saved independently for resumption.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .__main__ import write_report
from .mamba_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DEFAULT_TOKENIZER_REVISION,
    TOKENIZER_ID,
    MambaAdapter,
)
from .check_mamba import record_path, timed
from .measurement import CONDITIONS, build_histories, load_pair, save_pair
from .readout import classify_repeats


def plan_cases(rows, args):
    """Name records from their inputs, independent of the code revision."""
    by_id = {row["document_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("document_id must be unique in the input JSONL")
    targets = [row for row in rows if row["role"] == "target"]
    requested = set(args.document_id or ())
    if requested.difference(row["document_id"] for row in targets):
        raise ValueError("--document-id must name target-role documents in the JSONL")
    selected = [row for row in targets if not requested or row["document_id"] in requested]
    if not selected:
        raise ValueError("no selected target documents")
    cases = []
    for row in selected:
        donor = by_id.get(row.get("donor_id"))
        if donor is None or donor["role"] != "donor" or row["text"] == donor["text"]:
            raise ValueError("each target needs a separate donor-role document with different text")
        for end in args.ends:
            name = record_path(
                args.output / f"end-{end}", row, donor, SimpleNamespace(**dict(vars(args), end=end))
            )
            cases.append(
                {
                    "row": row,
                    "donor": donor,
                    "end": end,
                    "path": name.with_name("ieee-full-" + name.name),
                }
            )
    return cases


def eligibility(case, tokens, recent, targets):
    """Length feasibility is reported before allocating model memory."""
    available = len(tokens[case["row"]["document_id"]])
    donor_available = len(tokens[case["donor"]["document_id"]])
    needed = case["end"] - recent - targets
    if available < case["end"] or donor_available < needed:
        return {
            "status": "insufficient_text",
            "document_tokens": available,
            "donor_tokens": donor_available,
            "required_document_tokens": case["end"],
            "required_donor_tokens": needed,
        }
    return None


def _describe(case):
    return {
        "document_id": case["row"]["document_id"],
        "end": case["end"],
        "file": str(case["path"]),
    }


def _write_status(args, status):
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output / "status.json")


def run_cases(adapter, cases, tokens, args, hardware, status):
    """Save each complete triplet immediately; an OOM leaves earlier rows usable."""
    for index, case in enumerate(cases):
        row, donor, end = case["row"], case["donor"], case["end"]
        doc_id = row["document_id"]
        plan = build_histories(
            tokens[doc_id],
            tokens[donor["document_id"]],
            end=end,
            recent_tokens=args.recent,
            target_tokens=args.targets,
            prefix_id=adapter.prefix_id,
        )
        measurements, losses = {}, {}
        condition = "full"
        try:
            for condition in CONDITIONS:
                print(
                    json.dumps(
                        {"event": "score_started", **_describe(case), "condition": condition}
                    ),
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
                if args.repeat_first_block and index == 0:
                    first_pass = metric
                    loss, metric = timed(
                        adapter,
                        adapter.score_full,
                        plan.inputs[condition],
                        plan.target_starts[condition],
                        args.chunk_size,
                    )
                    metric["first_pass"] = first_pass
                    metric["timing_scope"] = "warm repeat after one identical first pass"
                measurements[condition], losses[condition] = metric, loss
                print(
                    json.dumps(
                        {
                            "event": "score_finished",
                            **_describe(case),
                            "condition": condition,
                            **metric,
                        }
                    ),
                    flush=True,
                )
        except adapter._torch.cuda.OutOfMemoryError as error:
            status["blocks"].append(
                {
                    **_describe(case),
                    "status": "out_of_memory",
                    "condition": condition,
                    "input_tokens": len(plan.inputs[condition]),
                    "max_memory_gb": args.max_memory_gb,
                    "error": str(error),
                    "peak_allocated_bytes": adapter._torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": adapter._torch.cuda.max_memory_reserved(),
                    "recovery": "Completed triplets remain readable; only this document and endpoint need a retry.",
                    "completed_condition_measurements": measurements,
                }
            )
            _write_status(args, status)
            # Only this failed call is discarded. Fresh full-forward calls do not
            # carry state; clearing unused allocations permits the next document.
            error.__traceback__ = None
            gc.collect()
            adapter._torch.cuda.empty_cache()
            print(
                json.dumps({"event": "out_of_memory", **_describe(case), "condition": condition}),
                flush=True,
            )
            continue
        bins = {
            str(n): classify_repeats(
                tokens[doc_id], plan.target_positions, plan.retained_start, n=n
            )
            for n in (4, 8)
        }
        save_pair(
            case["path"],
            document_id=doc_id,
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
                "kind": "public_model_length_pilot",
                "model": adapter.metadata,
                "hardware": hardware,
                "selected_execution": "full",
                "history": plan.describe(),
                "measurements": {"full": measurements},
                "chunk_size": args.chunk_size,
                "chunk_size_scope": "target logits only; hidden activations span the full input",
                "donor_id": donor["document_id"],
                "source": row.get("source"),
                "donor_source": donor.get("source"),
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "donor_text_sha256": hashlib.sha256(donor["text"].encode()).hexdigest(),
                "triton_fp32_dot_precision_requested": "ieee",
            },
        )
        status["blocks"].append({**_describe(case), "status": "completed"})
        _write_status(args, status)
        print(json.dumps({"event": "saved_triplet", **_describe(case)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--end", "--ends", dest="ends", type=int, nargs="+", default=[32768])
    parser.add_argument("--recent", type=int, default=2048)
    parser.add_argument("--targets", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256, help="target-logit chunk size")
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=14,
        help="per-process PyTorch allocator cap in decimal GB, at most14",
    )
    parser.add_argument("--document-id", action="append")
    parser.add_argument("--code-revision", default="unrecorded", help="provenance only")
    parser.add_argument(
        "--repeat-first-block",
        action="store_true",
        help="time one cold and one warm pass per history on the first unscored block",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    if not (
        0 < args.max_memory_gb <= 14
        and args.recent >= 1
        and args.targets >= 1
        and args.chunk_size >= 1
        and all(end >= args.recent + args.targets for end in args.ends)
    ):
        parser.error("require0<max-memory-gb<=14, positive C/T/chunk-size, and every E>=C+T")
    args.ends = list(dict.fromkeys(args.ends))
    # Set before importing transformers/torch/Triton, and preserve the effective
    # value in adapter metadata. This is the numerically validated execution path.
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    rows = [
        json.loads(line)
        for line in args.documents.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = plan_cases(rows, args)
    report_cases = plan_cases(rows, SimpleNamespace(**dict(vars(args), document_id=None)))
    status = {
        "kind": "public_model_length_pilot_status",
        "code_revision": args.code_revision,
        "requested_ends": args.ends,
        "max_memory_gb": args.max_memory_gb,
        "triton_fp32_dot_precision": "ieee",
        "blocks": [],
    }
    pending = []
    for case in cases:
        if case["path"].exists():
            saved = load_pair(case["path"])
            status["blocks"].append(
                {
                    **_describe(case),
                    "status": "reused",
                    "original_code_revision": saved["metadata"]["hardware"].get("code_revision"),
                }
            )
        else:
            pending.append(case)
    _write_status(args, status)
    if pending:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=args.tokenizer_revision)
        required = {
            row["document_id"]: row for case in pending for row in (case["row"], case["donor"])
        }
        token_start = time.perf_counter()
        tokens = {
            key: np.asarray(tokenizer.encode(row["text"], add_special_tokens=False), dtype=np.int64)
            for key, row in required.items()
        }
        status["tokenization_seconds"] = time.perf_counter() - token_start
        status["document_tokens"] = {key: len(value) for key, value in tokens.items()}
        eligible = []
        for case in pending:
            reason = eligibility(case, tokens, args.recent, args.targets)
            if reason:
                status["blocks"].append({**_describe(case), **reason})
            else:
                eligible.append(case)
        _write_status(args, status)
        print(
            json.dumps(
                {
                    "event": "tokenized",
                    "document_tokens": status["document_tokens"],
                    "eligible_blocks": len(eligible),
                }
            ),
            flush=True,
        )
        if eligible:
            begin = time.perf_counter()
            adapter = MambaAdapter(
                args.model_id,
                args.revision,
                args.tokenizer_revision,
                max_memory_gb=args.max_memory_gb,
            )
            adapter._torch.cuda.synchronize()
            hardware = {
                "gpu": adapter._torch.cuda.get_device_name(),
                "load_seconds": time.perf_counter() - begin,
                "code_revision": args.code_revision,
                "max_memory_gb": args.max_memory_gb,
            }
            run_cases(adapter, eligible, tokens, args, hardware, status)
    # Separate panels preserve context-length interpretation; document clusters
    # remain paired within each panel. No automatic cross-length pooling.
    for end in args.ends:
        paths = [
            case["path"] for case in report_cases if case["end"] == end and case["path"].exists()
        ]
        if paths:
            write_report(
                paths,
                args.output / f"end-{end}" / "summary.json",
                args.bootstrap_samples,
                args.seed,
            )
    _write_status(args, status)
    print(
        json.dumps(
            {
                "event": "panel_finished",
                "status_file": str(args.output / "status.json"),
                "blocks": status["blocks"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
