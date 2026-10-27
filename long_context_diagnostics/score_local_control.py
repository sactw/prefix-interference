"""Score a fixed target block under local original, short and foreign histories.

Defaults retain the original 32K block's 1024 targets, 512 recent tokens and
511 preceding actual/foreign tokens: 2048 serialized IDs including EOS/BOS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .mamba_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DEFAULT_TOKENIZER_REVISION,
    TOKENIZER_ID,
    MambaAdapter,
)
from .check_mamba import timed
from .measurement import CONDITIONS, build_histories, load_pair, save_pair
from .readout import classify_repeats


def build_local_control(
    ids, donor_ids, *, end=32768, recent=512, targets=1024, far=511, prefix_id=0
):
    """Return local histories, absolute target positions and visible-only bins."""
    source_start = end - targets - recent - far
    if source_start < 0 or end > len(ids) or far < 0:
        raise ValueError("selected local full history must lie inside the document")
    visible = np.asarray(ids)[source_start:end]
    plan = build_histories(
        visible,
        donor_ids,
        end=len(visible),
        recent_tokens=recent,
        target_tokens=targets,
        prefix_id=prefix_id,
    )
    bins = {
        str(n): classify_repeats(visible, plan.target_positions, plan.retained_start, n=n)
        for n in (4, 8)
    }
    history = plan.describe()
    history.update(
        end_exclusive=end,
        native_end_exclusive=len(visible),
        full_history_start=source_start,
        retained_start=source_start + plan.retained_start,
        cropped_context=True,
        repeat_label_scope="visible_full_slice",
        target_position_scope="absolute_original_document",
    )
    return plan, plan.target_positions + source_start, bins, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--donor-id", help="defaults to the target's assigned donor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--end", type=int, default=32768)
    parser.add_argument("--recent", type=int, default=512)
    parser.add_argument("--targets", type=int, default=1024)
    parser.add_argument("--far", type=int, default=511)
    parser.add_argument("--chunk-size", type=int, default=256, help="target logits only")
    parser.add_argument("--max-memory-gb", type=float, default=14)
    parser.add_argument("--code-revision", default="unrecorded")
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
    if row["document_id"] == donor["document_id"] or row["text"] == donor["text"]:
        parser.error("foreign history requires a different document")
    text_sha = hashlib.sha256(row["text"].encode()).hexdigest()
    donor_sha = hashlib.sha256(donor["text"].encode()).hexdigest()
    science = {
        "target": row["document_id"],
        "donor": donor["document_id"],
        "text_sha256": text_sha,
        "donor_text_sha256": donor_sha,
        "E": args.end,
        "C": args.recent,
        "T": args.targets,
        "F": args.far,
        "model": args.model_id,
        "revision": args.revision,
        "tokenizer": args.tokenizer_revision,
    }
    suffix = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()[:20]
    path = args.output / f"local-full-block-{suffix}.npz"
    if path.exists():
        load_pair(path)
        print(json.dumps({"event": "reused_local_control", "file": str(path)}), flush=True)
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=args.tokenizer_revision)
    ids = np.asarray(tokenizer.encode(row["text"], add_special_tokens=False), dtype=np.int64)
    donor_ids = np.asarray(
        tokenizer.encode(donor["text"], add_special_tokens=False), dtype=np.int64
    )
    begin = time.perf_counter()
    adapter = MambaAdapter(
        args.model_id, args.revision, args.tokenizer_revision, max_memory_gb=args.max_memory_gb
    )
    adapter._torch.cuda.synchronize()
    hardware = {
        "gpu": adapter._torch.cuda.get_device_name(),
        "load_seconds": time.perf_counter() - begin,
        "code_revision": args.code_revision,
    }
    plan, positions, bins, history = build_local_control(
        ids,
        donor_ids,
        end=args.end,
        recent=args.recent,
        targets=args.targets,
        far=args.far,
        prefix_id=adapter.prefix_id,
    )
    measurements, losses = {}, {}
    for condition in CONDITIONS:
        print(
            json.dumps(
                {
                    "event": "score_started",
                    "condition": condition,
                    "input_tokens": len(plan.inputs[condition]),
                }
            ),
            flush=True,
        )
        loss, timing = timed(
            adapter,
            adapter.score_full,
            plan.inputs[condition],
            plan.target_starts[condition],
            args.chunk_size,
        )
        losses[condition], measurements[condition] = loss, timing
        print(json.dumps({"event": "score_finished", "condition": condition, **timing}), flush=True)
    save_pair(
        path,
        document_id=row["document_id"],
        arms={
            name: {
                "target_ids": plan.inputs[name][plan.target_starts[name] :],
                "target_positions": positions,
                "losses": losses[name],
            }
            for name in CONDITIONS
        },
        repeat_bins=bins,
        metadata={
            "kind": "public_model_local_context_control",
            "model": adapter.metadata,
            "hardware": hardware,
            "selected_execution": "full",
            "history": history,
            "measurements": {"full": measurements},
            "chunk_size": args.chunk_size,
            "donor_id": donor["document_id"],
            "source": row.get("source"),
            "donor_source": donor.get("source"),
            "text_sha256": text_sha,
            "donor_text_sha256": donor_sha,
            "triton_fp32_dot_precision_requested": "ieee",
        },
    )
    print(
        json.dumps(
            {
                "event": "saved_local_control",
                "file": str(path),
                "history": history,
                "ce": {
                    key: float(value.astype(np.float64).mean()) for key, value in losses.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
