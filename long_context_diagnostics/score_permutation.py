"""Paired permutation of older prefix blocks for the Mamba-2 checkpoint.

The CLI fixes E=32768, C=T=1024, F=30720, 512-token blocks and seed 20261024.
Only the first 59 F/U blocks are reordered, with the same block order. Their
last block, the F/R seam, native boundary and R/Y are fixed. Original repeat
strata are retained in both separately saved triplets.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
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

END, RECENT, TARGETS, BLOCK_SIZE, SEED = 32768, 1024, 1024, 512, 20261024


def permute_old_prefix(plan, *, block_size=BLOCK_SIZE, seed=SEED, keep_tail_blocks=1):
    """Return a new plan without mutating any original array."""
    far = plan.retained_start
    if block_size < 1 or far < block_size or far % block_size:
        raise ValueError("the nonempty far prefix must contain whole blocks")
    blocks = far // block_size
    if not 0 <= keep_tail_blocks <= blocks - 2:
        raise ValueError("require at least two shuffled blocks and a nonnegative fixed tail")
    shuffled = blocks - keep_tail_blocks
    order = np.concatenate(
        (
            np.random.Generator(np.random.PCG64(seed)).permutation(shuffled),
            np.arange(shuffled, blocks),
        )
    )
    prefix = 0 if plan.prefix_id is None else 1
    inputs = {key: value.copy() for key, value in plan.inputs.items()}
    checks = {}
    for condition in ("full", "foreign"):
        before = plan.inputs[condition][prefix : prefix + far]
        inputs[condition][prefix : prefix + far] = before.reshape(-1, block_size)[order].reshape(-1)
        after = inputs[condition][prefix : prefix + far]
        checks[condition] = {
            "token_multiset_equal": bool(np.array_equal(np.sort(before), np.sort(after))),
            "boundary_equal": bool(
                np.array_equal(inputs[condition][:prefix], plan.inputs[condition][:prefix])
            ),
            "recent_and_targets_equal": bool(
                np.array_equal(
                    inputs[condition][prefix + far :], plan.inputs[condition][prefix + far :]
                )
            ),
            "anchored_tail_equal": bool(
                np.array_equal(after[shuffled * block_size :], before[shuffled * block_size :])
            ),
        }
    return replace(plan, inputs=inputs), {
        "name": "old_prefix_block_permutation",
        "seed": int(seed),
        "rng": "NumPy PCG64",
        "block_tokens": int(block_size),
        "far_tokens": int(far),
        "blocks": len(order),
        "shuffled_blocks": int(shuffled),
        "keep_tail_blocks": int(keep_tail_blocks),
        "anchored_tail_tokens": int(keep_tail_blocks * block_size),
        "order": order.tolist(),
        "order_convention": "destination block i receives original block order[i]",
        "same_order_for_actual_and_foreign": True,
        "fixed_blocks": int(np.sum(order == np.arange(len(order)))),
        "verification": checks,
        "short_input_equal": bool(np.array_equal(inputs["short"], plan.inputs["short"])),
    }


def _ngram_keys(tokens, n):
    return {tuple(map(int, tokens[i : i + n])) for i in range(max(0, len(tokens) - n + 1))}


def repeat_survival(original, permuted, bins):
    """Target n-gram matches in F/U after permutation, per original far stratum.

    N-grams are compared as token tuples, and n-grams created or destroyed at
    block junctions are reported. The saved repeat bins are those computed
    before the permutation.
    """
    prefix = 0 if original.prefix_id is None else 1
    far = original.retained_start
    full = original.inputs["full"][prefix:]
    result = {}
    for label_n, labels in bins.items():
        n = int(label_n)
        keys = [
            tuple(map(int, full[p - n + 1 : p + 1])) if p >= n - 1 else None
            for p in original.target_positions
        ]
        masks, by_prefix = {}, {}
        for condition in ("full", "foreign"):
            before_keys = _ngram_keys(original.inputs[condition][prefix : prefix + far], n)
            after_keys = _ngram_keys(permuted.inputs[condition][prefix : prefix + far], n)
            before = np.asarray([key in before_keys for key in keys], dtype=bool)
            after = np.asarray([key in after_keys for key in keys], dtype=bool)
            masks[condition] = before, after
            by_prefix[condition] = {
                "targets_matching_original_prefix": int(before.sum()),
                "targets_matching_permuted_prefix": int(after.sum()),
                "lost_target_positions": original.target_positions[before & ~after].tolist(),
                "gained_target_positions": original.target_positions[~before & after].tolist(),
            }
        original_far = np.char.startswith(labels.astype(str), "far:")
        after_far_match = masks["full"][1]
        after_labels = classify_repeats(
            permuted.inputs["full"][prefix:], original.target_positions, far, n=n
        )
        unique, counts = np.unique(after_labels[original_far], return_counts=True)
        result[label_n] = {
            "original_far_targets": int(original_far.sum()),
            "original_far_target_positions": original.target_positions[original_far].tolist(),
            "original_far_still_matching_permuted_far": int((original_far & after_far_match).sum()),
            "original_far_lost_match_positions": original.target_positions[
                original_far & ~after_far_match
            ].tolist(),
            "original_far_labels_after_permutation": dict(zip(map(str, unique), map(int, counts))),
            "prefix_matches": by_prefix,
        }
    return result


def build_permutation_pair(
    ids,
    donor_ids,
    *,
    end=END,
    recent=RECENT,
    targets=TARGETS,
    block_size=BLOCK_SIZE,
    seed=SEED,
    prefix_id=0,
    keep_tail_blocks=1,
):
    original = build_histories(
        ids, donor_ids, end=end, recent_tokens=recent, target_tokens=targets, prefix_id=prefix_id
    )
    permuted, intervention = permute_old_prefix(
        original, block_size=block_size, seed=seed, keep_tail_blocks=keep_tail_blocks
    )
    bins = {
        str(n): classify_repeats(
            np.asarray(ids), original.target_positions, original.retained_start, n=n
        )
        for n in (4, 8)
    }
    intervention["repeat_label_scope"] = "original full document; unchanged in both saved triplets"
    intervention["far_repeat_survival"] = repeat_survival(original, permuted, bins)
    return {"original": original, "permuted": permuted}, bins, intervention


def _means(losses):
    means = {key: float(np.mean(losses[key], dtype=np.float64)) for key in CONDITIONS}
    return {
        "ce": means,
        "effects": {
            "N": means["short"] - means["full"],
            "B": means["foreign"] - means["full"],
            "P": means["foreign"] - means["short"],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--donor-id", help="defaults to the target's assigned donor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--chunk-size", type=int, default=256, help="target logits only")
    parser.add_argument(
        "--keep-tail-blocks",
        type=int,
        default=1,
        help="far-prefix blocks kept in place next to R; the default 1 keeps the F/R seam",
    )
    parser.add_argument("--max-memory-gb", type=float, default=14)
    parser.add_argument("--code-revision", default="unrecorded", help="provenance only")
    args = parser.parse_args(argv)
    if not 0 < args.max_memory_gb <= 14 or args.chunk_size < 1:
        parser.error("require 0 < max-memory-gb <= 14 and positive chunk size")
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    rows = {
        row["document_id"]: row
        for row in map(json.loads, args.documents.read_text(encoding="utf-8").splitlines())
    }
    row = rows[args.document_id]
    donor = rows[args.donor_id or row["donor_id"]]
    if row["document_id"] == donor["document_id"] or row["text"] == donor["text"]:
        parser.error("foreign prefix requires a distinct document")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=args.tokenizer_revision)
    ids = np.asarray(tokenizer.encode(row["text"], add_special_tokens=False), dtype=np.int64)
    donor_ids = np.asarray(
        tokenizer.encode(donor["text"], add_special_tokens=False), dtype=np.int64
    )
    plans, bins, intervention = build_permutation_pair(
        ids,
        donor_ids,
        prefix_id=int(tokenizer.eos_token_id),
        keep_tail_blocks=args.keep_tail_blocks,
    )
    # Each paired attempt has its own folder, so a retry can rerun this pair
    # without replacing completed original/permuted outputs.
    attempt = datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S_%fZ")
    output = args.output / attempt
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "public_model_prefix_permutation",
        "status": "started",
        "attempt": attempt,
        "document_id": row["document_id"],
        "donor_id": donor["document_id"],
        "intervention": intervention,
        "code_revision": args.code_revision,
        "variants": {},
        "note": "Paired descriptive comparison for one document; no population interval.",
    }

    def write_report():
        (output / "comparison.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    write_report()
    try:
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
        report.update(model=adapter.metadata, hardware=hardware)
        for variant, plan in plans.items():
            losses, measurements = {}, {}
            for condition in CONDITIONS:
                print(
                    json.dumps(
                        {"event": "score_started", "variant": variant, "condition": condition}
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
                metric["timing_scope"] = "single pass; includes first-use compilation if any"
                losses[condition], measurements[condition] = loss, metric
                print(
                    json.dumps(
                        {
                            "event": "score_finished",
                            "variant": variant,
                            "condition": condition,
                            **metric,
                        }
                    ),
                    flush=True,
                )
            path = output / f"{variant}.npz"
            save_pair(
                path,
                document_id=row["document_id"],
                arms={
                    key: {
                        "target_ids": plan.inputs[key][plan.target_starts[key] :],
                        "target_positions": plan.target_positions,
                        "losses": losses[key],
                    }
                    for key in CONDITIONS
                },
                repeat_bins=bins,
                metadata={
                    "kind": "public_model_prefix_permutation",
                    "model": adapter.metadata,
                    "hardware": hardware,
                    "selected_execution": "full",
                    "history": plan.describe(),
                    "intervention": dict(intervention, variant=variant),
                    "paired_attempt": attempt,
                    "measurements": {"full": measurements},
                    "chunk_size": args.chunk_size,
                    "donor_id": donor["document_id"],
                    "source": row.get("source"),
                    "donor_source": donor.get("source"),
                    "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    "donor_text_sha256": hashlib.sha256(donor["text"].encode()).hexdigest(),
                },
            )
            report["variants"][variant] = {
                "file": str(path),
                "measurements": measurements,
                **_means(losses),
            }
            write_report()
        saved = {key: load_pair(output / f"{key}.npz") for key in plans}
        delta = {
            key: saved["permuted"][key].astype(np.float64) - saved["original"][key]
            for key in CONDITIONS
        }
        report["paired_change_permuted_minus_original"] = _means(delta)
        report["unchanged_short_repeat_max_abs"] = float(np.max(np.abs(delta["short"])))
        report["status"] = "completed"
        write_report()
        print(
            json.dumps(
                {
                    "event": "permutation_pair_finished",
                    "report": str(output / "comparison.json"),
                    "paired_change": report["paired_change_permuted_minus_original"],
                    "short_repeat_max_abs": report["unchanged_short_repeat_max_abs"],
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
