"""Attenuate only Mamba's old SSM cache before an identical recent suffix.

The intervention changes cached SSM tensors while preserving checkpoint
weights. Alpha zero retains convolution caches and differs from fresh short
history.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np

from .matched_text import load_text_plan
from .measurement import load_pair


def common_suffix_boundaries(plan):
    """Preserve the scored inputs, moving past any tokenizer seam at F/R.

    Matched-text plans guarantee the target suffix, but a token can straddle
    the textual F/R boundary. Use the longest identical native-token suffix
    of full/foreign/short, excluding their leading boundary token.
    """
    limit = min(len(ids) - 1 for ids in plan.inputs.values())
    shared = 0
    for distance in range(1, limit + 1):
        if len({int(ids[-distance]) for ids in plan.inputs.values()}) != 1:
            break
        shared = distance
    recent = shared - len(plan.target_ids)
    if recent < 1:
        raise ValueError("the histories need an identical recent token before Y")
    return {name: len(ids) - shared for name, ids in plan.inputs.items()}, recent


def clone_cache(state, alpha):
    """Independent native cache branch; scale SSM tensors, preserve convs."""
    branch = copy.copy(state)
    branch.key_value_memory_dict = {}
    for layer, (conv, ssm) in state.key_value_memory_dict.items():
        conv_copy, ssm_copy = conv.clone(), ssm.clone()
        if alpha != 1.0:
            ssm_copy.mul_(alpha)
        branch.key_value_memory_dict[layer] = (conv_copy, ssm_copy)
    lengths = getattr(state, "lengths_per_sample", None)
    if lengths is not None:
        branch.lengths_per_sample = lengths.clone()
    return branch


def continuation_losses(
    ids, target_start, boundary, state, forward, readout, concatenate, logits_chunk=256
):
    """Native one-token continuation; project only target-predicting rows."""
    if state.seqlen_offset != boundary or not 1 <= boundary < target_start:
        raise ValueError("cache offset must mark the prefix before the recent suffix")
    losses = np.empty(len(ids) - target_start, dtype=np.float32)
    hidden_rows, targets, written = [], [], 0

    def flush():
        nonlocal written
        values = readout(concatenate(hidden_rows), np.asarray(targets, dtype=np.int64))
        losses[written : written + len(targets)] = values
        written += len(targets)
        hidden_rows.clear()
        targets.clear()

    for position in range(boundary, len(ids) - 1):
        hidden = forward(ids[position : position + 1], state)
        state.seqlen_offset += 1
        if position + 1 >= target_start:
            hidden_rows.append(hidden)
            targets.append(int(ids[position + 1]))
            if len(targets) == logits_chunk:
                flush()
    if targets:
        flush()
    return losses


def difference(values, reference):
    delta = np.asarray(values, dtype=np.float64) - reference
    return {
        "mean_signed_nats": float(delta.mean()),
        "mean_absolute_nats": float(np.abs(delta).mean()),
        "max_absolute_nats": float(np.abs(delta).max()),
    }


def parity(values, reference):
    comparison = difference(values, reference)
    comparison["within_provisional_tolerances"] = (
        comparison["mean_absolute_nats"] <= 1e-4 and comparison["max_absolute_nats"] <= 1e-3
    )
    return comparison


def _reference(plan, path):
    result = load_pair(path)
    if result["document_id"] != plan.metadata["document_id"]:
        raise ValueError("reference is for another document")
    for key in ("target_ids", "target_positions"):
        if not np.array_equal(result[key], getattr(plan, key)):
            raise ValueError(f"reference {key} do not match the scored plan")
    history = result["metadata"]["history"]
    for key in (
        "donor_id",
        "text_sha256",
        "donor_text_sha256",
        "target_text_sha256",
        "source_char_spans",
        "target_starts",
        "documentary_input_tokens",
    ):
        if history[key] != plan.metadata[key]:
            raise ValueError(f"reference input differs: {key}")
    for key in ("model_id", "revision", "tokenizer_revision"):
        if result["metadata"]["model"][key] != plan.metadata["model_reference"][key]:
            raise ValueError(f"reference model differs: {key}")
    return {name: result[name] for name in ("full", "short", "foreign")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="existing matched-text Mamba NPZ; values are not overwritten",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history", choices=("full", "foreign", "both"), default="both")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1, 0.5, 0])
    parser.add_argument("--chunk-size", type=int, default=256, help="target-logit rows only")
    parser.add_argument("--max-memory-gb", type=float, default=14)
    parser.add_argument("--code-revision", default="unrecorded", help="provenance only")
    args = parser.parse_args(argv)
    if (
        not 0 < args.max_memory_gb <= 14
        or args.chunk_size < 1
        or 1 not in args.alphas
        or any(not 0 <= a <= 1 for a in args.alphas)
    ):
        parser.error("require memory in (0,14], positive chunk, and alphas in [0,1] including 1")
    alphas = [1.0, *dict.fromkeys(a for a in args.alphas if a != 1)]
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    plan = load_text_plan(args.plan)
    if plan.metadata["model_name"] != "mamba":
        parser.error("this intervention is for the native Mamba cache only")
    reference = _reference(plan, args.reference)
    boundaries, recent = common_suffix_boundaries(plan)
    conditions = ("full", "foreign") if args.history == "both" else (args.history,)
    report = {
        "kind": "mamba_old_state_boundary_attenuation",
        "status": "started",
        "document_id": plan.metadata["document_id"],
        "history": plan.metadata,
        "code_revision": args.code_revision,
        "reference": str(args.reference),
        "parity_tolerances": {
            "mean_absolute_nats": 1e-4,
            "max_absolute_nats": 1e-3,
            "scope": "alpha1/fresh-short versus saved fused reference; diagnostic comparison",
        },
        "intervention": {
            "alphas": alphas,
            "location": "before identical native R/Y suffix",
            "shared_recent_tokens": recent,
            "prefix_tokens_including_boundary": boundaries,
            "short_initial_tokens_outside_shared_suffix": boundaries["short"] - 1,
            "scaled": "all layers' SSM cache tensors only",
            "preserved": "cloned convolution caches, offsets, model weights and inputs",
            "alpha_zero_is_short": False,
            "note": "Tokenizer F/R seams may move the boundary by a few tokens; original inputs are unchanged.",
        },
        "measurements": {},
        "comparisons": {},
    }
    arrays = {
        "target_ids": plan.target_ids,
        "target_positions": plan.target_positions,
        **{f"reference_{key}_losses": value for key, value in reference.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        with args.output.open("wb") as handle:
            np.savez_compressed(
                handle, **arrays, metadata_json=np.asarray(json.dumps(report, allow_nan=False))
            )

    save()
    try:
        from .mamba_adapter import MambaAdapter

        begin = time.perf_counter()
        adapter = MambaAdapter(
            **{
                key: plan.metadata["model_reference"][key]
                for key in ("model_id", "revision", "tokenizer_revision")
            },
            dtype="float32",
            max_memory_gb=args.max_memory_gb,
        )
        torch = adapter._torch
        torch.cuda.synchronize()
        report.update(
            model=adapter.metadata,
            hardware={
                "gpu": torch.cuda.get_device_name(),
                "load_seconds": time.perf_counter() - begin,
            },
        )

        def timed(call):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            value = call()
            torch.cuda.synchronize()
            return value, {
                "seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }

        def forward(ids, state):
            tokens = torch.as_tensor(ids, dtype=torch.long, device=adapter.device)[None, :]
            return adapter.base(tokens, inference_params=state)

        with torch.inference_mode():
            short, metric = timed(
                lambda: adapter.score_full(
                    plan.inputs["short"], plan.target_starts["short"], args.chunk_size
                )
            )
            arrays["fresh_short_losses"] = short
            report["measurements"]["fresh_short"] = metric
            report["comparisons"]["short_vs_reference"] = parity(short, reference["short"])
            save()
            for condition in conditions:
                ids, boundary = plan.inputs[condition], boundaries[condition]
                state = adapter._InferenceParams(max_seqlen=len(ids), max_batch_size=1)

                def prefill():
                    hidden = forward(ids[:boundary], state)
                    state.seqlen_offset = boundary
                    del hidden

                _, metric = timed(prefill)
                metric.update(
                    processed_tokens=boundary,
                    cache_layers=len(state.key_value_memory_dict),
                    cache_bytes=sum(
                        t.numel() * t.element_size()
                        for pair in state.key_value_memory_dict.values()
                        for t in pair
                    ),
                )
                report["measurements"][f"{condition}_prefill"] = metric
                save()
                for alpha in alphas:
                    label = f"{condition}_alpha{alpha:g}"
                    print(
                        json.dumps({"event": "state_branch_started", "branch": label}), flush=True
                    )

                    def run_branch():
                        branch = clone_cache(state, alpha)
                        return continuation_losses(
                            ids,
                            plan.target_starts[condition],
                            boundary,
                            branch,
                            forward,
                            adapter._readout,
                            lambda rows: torch.cat(rows, dim=1),
                            args.chunk_size,
                        )

                    loss, metric = timed(run_branch)
                    arrays[f"{label}_losses"] = loss
                    if not np.isfinite(loss).all():
                        raise FloatingPointError(
                            f"nonfinite target loss in {label}; array retained"
                        )
                    baseline = arrays[f"{condition}_alpha1_losses"]
                    report["measurements"][label] = dict(
                        metric,
                        continuation_input_tokens=len(ids) - 1 - boundary,
                        target_tokens=len(loss),
                        ce_nats_per_token=float(loss.astype(np.float64).mean()),
                        delta_vs_cached_alpha1=difference(loss, baseline),
                        delta_vs_fused_reference=difference(loss, reference[condition]),
                    )
                    if alpha == 1:
                        report["comparisons"][f"{condition}_alpha1_vs_reference"] = parity(
                            loss, reference[condition]
                        )
                    save()
                    print(
                        json.dumps(
                            {
                                "event": "state_branch_finished",
                                "branch": label,
                                **report["measurements"][label],
                            }
                        ),
                        flush=True,
                    )
                state = None
        if args.history == "both":
            report["effects_nats_per_native_token"] = {}
            for alpha in alphas:
                full = arrays[f"full_alpha{alpha:g}_losses"].astype(np.float64)
                foreign = arrays[f"foreign_alpha{alpha:g}_losses"].astype(np.float64)
                report["effects_nats_per_native_token"][f"alpha{alpha:g}"] = {
                    "N": float((short - full).mean()),
                    "content": float((foreign - full).mean()),
                    "prefix_penalty": float((foreign - short).mean()),
                }
        report.update(
            status="completed",
            interpretation=(
                "Paired cache attenuation; alpha1-versus-fused discrepancies bound numerical interpretation. "
                "Alpha0 retains old convolution state and is not a fresh short history."
            ),
        )
        save()
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
