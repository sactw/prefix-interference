"""Native-tokenizer likelihoods on identical source-character spans.

The planner uses one common textual F/R/Y partition and a character-length-matched
foreign prefix for every model. It tokenizes Y together with its history and scores
only tokens whose characters lie entirely within the target span.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import sha256_text
from .measurement import CONDITIONS, save_pair, validate_arms


@dataclass
class MatchedTextPlan:
    inputs: dict[str, np.ndarray]
    target_starts: dict[str, int]
    target_ids: np.ndarray
    target_positions: np.ndarray
    metadata: dict[str, Any]


def _backend(tokenizer):
    # Hugging Face fast tokenizers expose the exact native Rust backend used
    # by encode(add_special_tokens=False). Raw tokenizers.Tokenizer works too.
    return getattr(tokenizer, "backend_tokenizer", tokenizer)


def _split_candidates(offsets, source_shift: int, lower: int, upper: int):
    """Return source-character boundaries with no straddling token.

    Include gaps in reported offsets: SentencePiece offsets may exclude a
    leading space that the token's decoder restores. Actual prefix decoding
    below determines which candidate represents the exact string boundary.
    Repeated Unicode-byte fallback offsets are never split inside a character.
    """
    result = {}
    previous_end = 0
    for index, (start, end) in enumerate(offsets):
        low = max(previous_end + source_shift, lower)
        high = min(start + source_shift, upper)
        if low <= high:
            for boundary in range(low, high + 1):
                result[boundary] = index
        previous_end = max(previous_end, end)
    return result


def plan_matched_text(
    text: str,
    donor_text: str,
    *,
    recent_start: int,
    target_start: int,
    end: int,
    tokenizers: dict[str, Any],
    prefix_ids: dict[str, int],
) -> tuple[dict[str, MatchedTextPlan], dict[str, Any]]:
    """Prepare all models together; returns unsupported status if unalignable.

    Offsets count Python Unicode code points in the *unchanged extracted text*.
    The requested target start can move right to a shared native token boundary;
    its actual value is recorded and used for every model and history. End is
    fixed. Foreign prefix is donor_text[:recent_start], equal in character count
    to F; native token/UTF8-byte lengths need not match and are reported.

    Planning uses tokenizers and source text only. If tokenizer normalization
    prevents an exact decoded round trip, the block is marked unsupported.
    """
    if not 0 <= recent_start < target_start < end <= len(text):
        raise ValueError("require 0 <= recent_start < target_start < end <= text length")
    if len(donor_text) < recent_start:
        raise ValueError("foreign donor must contain the requested character-length prefix")
    if not tokenizers or set(tokenizers) != set(prefix_ids):
        raise ValueError("each named tokenizer needs its native boundary token ID")
    if text == donor_text:
        raise ValueError("foreign donor must be a distinct document")
    strings = {
        "full": text[:end],
        "short": text[recent_start:end],
        "foreign": donor_text[:recent_start] + text[recent_start:end],
    }
    shifts = {"full": 0, "short": recent_start, "foreign": 0}
    report = {
        "protocol": "matched_text_span_v1",
        "requested_source_char_spans": {
            "far": [0, recent_start],
            "recent": [recent_start, target_start],
            "target": [target_start, end],
        },
        "offset_unit": "Python Unicode code points in original extracted text",
        "text_sha256": sha256_text(text),
        "donor_text_sha256": sha256_text(donor_text),
        "foreign_policy": "single donor prefix matched to far prefix in characters; no separator",
        "model_names": list(tokenizers),
    }
    encoded, candidates, backends = {}, {}, {}
    common = None
    for name, tokenizer in tokenizers.items():
        backend = backends[name] = _backend(tokenizer)
        encoded[name], candidates[name] = {}, {}
        for condition in CONDITIONS:
            value = strings[condition]
            encoding = backend.encode(value, add_special_tokens=False)
            ids = list(encoding.ids)
            # Exact round trip is needed for source-byte denominators. Some
            # tokenizers normalize whitespace/Unicode; do not hide that change.
            decoded = backend.decode(ids, skip_special_tokens=False)
            if decoded != value:
                report.update(
                    status="unsupported_text_normalization",
                    model=name,
                    condition=condition,
                    decoded_characters=len(decoded),
                    input_characters=len(value),
                )
                return {}, report
            encoded[name][condition] = ids
            boundaries = _split_candidates(
                encoding.offsets, shifts[condition], target_start, end - 1
            )
            candidates[name][condition] = boundaries
            common = set(boundaries) if common is None else common.intersection(boundaries)

    selected = None
    for boundary in sorted(common or ()):
        valid = True
        for name, backend in backends.items():
            suffix = None
            for condition in CONDITIONS:
                index = candidates[name][condition][boundary]
                ids = encoded[name][condition]
                prefix = backend.decode(ids[:index], skip_special_tokens=False)
                if prefix != strings[condition][: boundary - shifts[condition]]:
                    valid = False
                    break
                current_suffix = ids[index:]
                if suffix is not None and current_suffix != suffix:
                    valid = False
                    break
                suffix = current_suffix
            if not valid:
                break
        if valid:
            selected = boundary
            break
    if selected is None:
        report.update(status="unsupported_shared_boundary")
        return {}, report

    target_text = text[selected:end]
    report.update(
        status="ready",
        target_start_adjustment_characters=selected - target_start,
        source_char_spans={
            "far": [0, recent_start],
            "recent": [recent_start, selected],
            "target": [selected, end],
        },
        target_text_sha256=sha256_text(target_text),
        target_characters=len(target_text),
        target_utf8_bytes=len(target_text.encode("utf-8")),
        far_prefix_characters=recent_start,
        far_prefix_utf8_bytes=len(text[:recent_start].encode("utf-8")),
        foreign_prefix_utf8_bytes=len(donor_text[:recent_start].encode("utf-8")),
        input_characters={key: len(value) for key, value in strings.items()},
        input_utf8_bytes={key: len(value.encode("utf-8")) for key, value in strings.items()},
        normalization="exact native full-string and prefix decoding; no target-only encoding",
    )
    plans = {}
    for name in tokenizers:
        starts = {key: candidates[name][key][selected] for key in CONDITIONS}
        target_ids = np.asarray(encoded[name]["full"][starts["full"] :], dtype=np.int64)
        inputs = {
            key: np.asarray([prefix_ids[name], *encoded[name][key]], dtype=np.int64)
            for key in CONDITIONS
        }
        metadata = dict(
            report,
            model_name=name,
            prefix_id=int(prefix_ids[name]),
            target_tokens=len(target_ids),
            documentary_input_tokens={key: len(value) - 1 for key, value in inputs.items()},
            conditioning_tokens={key: starts[key] for key in CONDITIONS},
            target_starts={key: value + 1 for key, value in starts.items()},
        )
        plans[name] = MatchedTextPlan(
            inputs,
            metadata["target_starts"],
            target_ids,
            np.arange(starts["full"], len(encoded[name]["full"]), dtype=np.int64),
            metadata,
        )
    return plans, report


def summarize_matched_losses(plan: MatchedTextPlan, losses: dict[str, np.ndarray]) -> dict:
    """Comparable total canonical-token NLL per source byte/character.

    Native CE per token remains useful within each model. Model contrasts use
    identical text blocks and the same byte denominator. Canonical tokenizations
    are scored; alternative tokenizations of a string are not marginalized.
    """
    validate_arms(
        {
            key: {
                "target_ids": plan.target_ids,
                "target_positions": plan.target_positions,
                "losses": losses[key],
            }
            for key in CONDITIONS
        }
    )
    totals = {key: float(np.asarray(losses[key], dtype=np.float64).sum()) for key in CONDITIONS}
    totals.update(
        N=totals["short"] - totals["full"],
        content=totals["foreign"] - totals["full"],
        prefix_penalty=totals["foreign"] - totals["short"],
    )
    counts = {
        "tokens": len(plan.target_ids),
        "characters": plan.metadata["target_characters"],
        "utf8_bytes": plan.metadata["target_utf8_bytes"],
    }
    return {
        "counts": counts,
        "total_nll_nats": totals,
        "nats_per_native_token": {key: value / counts["tokens"] for key, value in totals.items()},
        "nats_per_character": {key: value / counts["characters"] for key, value in totals.items()},
        "nats_per_utf8_byte": {key: value / counts["utf8_bytes"] for key, value in totals.items()},
        "bits_per_utf8_byte": {
            key: value / counts["utf8_bytes"] / np.log(2) for key, value in totals.items()
        },
    }


def score_matched_text(adapter, plan: MatchedTextPlan, *, method="score_full", chunk_size=256):
    """Score a plan with a loaded adapter; each history starts from a fresh state."""
    score = getattr(adapter, method)
    losses = {
        key: score(plan.inputs[key], plan.target_starts[key], chunk_size=chunk_size)
        for key in CONDITIONS
    }
    return losses, summarize_matched_losses(plan, losses)


def save_matched_text(path, *, document_id: str, plan: MatchedTextPlan, losses, metadata):
    """Save in the paired-loss NPZ format with matched-text metadata.

    Token n-gram strata are omitted because token identities differ across
    tokenizers.
    """
    summary = summarize_matched_losses(plan, losses)
    save_pair(
        path,
        document_id=document_id,
        arms={
            key: {
                "target_ids": plan.target_ids,
                "target_positions": plan.target_positions,
                "losses": losses[key],
            }
            for key in CONDITIONS
        },
        repeat_bins={},
        metadata=dict(metadata, history=plan.metadata, matched_text=summary),
    )


def save_text_plan(path, plan: MatchedTextPlan):
    """Serialize the small native ID plan, without another copy of source text."""
    arrays = {f"{key}_input_ids": value for key, value in plan.inputs.items()}
    arrays.update(
        target_ids=plan.target_ids,
        target_positions=plan.target_positions,
        metadata_json=np.asarray(json.dumps(plan.metadata, allow_nan=False, sort_keys=True)),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_text_plan(path) -> MatchedTextPlan:
    """Read native IDs and validate the concrete scored suffix in every history."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        inputs = {key: saved[f"{key}_input_ids"] for key in CONDITIONS}
        target_ids, target_positions = saved["target_ids"], saved["target_positions"]
    starts = metadata["target_starts"]
    for key in CONDITIONS:
        if not 1 <= starts[key] < len(inputs[key]):
            raise ValueError(f"{key}: invalid target start in input plan")
    validate_arms(
        {
            key: {
                "target_ids": inputs[key][starts[key] :],
                "target_positions": target_positions,
                "losses": np.zeros(len(target_ids)),
            }
            for key in CONDITIONS
        }
    )
    if not np.array_equal(inputs["full"][starts["full"] :], target_ids):
        raise ValueError("serialized target IDs differ from the input suffix")
    return MatchedTextPlan(inputs, starts, target_ids, target_positions, metadata)
