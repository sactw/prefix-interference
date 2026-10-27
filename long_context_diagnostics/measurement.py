"""Matched histories and portable paired-loss records; no model dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np


CONDITIONS = ("full", "short", "foreign")


def token_vector(values, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(f"{name} must be a one-dimensional nonnegative integer array")
    return values.astype(np.int64, copy=False)


@dataclass
class Histories:
    inputs: dict[str, np.ndarray]
    target_starts: dict[str, int]
    target_ids: np.ndarray
    target_positions: np.ndarray
    retained_start: int
    recent_tokens: int
    end: int
    prefix_id: int | None

    def describe(self) -> dict:
        return {
            "protocol": "fixed_target_block_v1",
            "end_exclusive": self.end,
            "target_tokens": len(self.target_ids),
            "recent_prefix_tokens": self.recent_tokens,
            "far_prefix_tokens": self.retained_start,
            "retained_start": self.retained_start,
            "prefix_id": self.prefix_id,
            "input_lengths": {k: len(v) for k, v in self.inputs.items()},
            "target_starts": self.target_starts,
            "donor_policy": "single_document_prefix_no_separator",
        }


def build_histories(
    ids,
    donor_ids,
    *,
    end: int,
    recent_tokens: int,
    target_tokens: int,
    prefix_id: int | None = None,
) -> Histories:
    """C documentary tokens precede a fixed T-target block; C is not a window.

    An optional identical special prefix is outside all documentary counts.
    Donor IDs must come from a separately selected unrelated document.
    """
    ids = token_vector(ids, "ids")
    donor_ids = token_vector(donor_ids, "donor_ids")
    if not (
        target_tokens > 0
        and recent_tokens >= 0
        and target_tokens + recent_tokens <= end <= len(ids)
    ):
        raise ValueError("require 0 < T, 0 <= C and C + T <= E <= document length")
    start = end - target_tokens
    retained_start = start - recent_tokens
    if len(donor_ids) < retained_start:
        raise ValueError("selected donor is too short; select a longer donor or smaller E")
    prefix = np.empty(0, dtype=np.int64)
    if prefix_id is not None:
        prefix = token_vector([prefix_id], "prefix_id")
    inputs = {
        "full": np.concatenate((prefix, ids[:end])),
        "short": np.concatenate((prefix, ids[retained_start:end])),
        "foreign": np.concatenate((prefix, donor_ids[:retained_start], ids[retained_start:end])),
    }
    starts = {name: len(seq) - target_tokens for name, seq in inputs.items()}
    if min(starts.values()) < 1:
        raise ValueError("first target needs a preceding token; provide C >= 1 or prefix_id")
    return Histories(
        inputs,
        starts,
        ids[start:end].copy(),
        np.arange(start, end, dtype=np.int64),
        retained_start,
        recent_tokens,
        end,
        prefix_id,
    )


def evaluate_histories(plan: Histories, score: Callable) -> dict[str, dict]:
    """score(input_ids, target_start) must start from a fresh state on every call."""
    arms = {}
    for name in CONDITIONS:
        seq = plan.inputs[name]
        target_start = plan.target_starts[name]
        arms[name] = {
            "target_ids": seq[target_start:].copy(),
            "target_positions": plan.target_positions.copy(),
            "losses": np.asarray(score(seq, target_start), dtype=np.float32),
        }
    validate_arms(arms)
    return arms


def validate_arms(arms: dict[str, dict]) -> None:
    """Validate matching target IDs and positions across history arms."""
    if set(arms) != set(CONDITIONS):
        raise ValueError("one full, short and foreign arm is required")
    reference_ids = token_vector(arms["full"]["target_ids"], "full target_ids")
    reference_pos = token_vector(arms["full"]["target_positions"], "full target_positions")
    if not len(reference_ids) or reference_pos.shape != reference_ids.shape:
        raise ValueError("target IDs and positions must have the same nonzero length")
    if np.any(np.diff(reference_pos) <= 0):
        raise ValueError("target positions must be strictly increasing within a block")
    for name in CONDITIONS:
        arm = arms[name]
        target_ids = token_vector(arm["target_ids"], f"{name} target_ids")
        positions = token_vector(arm["target_positions"], f"{name} target_positions")
        if not np.array_equal(target_ids, reference_ids) or not np.array_equal(
            positions, reference_pos
        ):
            raise ValueError(f"{name}: target identity/order differs from full")
        losses = np.asarray(arm["losses"])
        if (
            losses.shape != reference_ids.shape
            or not np.all(np.isfinite(losses))
            or np.any(losses < 0)
        ):
            raise ValueError(f"{name}: losses must be finite nonnegative per-target NLLs")


def save_pair(
    path: str | Path,
    *,
    document_id: str,
    arms: dict[str, dict],
    repeat_bins: dict[str, np.ndarray],
    metadata: dict,
) -> None:
    """One compact self-contained block; completed blocks survive later retries."""
    validate_arms(arms)
    if not document_id:
        raise ValueError("document_id is required for document-level uncertainty")
    arrays = {}
    for name, arm in arms.items():
        arrays[f"{name}_target_ids"] = np.asarray(arm["target_ids"], dtype=np.int64)
        arrays[f"{name}_target_positions"] = np.asarray(arm["target_positions"], dtype=np.int64)
        arrays[f"{name}_losses"] = np.asarray(arm["losses"], dtype=np.float32)
    for n, labels in repeat_bins.items():
        labels = np.asarray(labels, dtype=str)
        if labels.shape != arrays["full_target_ids"].shape:
            raise ValueError(f"repeat bins for {n} must align with targets")
        arrays[f"repeat_{n}"] = labels
    header = dict(metadata, schema_version=1, document_id=str(document_id))
    arrays["metadata_json"] = np.asarray(json.dumps(header, allow_nan=False, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_pair(path: str | Path) -> dict:
    """Load independently saved arms and check identity before calculating effects."""
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        arms = {
            name: {
                "target_ids": saved[f"{name}_target_ids"],
                "target_positions": saved[f"{name}_target_positions"],
                "losses": saved[f"{name}_losses"],
            }
            for name in CONDITIONS
        }
        validate_arms(arms)
        if metadata.get("schema_version") != 1 or not metadata.get("document_id"):
            raise ValueError("expected paired-loss schema 1 with document_id")
        result = {
            "document_id": metadata["document_id"],
            "target_ids": arms["full"]["target_ids"],
            "target_positions": arms["full"]["target_positions"],
            "repeat_bins": {
                key[7:]: saved[key] for key in saved.files if key.startswith("repeat_")
            },
            "metadata": metadata,
        }
        result.update({name: arms[name]["losses"] for name in CONDITIONS})
    return result
