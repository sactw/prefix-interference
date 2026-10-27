"""Model-independent paired likelihood and repeat-distance measurements.

All positions are zero-based, inclusive token indexes in the original document.
Loss arrays must score the same target IDs in the same order in all three arms;
the scorer or saved-loss loader establishes that identity before calling here.
This module uses only NumPy and the standard library.
"""

from __future__ import annotations

from collections import deque

import numpy as np


DISTANCE_EDGES = (1024, 4096, 16384, 65536, 262144, 524288, 1048576)
EFFECTS = ("N", "content", "prefix_penalty")
CONDITIONS = ("full", "short", "foreign")


def repeat_labels(distance_edges=DISTANCE_EDGES):
    """Names of the mutually exclusive strata; far intervals are (low, high]."""
    edges = tuple(distance_edges)
    return (
        "incomplete_ngram",
        "never",
        "recent",
        "boundary",
        *(f"far:({lo},{hi}]" for lo, hi in zip((0,) + edges, edges)),
        f"far:({edges[-1] if edges else 0},inf)",
    )


def _integer_vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return array


def classify_repeats(ids, target_positions, retained_start, n=4, distance_edges=DISTANCE_EDGES):
    """Classify each target by the nearest earlier matching n-gram.

    The n-gram ends at the target token. ``recent`` means that the complete
    earlier occurrence starts in the retained short-history suffix.
    ``boundary`` means only part of that occurrence is retained; ``far`` means
    its end is excluded. ``never`` is document-local, not a pretraining claim.
    Overlapping occurrences count. Only positions through the last requested
    target are scanned, and the current occurrence is recorded after lookup.

    N-grams are compared as exact token tuples. Runtime is O(L*n), where L
    ends at the final target.
    """
    tokens = _integer_vector(ids, "ids")
    positions = _integer_vector(target_positions, "target_positions")
    edges = tuple(distance_edges)
    if (
        not isinstance(n, (int, np.integer))
        or n < 1
        or not isinstance(retained_start, (int, np.integer))
        or retained_start < 0
        or any(not isinstance(x, (int, np.integer)) or x <= 0 for x in edges)
        or any(a >= b for a, b in zip(edges, edges[1:]))
    ):
        raise ValueError(
            "n must be positive, retained_start nonnegative, and edges increasing positive integers"
        )
    if positions.size and (
        int(positions.min()) < retained_start or int(positions.max()) >= len(tokens)
    ):
        raise ValueError("target positions must lie within the retained suffix of ids")
    labels = repeat_labels(edges)
    output = np.empty(len(positions), dtype=f"<U{max(map(len, labels))}")
    if not positions.size:
        return output

    wanted = {}
    for index, position in enumerate(positions):
        wanted.setdefault(int(position), []).append(index)
    previous = {}
    window = deque(maxlen=int(n))
    for end in range(int(positions.max()) + 1):
        window.append(int(tokens[end]))
        key = tuple(window) if len(window) == n else None
        earlier_end = previous.get(key) if key is not None else None
        if end in wanted:
            if key is None:
                label = "incomplete_ngram"
            elif earlier_end is None:
                label = "never"
            elif earlier_end - n + 1 >= retained_start:
                label = "recent"
            elif earlier_end >= retained_start:
                label = "boundary"
            else:
                distance = end - earlier_end
                interval = int(np.searchsorted(edges, distance, side="left"))
                low = 0 if interval == 0 else edges[interval - 1]
                label = (
                    f"far:({low},{edges[interval]}]"
                    if interval < len(edges)
                    else f"far:({low},inf)"
                )
            output[wanted[end]] = label
        if key is not None:
            previous[key] = end
    return output


def _validated_blocks(documents):
    """Check that losses, positions and labels align with each block's targets."""
    blocks = []
    bin_names = None
    for row in documents:
        document_id = row["document_id"]
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a nonempty string")
        targets = _integer_vector(row["target_ids"], "target_ids")
        positions = _integer_vector(row["target_positions"], "target_positions")
        bins = row["repeat_bins"]
        if not isinstance(bins, dict) or "4" not in bins:
            raise ValueError("repeat_bins must contain the '4' n-gram partition")
        if bin_names is None:
            bin_names = set(bins)
        elif set(bins) != bin_names:
            raise ValueError("all blocks must supply the same repeat-bin partitions")
        arrays = {key: np.asarray(row[key], dtype=np.float64) for key in CONDITIONS}
        labels = {str(key): np.asarray(value) for key, value in bins.items()}
        all_arrays = [positions, *arrays.values(), *labels.values()]
        if any(value.ndim != 1 or len(value) != len(targets) for value in all_arrays):
            raise ValueError(
                f"{document_id}: losses, positions and bins must align with target_ids"
            )
        if (
            any(not np.isfinite(value).all() for value in arrays.values())
            or (positions.size and int(positions.min()) < 0)
            or any(value.dtype.kind not in "US" for value in labels.values())
            or any(np.any(value == "") for value in labels.values())
        ):
            raise ValueError(
                f"{document_id}: losses must be finite, positions nonnegative and labels nonempty strings"
            )
        labels = {key: value.astype(str, copy=False) for key, value in labels.items()}
        full, short, foreign = (arrays[key] for key in CONDITIONS)
        effects = np.stack((short - full, foreign - full, foreign - short), axis=1)
        blocks.append((document_id, len(targets), arrays, effects, labels))
    return blocks


def _ratio_estimate(sums, counts, draws):
    """Paired document bootstrap, retaining the multiplicity of each draw."""
    count = int(counts.sum())
    draw_counts = counts[draws].sum(axis=1)
    nonempty = draw_counts > 0
    draw_sums = sums[draws].sum(axis=1)
    ratios = draw_sums[nonempty] / draw_counts[nonempty]
    return {
        "mean": float(sums.sum(dtype=np.float64) / count) if count else None,
        "ci95": [float(x) for x in np.percentile(ratios, [2.5, 97.5])] if ratios.size else None,
        "bootstrap_valid": int(nonempty.sum()),
        "bootstrap_empty": int(len(draws) - nonempty.sum()),
    }


def summarize(documents: list[dict], bootstrap_samples=1000, seed=0) -> dict:
    """Return JSON-safe token-weighted CE, paired effects, and repeat strata.

    Each input dict supplies document_id, target_ids, target_positions, full,
    short, foreign and repeat_bins (a '4' partition, optionally also '8').
    Multiple blocks of a document form one bootstrap cluster. Every supplied
    target row counts once, so blocks should not overlap.

    N = short - full, content = foreign - full, and prefix_penalty = foreign -
    short, all in nats/target. A bin's weighted contribution divides its summed
    effects by the entire target count, so contributions sum to the overall
    effects. Bin means skip draws in which the bin is empty and report how many;
    contribution draws count an absent bin as zero. Intervals are null when
    no draw is valid. Sums use float64.
    """
    if not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    blocks = _validated_blocks(documents)
    by_document = {}
    partitions = set()
    observed_labels = {}
    for document_id, count, ce, effects, bins in blocks:
        partitions.update(bins)
        cluster = by_document.setdefault(
            document_id,
            {"count": 0, "ce": np.zeros(3), "effects": np.zeros(3), "bins": {}},
        )
        cluster["count"] += count
        cluster["ce"] += [ce[key].sum(dtype=np.float64) for key in CONDITIONS]
        cluster["effects"] += effects.sum(axis=0, dtype=np.float64)
        for n, labels in bins.items():
            observed_labels.setdefault(n, set()).update(labels.astype(str).tolist())
            for label in np.unique(labels):
                label = str(label)
                selected = labels == label
                entry = cluster["bins"].setdefault((n, label), [0, np.zeros(3)])
                entry[0] += int(selected.sum())
                entry[1] += effects[selected].sum(axis=0, dtype=np.float64)

    # Documents without scored targets carry no sampling weight.
    clusters = [value for value in by_document.values() if value["count"]]
    counts = np.asarray([value["count"] for value in clusters], dtype=np.int64)
    ce_sums = np.asarray([value["ce"] for value in clusters], dtype=np.float64).reshape(-1, 3)
    effect_sums = np.asarray([value["effects"] for value in clusters], dtype=np.float64).reshape(
        -1, 3
    )
    rng = np.random.default_rng(seed)
    draws = (
        rng.integers(0, len(clusters), size=(bootstrap_samples, len(clusters)))
        if clusters
        else np.empty((bootstrap_samples, 0), dtype=np.int64)
    )
    total = int(counts.sum())
    report = {
        "targets": total,
        "documents": len(clusters),
        "blocks": len(blocks),
        "aggregation": "token_weighted",
        "bootstrap": {
            "method": "paired_document_cluster_percentile_ratio",
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "confidence": 0.95,
        },
        "ce": {
            key: _ratio_estimate(ce_sums[:, i], counts, draws) for i, key in enumerate(CONDITIONS)
        },
        "effects": {
            key: _ratio_estimate(effect_sums[:, i], counts, draws) for i, key in enumerate(EFFECTS)
        },
        "repeat_bins": {},
    }
    for n in sorted(partitions or {"4"}):
        labels = list(repeat_labels())
        labels += sorted(observed_labels.get(n, set()).difference(labels))
        report["repeat_bins"][n] = {}
        for label in labels:
            entries = [cluster["bins"].get((n, label), (0, np.zeros(3))) for cluster in clusters]
            bin_counts = np.asarray([entry[0] for entry in entries], dtype=np.int64)
            bin_sums = np.asarray([entry[1] for entry in entries], dtype=np.float64).reshape(-1, 3)
            bin_total = int(bin_counts.sum())
            report["repeat_bins"][n][label] = {
                "targets": bin_total,
                "documents": int(np.count_nonzero(bin_counts)),
                "share": float(bin_total / total) if total else None,
                "effects": {
                    key: _ratio_estimate(bin_sums[:, i], bin_counts, draws)
                    for i, key in enumerate(EFFECTS)
                },
                "weighted_contribution": {
                    key: _ratio_estimate(bin_sums[:, i], counts, draws)
                    for i, key in enumerate(EFFECTS)
                },
            }
    return report
