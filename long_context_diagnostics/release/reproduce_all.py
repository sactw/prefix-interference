"""Recompute every paper result from the released per-target loss arrays.

Needs only NumPy, plus Matplotlib for --figures. No model is loaded and no GPU
is used; reports are written under --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ..ladder_readout import summarize_ladder
from ..matched_panel_readout import summarize_matched_panel
from ..measurement import CONDITIONS, load_pair
from ..panel_readout import average_donors, summarize_panel
from ..permutation_readout import summarize_permutation
from ..readout import summarize
from ..second_donor_readout import reproduce as reproduce_second_donor
from ..state_attenuation_readout import summarize as summarize_state_attenuation
from ..three_model_readout import reproduce as reproduce_three_models


SAMPLES, SEED = 2000, 20261020


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(path)}), flush=True)


def _rows(folder):
    paths = sorted(folder.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No per-target NPZ results in {folder}")
    return [load_pair(path) for path in paths]


def _cross_donor(original, alternate):
    """The saved cyclic-donor contrast, using the existing paired bootstrap.

    Average the two original donors within each target first. The difference
    arrays then retain one row per scored target and one bootstrap cluster per
    document. No new scores or independent donor observations are introduced.
    """
    old, _, _ = average_donors(original)
    new, _, _ = average_donors(alternate)
    old = {row["document_id"]: row for row in old}
    new = {row["document_id"]: row for row in new}
    if set(old) != set(new):
        raise ValueError("The cyclic-donor comparison requires the same target documents")
    differences = []
    for document in sorted(new):
        before, after = old[document], new[document]
        for key in ("target_ids", "target_positions"):
            np.testing.assert_array_equal(before[key], after[key])
        for order in before["repeat_bins"]:
            np.testing.assert_array_equal(before["repeat_bins"][order], after["repeat_bins"][order])
        differences.append({**after, **{key: after[key] - before[key] for key in CONDITIONS}})
    return {
        "comparison": "cyclic cross-document donor minus two GAO donors averaged within target",
        "documents": len(new),
        "targets": sum(len(row["target_ids"]) for row in new.values()),
        "new_records": len(alternate),
        "old_records": len(original),
        "new": summarize([new[d] for d in sorted(new)], SAMPLES, SEED),
        "old_two_GAO_average": summarize([old[d] for d in sorted(old)], SAMPLES, SEED),
        "paired_new_minus_old": summarize(differences, SAMPLES, SEED),
        "interpretation": "Pointwise document bootstrap conditional on the fixed cyclic donor map; not donor-population uncertainty.",
    }


def reproduce(study, output, *, figures=False):
    study, output = Path(study).resolve(), Path(output).resolve()
    reports = {}

    native = _rows(study / "fresh_panel/losses")
    reports["fresh_panel"] = summarize_panel(
        native, SAMPLES, SEED, expected_documents=8, expected_donors=2
    )
    _write(output / "fresh_panel/RESULTS.json", reports["fresh_panel"])

    local = study / "local_control/losses"
    reports["local_control"] = summarize_panel(
        _rows(local), SAMPLES, SEED, expected_documents=8, expected_donors=2
    )
    _write(output / "local_control/RESULTS.json", reports["local_control"])

    reports["length_ladder"] = summarize_ladder(
        _rows(study / "length_ladder/losses"),
        donor_id="GAO-25-107705__donor",
        fars=[511, 6655, 31232],
        baseline_far=511,
        bootstrap_samples=SAMPLES,
        seed=SEED,
    )
    _write(output / "length_ladder/RESULTS.json", reports["length_ladder"])

    reports["permutation"] = summarize_permutation(
        _rows(study / "permutation/losses"), bootstrap_samples=SAMPLES, seed=SEED
    )
    _write(output / "permutation/RESULTS.json", reports["permutation"])

    cross = study / "cross_donor/losses"
    original = [
        row
        for row in native
        if row["metadata"]["history"]["end_exclusive"] == 65536
        and row["metadata"]["history"]["recent_prefix_tokens"] == 1024
    ]
    reports["cross_donor"] = _cross_donor(original, _rows(cross))
    _write(output / "cross_donor/SUMMARY.json", reports["cross_donor"])

    matched = _rows(study / "matched_panel/losses")
    reports["matched_panel"] = summarize_matched_panel(
        matched, bootstrap_samples=SAMPLES, seed=SEED
    )
    _write(output / "matched_panel/RESULTS.json", reports["matched_panel"])

    reports.update(reproduce_second_donor(study, output))
    print(json.dumps({"written": "donor2 and separate four-report replication"}), flush=True)

    documents = sorted({row["document_id"] for row in matched})
    reports["state_attenuation"] = summarize_state_attenuation(
        study / "state_attenuation/losses",
        expected=8,
        expected_ids=documents,
        samples=SAMPLES,
        seed=SEED,
    )
    _write(output / "state_attenuation/RESULTS.json", reports["state_attenuation"])

    reports["rwkv_panel"] = reproduce_three_models(study, output / "rwkv_panel")
    print(
        json.dumps({"written": "RWKV original-eight and replication-four comparisons"}), flush=True
    )

    figure_paths = []
    if figures:
        from ..plot_panel import plot_panel
        from ..plot_ladder import plot_ladder

        figure_paths.extend(
            plot_panel(reports["fresh_panel"], output / "figures/mamba_panel", "Context effects")
        )
        figure_paths.extend(plot_ladder(reports["length_ladder"], output / "figures/length_ladder"))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "long_context_diagnostics.plot_matched",
                "--reports",
                str(output / "rwkv_panel"),
                "--output",
                str(output / "figures/matched_three_models"),
            ],
            check=True,
        )
        figure_paths.extend(
            output / f"figures/matched_three_models{ext}" for ext in (".pdf", ".png")
        )
    index = {
        "kind": "saved_loss_reproduction",
        "bootstrap_samples": SAMPLES,
        "seed": SEED,
        "model_evaluation_performed": False,
        "reports": [
            path.relative_to(output).as_posix()
            for path in sorted(output.rglob("*.json"))
            if path.name != "REPRODUCED.json"
        ],
        "figures": [path.relative_to(output).as_posix() for path in figure_paths],
    }
    _write(output / "REPRODUCED.json", index)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root", type=Path, default=Path(__file__).resolve().parents[2] / "results"
    )
    parser.add_argument("--output", type=Path, default=Path("reproduced"))
    parser.add_argument(
        "--figures", action="store_true", help="also regenerate all three manuscript figures"
    )
    args = parser.parse_args(argv)
    reproduce(args.study_root, args.output, figures=args.figures)


if __name__ == "__main__":
    main()
