"""Three-model readout (Mamba-2, Gated DeltaNet, RWKV-7) on matched target text.

The original eight reports and the four later reports are separate cohorts.
Documents missing a model are listed under coverage and left out.
"""

import argparse
import json
from pathlib import Path

from .measurement import load_pair
from .matched_panel_readout import summarize_matched_panel


def _records(folder):
    result = {}
    for path in sorted(folder.glob("*.npz")):
        record = load_pair(path)
        key = (record["document_id"], record["metadata"]["history"]["model_name"])
        if key in result:
            raise ValueError(f"duplicate document/model result: {key}")
        result[key] = (path, record)
    return result


def reproduce(study, output, *, smoke_2k=None, smoke_32k=None):
    """Pair saved result records by document."""
    study, output = Path(study), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = {}
    for label, supplied, filename in [
        ("2k", smoke_2k, "2k_fix1.json"),
        ("32k", smoke_32k, "32k_fix2.json"),
    ]:
        path = Path(supplied) if supplied is not None else study / "rwkv_panel/smokes" / filename
        smoke = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        display = path.relative_to(study).as_posix() if path.is_relative_to(study) else str(path)
        validation[label] = {
            "path": display,
            "status": smoke.get("status", "missing"),
            "passed": smoke.get("passed", False),
        }
    rwkv = _records(study / "rwkv_panel/losses")
    reports = {}
    for group, phase, expected in [
        ("original8", "matched_panel", 8),
        ("replication4", "replication", 4),
    ]:
        earlier = _records(study / phase / "losses")
        documents = sorted({doc for doc, _ in earlier})
        records, paths, pending = [], [], []
        for doc in documents:
            cells = [earlier.get((doc, name)) for name in ("mamba", "gdn")]
            cells.append(rwkv.get((doc, "rwkv")))
            missing = [name for name, cell in zip(("mamba", "gdn", "rwkv"), cells) if cell is None]
            if missing:
                pending.append({"document_id": doc, "missing_models": missing})
            else:
                records.extend(record for path, record in cells)
                paths.extend(path.relative_to(study).as_posix() for path, record in cells)
        report = (
            summarize_matched_panel(
                records, models=("mamba", "gdn", "rwkv"), bootstrap_samples=2000, seed=20261020
            )
            if records
            else {}
        )
        report.update(
            coverage={
                "expected_documents": expected,
                "completed_triplets": len(records) // 3,
                "pending": pending,
                "complete": len(records) // 3 == expected and not pending,
            },
            files=paths,
            runtime_validation=validation,
            interpretation_status=(
                "numerical_smokes_passed"
                if all(x["status"] == "completed" and x["passed"] for x in validation.values())
                else "provisional_pending_numerical_review"
            ),
        )
        (output / (group + "_RESULTS.json")).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        reports[group] = report
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-2k", type=Path, help="optional path to the recorded 2K smoke JSON")
    parser.add_argument(
        "--smoke-32k", type=Path, help="optional path to the recorded 32K smoke JSON"
    )
    args = parser.parse_args(argv)
    reports = reproduce(
        args.study_root, args.output, smoke_2k=args.smoke_2k, smoke_32k=args.smoke_32k
    )
    summary = {
        group: {
            "coverage": report["coverage"],
            "models": {
                name: values["primary_byte_weighted"]
                for name, values in report.get("models", {}).items()
            },
        }
        for group, report in reports.items()
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
