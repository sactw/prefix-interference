"""Write the state-attenuation scoring commands, one per report and history.

The original run reused the already measured IEA pair; pass --include-iea to
prepare all sixteen histories for a fresh run with reconstructed plans.
"""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans", type=Path, required=True, help="reconstructed matched donor1 plans and PLAN.json"
    )
    parser.add_argument(
        "--references", type=Path, required=True, help="completed matched donor1 Mamba result NPZs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python", default="python", help="interpreter in the Mamba scoring environment"
    )
    parser.add_argument("--revision", default="unrecorded", help="source provenance only")
    parser.add_argument("--include-iea", action="store_true", help="also prepare the IEA pair")
    args = parser.parse_args(argv)
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.plans / "PLAN.json").read_text(encoding="utf-8"))
    commands = []
    for case in plan["cases"]:
        doc = case["document_id"]
        if doc == "IEA-WEO-2025" and not args.include_iea:
            continue  # The IEA pair was measured first and is reused.
        for history in ("full", "foreign"):
            cmd = [
                args.python,
                "-m",
                "long_context_diagnostics.score_state_attenuation",
                "--plan",
                str(args.plans / f"{doc}_donor1_mamba.npz"),
                "--reference",
                str(args.references / f"{doc}_donor1_mamba.npz"),
                "--output",
                str(folder / f"{doc}_{history}.npz"),
                "--history",
                history,
                "--alphas",
                "1",
                "0.5",
                "0",
                "--chunk-size",
                "256",
                "--max-memory-gb",
                "14",
                "--code-revision",
                args.revision,
            ]
            commands.append(cmd)
    report = {
        "kind": "old_ssm_state_attenuation_panel",
        "selection": (
            "All eight original matched-text donor1 documents."
            if args.include_iea
            else "All eight original matched-text donor1 documents; completed IEA pilot reused."
        ),
        "reason": "Paired IEA pilot preserved alpha1 numerics and showed a large effect; test replication.",
        "expected_total_documents": 8,
        "new_cells": len(commands),
        "alphas": [1, 0.5, 0],
        "measured_pilot_job_seconds": [66, 67],
        "estimated_additional_active_minutes": len(commands) * 67 / 60,
        "analysis": "Paired document bootstrap; separate alpha effects, original-versus-foreign and short comparison.",
        "limits": [
            "Selected after one-document pilot; exploratory.",
            "Scale old SSM caches only; retain convolution caches and offsets.",
            "Alpha0 is not the fresh short baseline.",
            "No claim of unique finite-capacity cause or restored useful distant content.",
        ],
        "commands": commands,
    }
    (folder / "PLAN.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"new_cells": len(commands), "plan": str(folder / "PLAN.json")}))


if __name__ == "__main__":
    main()
