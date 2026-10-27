"""Prepare the eight-document matched-text panel using cached CPU tokenizers.

The 32-case alignment report fixes the requested text spans. Both tokenizers
are reconstructed together.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from tokenizers import Tokenizer

from .hashing import sha256_file
from .matched_text import load_text_plan, plan_matched_text, save_text_plan
from .score_matched_text import scoring_command


def same_arrays(left, right):
    return (
        left.target_starts == right.target_starts
        and np.array_equal(left.target_ids, right.target_ids)
        and np.array_equal(left.target_positions, right.target_positions)
        and all(
            np.array_equal(left.inputs[k], right.inputs[k]) for k in ("full", "short", "foreign")
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--mamba-tokenizer", type=Path, required=True)
    parser.add_argument("--gdn-tokenizer", type=Path, required=True)
    parser.add_argument("--existing-plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", default="unrecorded")
    parser.add_argument("--mamba-python", default=sys.executable)
    parser.add_argument("--gdn-python", default=sys.executable)
    args = parser.parse_args(argv)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    begin = time.perf_counter()
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    selected = [
        row
        for row in alignment["cells"]
        if row["reference_end"] == 32768 and row["donor_id"] == "GAO-25-107705__donor"
    ]
    assert len(selected) == len({row["document_id"] for row in selected}) == 8
    documents = {
        row["document_id"]: row
        for row in (
            json.loads(line)
            for line in args.documents.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    token_paths = {"mamba": args.mamba_tokenizer, "gdn": args.gdn_tokenizer}
    tokenizers = {name: Tokenizer.from_file(str(path)) for name, path in token_paths.items()}
    iea_plans = {
        name: load_text_plan(args.existing_plans / f"IEA-WEO-2025_donor1_{name}.npz")
        for name in tokenizers
    }
    panel_sha = sha256_file(args.documents)
    for name in tokenizers:
        assert sha256_file(token_paths[name]) == iea_plans[name].metadata["tokenizer_json_sha256"]
        assert panel_sha == iea_plans[name].metadata["source_panel_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "matched_text_eight_document_preparation",
        "status": "prepared",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": args.code_revision,
        "preparation_script_sha256": sha256_file(__file__),
        "source_panel_sha256": panel_sha,
        "alignment_report_sha256": sha256_file(args.alignment),
        "selection": {
            "documents": 8,
            "donor": "GAO-25-107705__donor",
            "reference_span": {"tokenizer": "mamba", "E": 32768, "C": 1024, "T": 1024},
        },
        "selection_timing": "Eight eligible alignment-report cases fixed before the 14 additional scores; the IEA pilot is already known.",
        "analysis": "Compare paired effects on identical target bytes; sum target NLL over bytes with document-cluster bootstrap. Keep separate from native-token panel.",
        "limits": [
            "One fixed donor shared across eight targets; not independent donor samples.",
            "Foreign prefix matches characters, not native tokens; report actual lengths.",
            "Byte NLL scores canonical tokenizations, not marginal text probabilities.",
            "Unequal model size and training; not a controlled architecture ranking.",
        ],
        "tokenizers": {
            name: {
                "file": str(path),
                "sha256": sha256_file(path),
                "reference": iea_plans[name].metadata["model_reference"],
            }
            for name, path in token_paths.items()
        },
        "cases": [],
        "commands": [],
    }
    for row in selected:
        doc_id, donor_id = row["document_id"], row["donor_id"]
        doc, donor = documents[doc_id], documents[donor_id]
        spans = row["report"]["requested_source_char_spans"]
        plans, rebuilt = plan_matched_text(
            doc["text"],
            donor["text"],
            recent_start=spans["recent"][0],
            target_start=spans["target"][0],
            end=spans["target"][1],
            tokenizers=tokenizers,
            prefix_ids={"mamba": 0, "gdn": 1},
        )
        assert rebuilt == row["report"], f"alignment changed for {doc_id}"
        case = {
            "document_id": doc_id,
            "donor_id": donor_id,
            "target_utf8_bytes": rebuilt["target_utf8_bytes"],
            "target_text_sha256": rebuilt["target_text_sha256"],
            "requested_source_char_spans": spans,
            "source_char_spans": rebuilt["source_char_spans"],
            "target_start_adjustment_characters": rebuilt["target_start_adjustment_characters"],
            "models": {},
        }
        for name, plan in plans.items():
            assert plan.metadata == row["plan_metadata"][name], (
                f"plan metadata changed: {doc_id}/{name}"
            )
            iea_plan = iea_plans[name]
            plan.metadata.update(
                document_id=doc_id,
                donor_id=donor_id,
                source=doc.get("source"),
                donor_source=donor.get("source"),
                model_reference=iea_plan.metadata["model_reference"],
                reference_span={"tokenizer": "mamba", "E": 32768, "C": 1024, "T": 1024},
                tokenizer_json_sha256=sha256_file(token_paths[name]),
                source_panel_sha256=panel_sha,
            )
            path = args.output / f"{doc_id}_donor1_{name}.npz"
            if doc_id == "IEA-WEO-2025":
                assert same_arrays(plan, iea_plan) and plan.metadata == iea_plan.metadata
                shutil.copyfile(args.existing_plans / path.name, path)
                action = "reuse_completed_result"
                result = args.root / "results/matched_panel/losses" / path.name
            else:
                save_text_plan(path, plan)
                action = "prepared"
                result = args.root / "results/matched_panel/losses" / path.name
                report["commands"].append(
                    scoring_command(
                        args.root,
                        name=name,
                        plan_path=path,
                        result_path=result,
                        revision=args.code_revision,
                        python_executable=getattr(args, name + "_python"),
                    )
                )
            saved = load_text_plan(path)
            assert same_arrays(plan, saved) and plan.metadata == saved.metadata
            case["models"][name] = {
                "plan": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "action": action,
                "result": str(result),
                "target_tokens": saved.metadata["target_tokens"],
                "documentary_input_tokens": saved.metadata["documentary_input_tokens"],
                "alignment_metadata_equal": True,
                "serialized_plan_equal": True,
            }
        report["cases"].append(case)
        print(
            json.dumps(
                {
                    "document": doc_id,
                    "target_bytes": case["target_utf8_bytes"],
                    "status": "ready",
                    "reused": doc_id == "IEA-WEO-2025",
                }
            ),
            flush=True,
        )
    report.update(
        elapsed_seconds=time.perf_counter() - begin,
        prepared_plans=16,
        reused_results=2,
        additional_model_cells=14,
    )
    (args.output / "PLAN.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output / "PLAN.json"),
                "prepared_plans": 16,
                "additional_model_cells": 14,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
