"""Prepare the second-donor matched-text plans and scoring commands (CPU only)."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .hashing import sha256_file
from .matched_text import load_text_plan, plan_matched_text, save_text_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-plans",
        type=Path,
        required=True,
        help="reconstructed donor1 plans and their PLAN.json",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="reconstructed source JSONL including the second donor",
    )
    parser.add_argument("--mamba-tokenizer", type=Path, required=True)
    parser.add_argument("--gdn-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new donor2 plan directory")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--python", default="python", help="interpreter in each model scoring environment"
    )
    parser.add_argument("--revision", default="unrecorded", help="source provenance only")
    args = parser.parse_args(argv)
    previous = args.previous_plans
    previous_report = json.loads((previous / "PLAN.json").read_text(encoding="utf-8"))
    source = args.documents
    docs = {
        r["document_id"]: r
        for r in map(json.loads, source.read_text(encoding="utf-8").splitlines())
    }
    donor_id = "GAO-25-107421__donor"
    tokenizers = {
        name: Tokenizer.from_file(str(path))
        for name, path in (("mamba", args.mamba_tokenizer), ("gdn", args.gdn_tokenizer))
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result_dir = args.results_dir
    report = dict(
        kind="second_donor_matched_text_replication",
        status="prepared",
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        revision=args.revision,
        donor_id=donor_id,
        documents=8,
        models=["mamba", "gdn"],
        reference=dict(E=32768, C=1024, T=1024),
        analysis="Paired fixed-target within-model effects; donor2 is a sensitivity check, not new independent target documents.",
        aggregation="Same target bytes as donor1; per-donor document bootstrap, paired donor contrast secondary.",
        limits=[
            "One fixed second GAO donor; targets selected in earlier phase.",
            "Foreign matches characters, not native token count.",
            "Unequal public checkpoints and training; no architecture causality.",
        ],
        source_panel_sha256=sha256_file(source),
        cases=[],
        commands=[],
    )
    for case in previous_report["cases"]:
        doc_id = case["document_id"]
        spans = case["requested_source_char_spans"]
        plans, alignment = plan_matched_text(
            docs[doc_id]["text"],
            docs[donor_id]["text"],
            recent_start=spans["recent"][0],
            target_start=spans["target"][0],
            end=spans["target"][1],
            tokenizers=tokenizers,
            prefix_ids={"mamba": 0, "gdn": 1},
        )
        if alignment["status"] != "ready":
            raise ValueError(f"Unsupported alignment: {doc_id}: {alignment}")
        row = dict(
            document_id=doc_id,
            donor_id=donor_id,
            target_utf8_bytes=alignment["target_utf8_bytes"],
            models={},
        )
        for name, plan in plans.items():
            old = load_text_plan(previous / f"{doc_id}_donor1_{name}.npz")
            assert np.array_equal(plan.target_ids, old.target_ids)
            assert plan.metadata["target_text_sha256"] == old.metadata["target_text_sha256"]
            assert all(np.array_equal(plan.inputs[k], old.inputs[k]) for k in ("full", "short"))
            plan.metadata.update(
                document_id=doc_id,
                donor_id=donor_id,
                source=docs[doc_id].get("source"),
                donor_source=docs[donor_id].get("source"),
                model_reference=old.metadata["model_reference"],
                reference_span=old.metadata["reference_span"],
                tokenizer_json_sha256=old.metadata["tokenizer_json_sha256"],
                source_panel_sha256=report["source_panel_sha256"],
            )
            path = out / f"{doc_id}_donor2_{name}.npz"
            result = result_dir / path.name
            save_text_plan(path, plan)
            command = [
                args.python,
                "-m",
                "long_context_diagnostics.score_matched_text",
                "--plan",
                str(path),
                "--output",
                str(result),
                "--repeat",
                "--solve-precision",
                "native",
                "--max-memory-gb",
                "14",
                "--code-revision",
                args.revision,
            ]
            report["commands"].append(command)
            row["models"][name] = dict(
                plan=str(path),
                result=str(result),
                sha256=sha256_file(path),
                target_tokens=len(plan.target_ids),
                documentary_input_tokens=plan.metadata["documentary_input_tokens"],
                full_short_target_identical_to_donor1=True,
            )
        report["cases"].append(row)
    (out / "PLAN.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                plan=str(out / "PLAN.json"),
                cells=len(report["commands"]),
                plan_bytes=sum(p.stat().st_size for p in out.glob("*.npz")),
            )
        )
    )


if __name__ == "__main__":
    main()
