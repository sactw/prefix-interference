"""Prepare matched-text replication plans with cached CPU tokenizers.

Select documents and a donor before inspecting their scores. The Mamba token
offsets define the requested text spans; matched_text then finds the common
native target boundary. The script writes the plans and their scoring commands;
it loads no model weights and leaves earlier results unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

# Configure worker counts before NumPy and the Rust tokenizer runtime load.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from . import gdn_adapter, mamba_adapter  # noqa: E402
from .hashing import sha256_file  # noqa: E402
from .matched_text import plan_matched_text, save_text_plan  # noqa: E402
from .score_matched_text import scoring_command  # noqa: E402


MODEL_REFERENCES = {
    name: {
        "model_id": module.DEFAULT_MODEL_ID,
        "revision": module.DEFAULT_REVISION,
        "tokenizer_id": module.TOKENIZER_ID,
        "tokenizer_revision": module.DEFAULT_TOKENIZER_REVISION,
    }
    for name, module in (("mamba", mamba_adapter), ("gdn", gdn_adapter))
}
PREFIX_IDS = {"mamba": 0, "gdn": 1}


def reference_char_spans(text, tokenizer, *, end=32768, recent=1024, targets=1024):
    """Use native Mamba offsets in unchanged source text, without special IDs."""
    if not (recent > 0 and targets > 0 and end > recent + targets):
        raise ValueError("require positive C/T and E > C+T")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) < end:
        raise ValueError(f"reference document has {len(encoding.ids)} tokens; needs {end}")
    recent_start = encoding.offsets[end - targets - recent][0]
    target_start = encoding.offsets[end - targets][0]
    char_end = encoding.offsets[end - 1][1]
    return {
        "far": [0, recent_start],
        "recent": [recent_start, target_start],
        "target": [target_start, char_end],
    }


def prepare_case(
    document,
    donor,
    tokenizers,
    tokenizer_shas,
    *,
    source_panel_sha256,
    end=32768,
    recent=1024,
    targets=1024,
    code_revision="unrecorded",
):
    """Prepare one selected document; unsupported alignments remain explicit."""
    doc_id, donor_id = document["document_id"], donor["document_id"]
    if doc_id == donor_id:
        raise ValueError("target and foreign donor must be distinct documents")
    spans = reference_char_spans(
        document["text"], tokenizers["mamba"], end=end, recent=recent, targets=targets
    )
    plans, report = plan_matched_text(
        document["text"],
        donor["text"],
        recent_start=spans["recent"][0],
        target_start=spans["target"][0],
        end=spans["target"][1],
        tokenizers=tokenizers,
        prefix_ids=PREFIX_IDS,
    )
    reference = {"tokenizer": "mamba", "E": end, "C": recent, "T": targets}
    report.update(document_id=doc_id, donor_id=donor_id, reference_span=reference)
    for name, plan in plans.items():
        plan.metadata.update(
            document_id=doc_id,
            donor_id=donor_id,
            source=document.get("source"),
            donor_source=donor.get("source"),
            model_reference=MODEL_REFERENCES[name],
            reference_span=reference,
            tokenizer_json_sha256=tokenizer_shas[name],
            source_panel_sha256=source_panel_sha256,
            preparation_revision=code_revision,
        )
    return plans, report


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-") or "document"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="JSONL rows with document_id, text and source",
    )
    parser.add_argument("--document-id", action="append", required=True)
    parser.add_argument("--donor-id", required=True)
    parser.add_argument("--mamba-tokenizer", type=Path, required=True)
    parser.add_argument("--gdn-tokenizer", type=Path, required=True)
    parser.add_argument("--end", type=int, default=32768)
    parser.add_argument("--recent", type=int, default=1024)
    parser.add_argument("--targets", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="local plan directory; use a new directory for a new panel",
    )
    parser.add_argument(
        "--panel-prefix", required=True, help="panel name used in the default runtime directories"
    )
    parser.add_argument(
        "--runtime-root", default=".", help="root for caches and directories in the commands"
    )
    parser.add_argument("--mamba-python", default=sys.executable)
    parser.add_argument("--gdn-python", default=sys.executable)
    parser.add_argument(
        "--runtime-plans-dir",
        help="plan directory used in the commands; default <runtime-root>/data/<panel-prefix>/plans",
    )
    parser.add_argument(
        "--runtime-results-dir",
        help="result directory used in the commands; default <runtime-root>/results/<panel-prefix>/losses",
    )
    parser.add_argument("--code-revision", default="unrecorded", help="provenance only")
    args = parser.parse_args(argv)
    from tokenizers import Tokenizer

    began = time.perf_counter()
    rows = [
        json.loads(line)
        for line in args.documents.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    documents = {row["document_id"]: row for row in rows}
    selected = list(dict.fromkeys(args.document_id))
    missing = [key for key in [*selected, args.donor_id] if key not in documents]
    if missing:
        parser.error(f"document IDs absent from JSONL: {missing}")
    token_paths = {"mamba": args.mamba_tokenizer, "gdn": args.gdn_tokenizer}
    tokenizers = {name: Tokenizer.from_file(str(path)) for name, path in token_paths.items()}
    token_shas = {name: sha256_file(path) for name, path in token_paths.items()}
    panel_sha = sha256_file(args.documents)
    root = PurePosixPath(args.runtime_root)
    prefix = safe_name(args.panel_prefix)
    runtime_plans = (
        PurePosixPath(args.runtime_plans_dir)
        if args.runtime_plans_dir
        else root / "data" / prefix / "plans"
    )
    runtime_results = (
        PurePosixPath(args.runtime_results_dir)
        if args.runtime_results_dir
        else root / "results" / prefix / "losses"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "matched_text_replication_preparation",
        "status": "prepared",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": args.code_revision,
        "source_panel_sha256": panel_sha,
        "selected_document_ids": selected,
        "donor_id": args.donor_id,
        "reference_span": {
            "tokenizer": "mamba",
            "E": args.end,
            "C": args.recent,
            "T": args.targets,
        },
        "runtime_plans_dir": str(runtime_plans),
        "runtime_results_dir": str(runtime_results),
        "tokenizers": {
            name: {
                "file": str(path),
                "sha256": token_shas[name],
                "reference": MODEL_REFERENCES[name],
            }
            for name, path in token_paths.items()
        },
        "cases": [],
        "commands": [],
    }
    for doc_id in selected:
        try:
            plans, case = prepare_case(
                documents[doc_id],
                documents[args.donor_id],
                tokenizers,
                token_shas,
                source_panel_sha256=panel_sha,
                end=args.end,
                recent=args.recent,
                targets=args.targets,
                code_revision=args.code_revision,
            )
        except ValueError as error:
            plans, case = (
                {},
                {
                    "document_id": doc_id,
                    "donor_id": args.donor_id,
                    "status": "unsupported_input",
                    "reason": str(error),
                },
            )
        case["models"] = {}
        for name, plan in plans.items():
            filename = f"{safe_name(doc_id)}_{safe_name(args.donor_id)}_{name}.npz"
            path = args.output / filename
            save_text_plan(path, plan)
            command = scoring_command(
                root,
                name=name,
                plan_path=runtime_plans / filename,
                result_path=runtime_results / filename,
                revision=args.code_revision,
                python_executable=getattr(args, name + "_python"),
            )
            case["models"][name] = {
                "plan": str(path),
                "runtime_plan": str(runtime_plans / filename),
                "result": str(runtime_results / filename),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "target_tokens": len(plan.target_ids),
                "documentary_input_tokens": plan.metadata["documentary_input_tokens"],
            }
            report["commands"].append(command)
        report["cases"].append(case)
        (args.output / "PLAN.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "document_id": doc_id,
                    "status": case["status"],
                    "plans": len(plans),
                    "target_utf8_bytes": case.get("target_utf8_bytes"),
                }
            ),
            flush=True,
        )
    report.update(
        elapsed_seconds=time.perf_counter() - began,
        prepared_plans=len(report["commands"]),
    )
    if any(case["status"] != "ready" for case in report["cases"]):
        report["status"] = "partially_prepared"
    (args.output / "PLAN.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output / "PLAN.json"),
                "status": report["status"],
                "prepared_plans": report["prepared_plans"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
