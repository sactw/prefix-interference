"""Derive RWKV plans from the existing Mamba matched-text plans (CPU only).

The official Pile tokenizer must reproduce every documentary input and target ID
exactly, so source boundaries stay unchanged and the Mamba/GDN results keep their
byte denominators.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

# Configure worker counts before NumPy and the Rust tokenizer runtime load.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402

from .hashing import sha256_file, sha256_text  # noqa: E402
from .matched_text import MatchedTextPlan, load_text_plan, save_text_plan  # noqa: E402


REFERENCE = {
    "model_id": "BlinkDL/rwkv-7-pile",
    "revision": "dac041b037030e33db4ccaccffb67b3fd6d63ff4",
    "tokenizer_id": "BlinkDL/ChatRWKV/20B_tokenizer.json",
    "tokenizer_revision": "2e2bb1cd390cefbedae0a89f4a343f6f754d6621",
}
MODEL_FILENAME = "RWKV-x070-Pile-421M-20241127-ctx4096.pth"
TOKENIZER_SOURCE = (
    "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/"
    + REFERENCE["tokenizer_revision"]
    + "/20B_tokenizer.json"
)


def derive_plan(plan, source_tokenizer, rwkv_tokenizer, *, rwkv_sha, source_plan, source_sha):
    """Check that the RWKV tokenizer reproduces the plan, then copy its IDs and metadata."""
    meta = plan.metadata
    if meta["model_name"] != "mamba" or meta["prefix_id"] != 0:
        raise ValueError("expected an existing Mamba matched-text plan with boundary ID 0")
    if source_tokenizer.id_to_token(0) != rwkv_tokenizer.id_to_token(0):
        raise ValueError("boundary token 0 has different native meaning")
    target = source_tokenizer.decode(plan.target_ids.tolist(), skip_special_tokens=False)
    if (
        sha256_text(target) != meta["target_text_sha256"]
        or len(target) != meta["target_characters"]
        or len(target.encode("utf-8")) != meta["target_utf8_bytes"]
    ):
        raise ValueError("saved target IDs do not reconstruct their source text")
    checks = {}
    for condition, all_ids in plan.inputs.items():
        ids = all_ids[1:].tolist()
        if int(all_ids[0]) != 0:
            raise ValueError(f"{condition}: changed boundary token")
        text = source_tokenizer.decode(ids, skip_special_tokens=False)
        old_ids = source_tokenizer.encode(text, add_special_tokens=False).ids
        new_ids = rwkv_tokenizer.encode(text, add_special_tokens=False).ids
        if old_ids != ids or new_ids != ids:
            raise ValueError(
                f"{condition}: official RWKV encoding differs; source span was not changed"
            )
        if rwkv_tokenizer.decode(new_ids, skip_special_tokens=False) != text:
            raise ValueError(
                f"{condition}: RWKV decoding differs from the plan's documentary input"
            )
        start = plan.target_starts[condition] - 1
        before = rwkv_tokenizer.decode(new_ids[:start], skip_special_tokens=False)
        suffix = rwkv_tokenizer.decode(new_ids[start:], skip_special_tokens=False)
        if suffix != target or before + suffix != text:
            raise ValueError(f"{condition}: target boundary fails exact source decoding")
        if (
            len(text) != meta["input_characters"][condition]
            or len(text.encode("utf-8")) != meta["input_utf8_bytes"][condition]
        ):
            raise ValueError(f"{condition}: documentary text lengths differ from the plan metadata")
        checks[condition] = {
            "documentary_tokens": len(ids),
            "input_ids_identical": True,
            "exact_text_roundtrip": True,
            "target_boundary_unchanged": True,
        }
    converted = copy.deepcopy(meta)
    converted.update(
        model_name="rwkv",
        model_reference=dict(REFERENCE),
        model_filename=MODEL_FILENAME,
        tokenizer_json_sha256=rwkv_sha,
    )
    converted["rwkv_plan_derivation"] = {
        "method": "official Pile tokenizer exact decode/re-encode of each frozen Mamba input",
        "source_plan": str(source_plan),
        "source_plan_sha256": source_sha,
        "source_model_reference": meta["model_reference"],
        "source_tokenizer_json_sha256": meta["tokenizer_json_sha256"],
        "tokenizer_source": TOKENIZER_SOURCE,
        "source_char_spans_unchanged": True,
        "target_ids_and_positions_unchanged": True,
        "checks": checks,
    }
    return MatchedTextPlan(
        {key: value.copy() for key, value in plan.inputs.items()},
        dict(plan.target_starts),
        plan.target_ids.copy(),
        plan.target_positions.copy(),
        converted,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans",
        type=Path,
        nargs="+",
        required=True,
        help="existing Mamba NPZ files or directories containing *_mamba.npz",
    )
    parser.add_argument("--mamba-tokenizer", type=Path, required=True)
    parser.add_argument("--rwkv-tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", default="unrecorded", help="provenance only")
    args = parser.parse_args(argv)
    from tokenizers import Tokenizer

    begin = time.perf_counter()
    source_json = json.loads(args.mamba_tokenizer.read_text(encoding="utf-8"))
    rwkv_json = json.loads(args.rwkv_tokenizer.read_text(encoding="utf-8"))
    # Vocabulary/merges and all native BPE settings must agree. Serialized
    # ByteLevel options can differ; actual input encodings are checked below.
    if source_json["model"] != rwkv_json["model"]:
        parser.error("native BPE vocabulary/merges/settings differ; no plans were rewritten")
    paths = sorted(
        {
            path.resolve()
            for entry in args.plans
            for path in (entry.glob("*_mamba.npz") if entry.is_dir() else [entry])
        }
    )
    if not paths:
        parser.error("no Mamba plans found")
    source_tok = Tokenizer.from_file(str(args.mamba_tokenizer))
    rwkv_tok = Tokenizer.from_file(str(args.rwkv_tokenizer))
    source_sha, rwkv_sha = sha256_file(args.mamba_tokenizer), sha256_file(args.rwkv_tokenizer)
    report = {
        "kind": "rwkv_frozen_matched_text_preparation",
        "status": "preparing",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": args.code_revision,
        "model_reference": REFERENCE,
        "model_filename": MODEL_FILENAME,
        "tokenizer_source": TOKENIZER_SOURCE,
        "tokenizer_json_sha256": rwkv_sha,
        "source_tokenizer_json_sha256": source_sha,
        "native_bpe_settings_equal": True,
        "selection": "all supplied frozen source plans; no model scores read",
        "scope": "new RWKV cells only; existing Mamba/GDN plans and results unchanged",
        "analysis": "reuse identical target bytes with paired document resampling; model training differs",
        "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "PLAN.json"
    seen = set()
    try:
        for path in paths:
            plan = load_text_plan(path)
            key = (plan.metadata["document_id"], plan.metadata["donor_id"])
            if key in seen:
                raise ValueError(f"duplicate source document/donor: {key}")
            seen.add(key)
            if source_sha != plan.metadata["tokenizer_json_sha256"]:
                raise ValueError(f"{path}: supplied source tokenizer does not match the plan input")
            converted = derive_plan(
                plan,
                source_tok,
                rwkv_tok,
                rwkv_sha=rwkv_sha,
                source_plan=path,
                source_sha=sha256_file(path),
            )
            output = args.output / (path.stem.removesuffix("_mamba") + "_rwkv.npz")
            if output.resolve() == path.resolve():
                raise ValueError("output must not replace the source plan")
            save_text_plan(output, converted)
            saved = load_text_plan(output)
            if (
                saved.metadata != converted.metadata
                or not np.array_equal(saved.target_positions, plan.target_positions)
                or not np.array_equal(saved.target_ids, plan.target_ids)
                or any(not np.array_equal(saved.inputs[c], plan.inputs[c]) for c in plan.inputs)
            ):
                raise ValueError(f"{output}: serialized output differs from the verified plan")
            report["cases"].append(
                {
                    "document_id": key[0],
                    "donor_id": key[1],
                    "source_plan": str(path),
                    "plan": str(output),
                    "sha256": sha256_file(output),
                    "target_utf8_bytes": plan.metadata["target_utf8_bytes"],
                    "target_tokens": len(plan.target_ids),
                    "source_char_spans": plan.metadata["source_char_spans"],
                    "checks": converted.metadata["rwkv_plan_derivation"]["checks"],
                }
            )
            print(
                json.dumps({"document_id": key[0], "status": "prepared", "plan": str(output)}),
                flush=True,
            )
        report.update(status="prepared", prepared_plans=len(report["cases"]))
    except Exception as error:
        report.update(status="incomplete", error=str(error), prepared_plans=len(report["cases"]))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - begin
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
