"""Extract the report PDFs for the two-donor native-token panel (CPU only).

Sources and their order come from the source list, fixed before any scoring.
The prose filters keep native PDF block order; documents are never concatenated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from tokenizers import Tokenizer

from .hashing import sha256
from .prepare_gao import word_shingles
from .prepare_gao_long import TOKENIZER_ID, TOKENIZER_REVISION
from .prepare_gao_panel import extract_pdf

MAX_DONOR_TOKENS = 65536 - 1024 - 1024


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--donor-panel", type=Path, required=True)
    parser.add_argument("--donors", nargs=2, default=["GAO-25-107705", "GAO-25-107421"])
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    documents_dir = args.output_dir / "documents"
    documents_dir.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    targets, report, inspections = [], [], []
    for source in sources:
        document_id = source["document_id"]
        path = raw_dir / f"{document_id}.pdf"
        try:
            if not path.exists():
                with urlopen(
                    Request(
                        source["url"], headers={"User-Agent": "Mozilla/5.0 public-text-research"}
                    ),
                    timeout=60,
                ) as response:
                    raw = response.read(50_000_001)
                if len(raw) > 50_000_000:
                    raise ValueError("source exceeds 50 MB download limit")
                if not raw.startswith(b"%PDF"):
                    raise ValueError("source response is not a PDF")
                path.write_bytes(raw)
            text, extraction, spans = extract_pdf(path, sort_blocks=False)
            encoding = tokenizer.encode(text, add_special_tokens=False)
            meta = dict(
                source,
                **extraction,
                native_tokens=len(encoding.ids),
                source_sha256=sha256(path.read_bytes()),
                source_bytes=path.stat().st_size,
                tokenizer_id=TOKENIZER_ID,
                tokenizer_revision=TOKENIZER_REVISION,
                tokenizer_sha256=sha256(args.tokenizer.read_bytes()),
                special_tokens_added=False,
            )
            eligible = len(encoding.ids) >= 65536
            report.append(dict(**meta, eligible=eligible))
            row = dict(
                document_id=document_id,
                role="target",
                split="fresh_panel_20261023",
                text=text,
                source=meta,
            )
            write_json(documents_dir / f"{document_id}.json", row)
            print(
                document_id, len(encoding.ids), "eligible" if eligible else "too short", flush=True
            )
            if eligible:
                targets.append(row)
                for end in (32768, 65536):
                    char_start = encoding.offsets[end - 1024][0]
                    char_end = encoding.offsets[end - 1][1]
                    pages = sorted(
                        {
                            s["page"]
                            for s in spans
                            if s["char_end"] > char_start and s["char_start"] < char_end
                        }
                    )
                    inspections.append(
                        dict(
                            document_id=document_id,
                            end=end,
                            target_pdf_pages=pages,
                            target_text=tokenizer.decode(encoding.ids[end - 1024 : end]),
                        )
                    )
        except Exception as exc:
            report.append(dict(**source, eligible=False, preparation_error=str(exc)))
            print(document_id, "ERROR", str(exc), flush=True)
        write_json(args.output_dir / "source_report.json", report)
        write_json(args.output_dir / "inspection.json", inspections)
    donor_rows = {
        r["document_id"]: r
        for r in map(json.loads, args.donor_panel.read_text(encoding="utf-8").splitlines())
    }
    overlap = []
    for donor_no, donor_id in enumerate(args.donors, 1):
        donor = donor_rows[donor_id]
        # Apply the same native-order extraction to this panel's donor copies.
        donor_text, donor_extraction, _ = extract_pdf(
            args.donor_panel.parent / "raw" / f"{donor_id}.pdf", sort_blocks=False
        )
        donor = dict(donor, text=donor_text, source=dict(donor["source"], **donor_extraction))
        donor_ids = tokenizer.encode(donor["text"], add_special_tokens=False).ids
        donor["source"]["native_tokens"] = len(donor_ids)
        donor_row = dict(
            donor, document_id=donor_id + "__donor", canonical_document_id=donor_id, role="donor"
        )
        rows = [dict(r, donor_id=donor_row["document_id"]) for r in targets] + [donor_row]
        output = args.output_dir / f"panel_donor{donor_no}.jsonl"
        output.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        for row in targets:
            a = word_shingles(row["text"])
            b = word_shingles(tokenizer.decode(donor_ids[:MAX_DONOR_TOKENS]))
            overlap.append(
                dict(
                    target=row["document_id"],
                    donor=donor_id,
                    donor_tokens=len(donor_ids),
                    required_donor_tokens=MAX_DONOR_TOKENS,
                    eligible=len(donor_ids) >= MAX_DONOR_TOKENS,
                    shared_13_word_shingles=len(a & b),
                    jaccard=len(a & b) / len(a | b),
                )
            )
    write_json(
        args.output_dir / "preparation.json",
        dict(
            prepared_at=datetime.now(timezone.utc).isoformat(),
            selection="Explicit source list, public contemporary PDF reports, readable extracted prose >=65536 native tokens; no model scores used",
            target_document_ids=[r["document_id"] for r in targets],
            donors=args.donors,
            donor_policy="Two fixed development GAO reports; excluded from fresh targets. Both shared across targets, so donors are not independent population samples.",
            scope="Diverse report topics, not representative of all prose. Annual reports/editions can reuse earlier text; publication date does not prove pretraining disjointness.",
            source_list_sha256=sha256(args.sources.read_bytes()),
            maximum_required_donor_tokens=MAX_DONOR_TOKENS,
            overlap=overlap,
            panels=[
                dict(
                    name=f"panel_donor{i}.jsonl",
                    sha256=sha256((args.output_dir / f"panel_donor{i}.jsonl").read_bytes()),
                )
                for i in (1, 2)
            ],
        ),
    )


if __name__ == "__main__":
    main()
