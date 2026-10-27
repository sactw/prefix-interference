"""Prepare the two GAO 2025 reports later used as donors, from cached PDF and tokenizer files.

The PDF prose extraction is a separate variant from gao_long_prose_v1. Text is
never repeated, padded or concatenated across documents.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen

import fitz
from tokenizers import Tokenizer

from .hashing import sha256
from .prepare_gao import word_shingles
from .prepare_gao_long import TOKENIZER_ID, TOKENIZER_REVISION

REPORTS = {
    "GAO-25-107705": {
        "title": "Financial Audit Manual: Volume 1, June 2025",
        "publication_date": "2025-06-12",
        "donor_id": "GAO-25-107421",
    },
    "GAO-25-107421": {
        "title": "Financial Audit: FY 2024 and FY 2023 Consolidated Financial Statements of the U.S. Government",
        "publication_date": "2025-01-16",
        "donor_id": "GAO-25-107705",
    },
}
EXTRACTION = (
    "gao_pdf_prose_blocks_v1: PyMuPDF page-order get_text(blocks,sort=True); "
    "keep text blocks inside y=35 to page_height-40 points, at least150 "
    "characters after whitespace collapse and at least62% alphabetic "
    "characters; omit blocks beginning Page<space><digit>, GAO-<year>-, "
    "or Source:. Preserve retained blocks verbatim apart from whitespace, "
    "joined by two newlines. No repeated/padded/concatenated documents. "
    "Short headings/bullets and many numeric table blocks are omitted; "
    "footnotes or prose table cells meeting the rule can remain."
)


def fetch_pdf(raw_dir: Path, report_id: str) -> tuple[Path, dict]:
    path = raw_dir / f"{report_id}.pdf"
    url = f"https://files.gao.gov/assets/{report_id.lower()}.pdf"
    reused = path.exists()
    if not reused:
        with urlopen(
            Request(url, headers={"User-Agent": "long-context-diagnostics/0.1 research"}),
            timeout=60,
        ) as response:
            raw = response.read(30_000_001)
        if len(raw) > 30_000_000:
            raise ValueError(f"{report_id} exceeds the 30 MB preparation download limit")
        path.write_bytes(raw)
    raw = path.read_bytes()
    meta = dict(
        url=url,
        source_sha256=sha256(raw),
        source_bytes=len(raw),
        local_source_verified_at=datetime.now(timezone.utc).isoformat(),
        reused_cached_source=reused,
    )
    (raw_dir / f"{report_id}.pdf.source.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return path, meta


def extract_pdf(path: Path, *, sort_blocks: bool = True) -> tuple[str, dict, list[dict]]:
    blocks, spans = [], []
    offset = 0
    with fitz.open(path) as doc:
        pages = len(doc)
        for page_number, page in enumerate(doc, 1):
            for block in page.get_text("blocks", sort=sort_blocks):
                if block[6] != 0 or block[1] < 35 or block[3] > page.rect.height - 40:
                    continue
                text = " ".join(block[4].split())
                if len(text) < 150 or sum(c.isalpha() for c in text) / len(text) < 0.62:
                    continue
                if re.match(r"^(Page\s+\d+|GAO-\d{2}-|Source:)", text):
                    continue
                if blocks:
                    offset += 2
                spans.append(dict(page=page_number, char_start=offset, char_end=offset + len(text)))
                blocks.append(text)
                offset += len(text)
    text = "\n\n".join(blocks) + "\n"
    extraction = (
        EXTRACTION
        if sort_blocks
        else EXTRACTION.replace("gao_pdf_prose_blocks_v1", "pdf_prose_native_blocks_v2").replace(
            "get_text(blocks,sort=True)",
            "get_text(blocks,sort=False), preserving native PDF content-stream reading order",
        )
    )
    meta = dict(
        extraction=extraction,
        extractor_version=fitz.VersionBind,
        pages=pages,
        retained_blocks=len(blocks),
        character_count=len(text),
        word_count=len(text.split()),
        text_sha256=sha256(text.encode()),
        replacement_character_count=text.count("\ufffd"),
    )
    return text, meta, spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="existing pinned GPT-NeoX tokenizer JSON; never downloads weights",
    )
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    texts, ids, sources, spans_by_id, encodings = {}, {}, {}, {}, {}
    for report_id, report in REPORTS.items():
        path, source = fetch_pdf(raw_dir, report_id)
        text, extraction, spans = extract_pdf(path)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        source.update(
            extraction,
            title=report["title"],
            publication_date=report["publication_date"],
            product_url=f"https://www.gao.gov/products/{report_id.lower()}",
            tokenizer_id=TOKENIZER_ID,
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_sha256=sha256(args.tokenizer.read_bytes()),
            native_tokens=len(encoding.ids),
            special_tokens_added=False,
        )
        texts[report_id], ids[report_id], sources[report_id] = text, encoding.ids, source
        spans_by_id[report_id], encodings[report_id] = spans, encoding
        print(report_id, len(encoding.ids), "native tokens", flush=True)
    rows = []
    for report_id, report in REPORTS.items():
        rows.append(
            dict(
                document_id=report_id,
                role="target",
                split="evaluation_expansion_20261023",
                donor_id=report["donor_id"] + "__donor",
                text=texts[report_id],
                source=sources[report_id],
            )
        )
    # Reader requires donor-role rows. Aliases refer to the other real report;
    # they do not create extra documents or independent experimental samples.
    for report_id in REPORTS:
        rows.append(
            dict(
                document_id=report_id + "__donor",
                role="donor",
                split="evaluation_expansion_20261023",
                canonical_document_id=report_id,
                text=texts[report_id],
                source=sources[report_id],
            )
        )
    output = args.output_dir / "panel.jsonl"
    output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    checks, inspection = [], []
    for report_id, report in REPORTS.items():
        donor_id = report["donor_id"]
        for end in (32768, 65536):
            needed = end - 2048 - 1024
            target_prefix = tokenizer.decode(ids[report_id][:end])
            donor_prefix = tokenizer.decode(ids[donor_id][:needed])
            sh1, sh2 = word_shingles(target_prefix), word_shingles(donor_prefix)
            checks.append(
                dict(
                    document_id=report_id,
                    donor_id=donor_id,
                    end=end,
                    recent=2048,
                    targets=1024,
                    required_donor_tokens=needed,
                    eligible=len(ids[report_id]) >= end and len(ids[donor_id]) >= needed,
                    shared_prefix_13_word_shingles=len(sh1 & sh2),
                    prefix_shingle_jaccard=len(sh1 & sh2) / len(sh1 | sh2),
                )
            )
            char_start = encodings[report_id].offsets[end - 1024][0]
            char_end = encodings[report_id].offsets[end - 1][1]
            pages = sorted(
                {
                    s["page"]
                    for s in spans_by_id[report_id]
                    if s["char_end"] > char_start and s["char_start"] < char_end
                }
            )
            inspection.append(
                dict(
                    document_id=report_id,
                    end=end,
                    target_pdf_pages=pages,
                    target_text=tokenizer.decode(ids[report_id][end - 1024 : end]),
                    donor_prefix_end=donor_prefix[-1500:],
                )
            )
    manifest = dict(
        prepared_at=datetime.now(timezone.utc).isoformat(),
        selection="2025 official publication date, genuine extracted length and readable prose; no model scores used",
        independent_source_documents=2,
        targets=2,
        donor_policy="reciprocal distinct reports; shared donor/target sources make results dependent",
        source_scope="One audit-methodology manual and one consolidated financial report, not representative of all natural documents",
        extraction_scope="PDF prose extraction variant; not byte-compatible with earlier HTML development panels",
        pretraining_scope="Publication dates are 2025; editions can reuse older published prose, so no claim of pretraining disjointness",
        jsonl_sha256=sha256(output.read_bytes()),
        jsonl_bytes=output.stat().st_size,
        documents=[dict(document_id=k, **v) for k, v in sources.items()],
        eligibility=checks,
    )
    (args.output_dir / "preparation.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "inspection.json").write_text(
        json.dumps(inspection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Wrote", output, flush=True)


if __name__ == "__main__":
    main()
