"""Extract the replication reports with the same prose extractor (CPU only).

Reports are chosen by source order, length and legibility, not model scores.
The GAO donor text is copied unchanged from the original panel.
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
from .prepare_gao_panel import extract_pdf
from .prepare_gao_long import TOKENIZER_ID, TOKENIZER_REVISION


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--donor-panel", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    original_rows = [
        json.loads(line) for line in args.donor_panel.read_text(encoding="utf-8").splitlines()
    ]
    donor = next(row for row in original_rows if row["document_id"] == "GAO-25-107705__donor")
    excluded = {row["document_id"] for row in original_rows if row.get("role") == "target"}
    donor_ids = tokenizer.encode(donor["text"], add_special_tokens=False).ids
    donor_prefix = tokenizer.decode(donor_ids[:30720])
    donor_shingles = word_shingles(donor_prefix)
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    rows, reports, inspections = [], [], []
    for source in sources:
        key = source["document_id"]
        if key in excluded:
            raise ValueError(f"previous target cannot be a replication target: {key}")
        pdf = raw_dir / f"{key}.pdf"
        try:
            if not pdf.exists():
                with urlopen(
                    Request(
                        source["url"], headers={"User-Agent": "Mozilla/5.0 public-text-research"}
                    ),
                    timeout=90,
                ) as response:
                    raw = response.read(50_000_001)
                if len(raw) > 50_000_000 or not raw.startswith(b"%PDF"):
                    raise ValueError("not a PDF below the 50 MB download limit")
                pdf.write_bytes(raw)
            text, extraction, spans = extract_pdf(pdf, sort_blocks=False)
            encoding = tokenizer.encode(text, add_special_tokens=False)
            meta = dict(
                source,
                **extraction,
                native_tokens=len(encoding.ids),
                source_sha256=sha256(pdf.read_bytes()),
                source_bytes=pdf.stat().st_size,
                tokenizer_id=TOKENIZER_ID,
                tokenizer_revision=TOKENIZER_REVISION,
                tokenizer_sha256=sha256(args.tokenizer.read_bytes()),
                special_tokens_added=False,
            )
            length_eligible = len(encoding.ids) >= 32768
            eligible = length_eligible and source.get("include_in_panel", True)
            reports.append(dict(meta, length_eligible=length_eligible, eligible=eligible))
            if eligible:
                rows.append(
                    dict(
                        document_id=key,
                        role="target",
                        split="replication_20261024",
                        donor_id=donor["document_id"],
                        text=text,
                        source=meta,
                    )
                )
                for start, end in ((0, 256), (15360, 15616), (30720, 31744), (31744, 32768)):
                    a, b = encoding.offsets[start][0], encoding.offsets[end - 1][1]
                    pages = sorted(
                        {s["page"] for s in spans if s["char_end"] > a and s["char_start"] < b}
                    )
                    inspections.append(
                        dict(
                            document_id=key,
                            token_start=start,
                            token_end=end,
                            pdf_pages=pages,
                            text=text[a:b],
                        )
                    )
                report_shingles = word_shingles(tokenizer.decode(encoding.ids[:32768]))
                reports[-1]["shared_13_word_shingles_with_donor_prefix"] = len(
                    report_shingles & donor_shingles
                )
                if len(rows) == 2:
                    (args.output_dir / "panel_donor1_first2.jsonl").write_text(
                        "".join(
                            json.dumps(row, ensure_ascii=False) + "\n" for row in rows + [donor]
                        ),
                        encoding="utf-8",
                    )
            print(key, len(encoding.ids), "eligible" if eligible else "not selected", flush=True)
        except Exception as exc:
            reports.append(dict(source, eligible=False, preparation_error=str(exc)))
            print(key, "ERROR", str(exc), flush=True)
        write_json(args.output_dir / "source_report.json", reports)
        write_json(args.output_dir / "inspection.json", inspections)
    panel = args.output_dir / "panel_donor1.jsonl"
    panel.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows + [donor]),
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "preparation.json",
        dict(
            prepared_at=datetime.now(timezone.utc).isoformat(),
            source_list_sha256=sha256(args.sources.read_bytes()),
            selection="Explicit fresh source list; >=32768 native tokens and readable extraction only, no model scores used.",
            scope="Filtered public report prose. Fresh to this study, not proven pretraining-unseen. Reports can quote/reuse earlier text.",
            target_document_ids=[row["document_id"] for row in rows],
            excluded_previous_targets=sorted(excluded),
            donor_id=donor["document_id"],
            donor_text_sha256=sha256(donor["text"].encode()),
            donor_native_tokens=len(donor_ids),
            donor_policy="Exact existing GAO donor text, shared across all replication targets.",
            input_sha256=sha256(panel.read_bytes()),
            input_bytes=panel.stat().st_size,
        ),
    )


if __name__ == "__main__":
    main()
