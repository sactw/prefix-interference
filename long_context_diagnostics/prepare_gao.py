"""Prepare the four GAO development reports from their HTML versions.

Requires beautifulsoup4. Existing raw downloads are reused; --refresh fetches
and records a new source snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from .hashing import sha256


REPORTS = (
    ("GAO-25-107024", "target", "GAO-25-107468"),
    ("GAO-25-107363", "target", "GAO-25-107226"),
    ("GAO-25-107468", "donor", None),
    ("GAO-25-107226", "donor", None),
)
EXTRACTION = (
    "gao_letter_prose_v1: after h2 Letter, skip date/addressees until the first "
    "MsoBodyText paragraph >=200 characters that does not start The Honorable; "
    "retain non-table MsoBodyText/MsoListBullet paragraphs and h3/h4 headings; "
    "remove numeric footnote-link anchors; collapse whitespace within blocks; "
    "stop before the next h2 or the distribution/contact closing; join blocks "
    "with two newlines. Omit the standardized We conducted this performance "
    "audit paragraph, images, tables, captions, sources and sidebars."
)
CLOSINGS = (
    "We are sending copies of this report",
    "As agreed with your offices",
    "If you or your staff have any questions",
)


def normalized_text(tag) -> str:
    # Keep inline punctuation attached to its word; rendered whitespace in these
    # Word-exported reports is already present between ordinary inline spans.
    return " ".join(tag.get_text().split())


def extract_prose(raw: bytes) -> tuple[str, dict]:
    soup = BeautifulSoup(raw, "html.parser")
    letter = next(h for h in soup.find_all("h2") if normalized_text(h) == "Letter")
    blocks: list[str] = []
    started = False
    stop_reason = "end_of_document"
    for element in letter.find_all_next(["h2", "h3", "h4", "p"]):
        if element.name == "h2":
            stop_reason = normalized_text(element)
            break
        if element.find_parent("table") is not None:
            continue
        classes = set(element.get("class", []))
        if element.name == "p" and not classes.intersection({"MsoBodyText", "MsoListBullet"}):
            continue
        for anchor in list(element.find_all("a")):
            if re.fullmatch(r"\[\d+\]", normalized_text(anchor)):
                anchor.decompose()
        text = normalized_text(element)
        if not started:
            started = (
                element.name == "p"
                and "MsoBodyText" in classes
                and len(text) >= 200
                and not text.startswith("The Honorable")
            )
            if not started:
                continue
        if text.startswith(CLOSINGS):
            stop_reason = text
            break
        if text.startswith("We conducted this performance audit"):
            continue
        if text:
            blocks.append(text)
    text = "\n\n".join(blocks) + "\n"
    return text, {
        "extraction": EXTRACTION,
        "detected_encoding": soup.original_encoding,
        "stop_before": stop_reason,
        "paragraph_or_heading_count": len(blocks),
        "character_count": len(text),
        "word_count": len(text.split()),
        "replacement_character_count": text.count("\ufffd"),
        "text_sha256": sha256(text.encode("utf-8")),
    }


def fetch_source(raw_dir: Path, report_id: str, refresh: bool) -> tuple[bytes, dict]:
    html_path = raw_dir / f"{report_id}.html"
    meta_path = raw_dir / f"{report_id}.source.json"
    url = f"https://files.gao.gov/reports/{report_id}/index.html"
    if refresh or not html_path.exists():
        request = Request(url, headers={"User-Agent": "long-context-diagnostics/0.1 research"})
        with urlopen(request, timeout=60) as response:
            raw = response.read()
        meta = {
            "url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(raw),
            "source_bytes": len(raw),
        }
        html_path.write_bytes(raw)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    else:
        raw = html_path.read_bytes()
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {"url": url, "retrieved_at": None}
        )
        # Record the downloaded source bytes and checksum.
        meta.update(source_sha256=sha256(raw), source_bytes=len(raw))
    return raw, meta


def word_shingles(text: str, n: int = 13) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    inventory = []
    for report_id, role, donor_id in REPORTS:
        raw, source = fetch_source(raw_dir, report_id, args.refresh)
        text, extraction = extract_prose(raw)
        source.update(extraction)
        row = {
            "document_id": report_id,
            "role": role,
            "split": "development",
            "text": text,
            "source": source,
        }
        if donor_id:
            row["donor_id"] = donor_id
        rows.append(row)
        inventory.append({key: value for key, value in row.items() if key != "text"})
        print(
            f"{report_id}: {extraction['character_count']} chars, "
            f"{extraction['word_count']} words, "
            f"{extraction['paragraph_or_heading_count']} prose blocks"
        )
    output = args.output_dir / "development.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    shingles = [word_shingles(row["text"]) for row in rows]
    overlaps = [
        {
            "left": rows[i]["document_id"],
            "right": rows[j]["document_id"],
            "shared_13_word_shingles": len(shingles[i] & shingles[j]),
            "jaccard_13_word_shingles": (
                len(shingles[i] & shingles[j]) / len(shingles[i] | shingles[j])
            ),
            "same_extracted_text": rows[i]["text"] == rows[j]["text"],
        }
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
    ]
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "development smoke only; not a held-out reporting panel",
        "jsonl_sha256": sha256(output.read_bytes()),
        "documents": inventory,
        "pairwise_overlap": overlaps,
        "overlap_scope": "four extracted documents only, not a pretraining audit",
    }
    (args.output_dir / "preparation.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
