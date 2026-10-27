"""Prepare two complete 2025 GAO report extracts for the 32K development runs.

CPU only; downloads the pinned tokenizer JSON but no model weights. The extracts
keep the published chapter order through appendix III.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from tokenizers import Tokenizer

from .hashing import sha256
from .prepare_gao import fetch_source, normalized_text, word_shingles


TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
REPORTS = (("GAO-25-107743", "target", "GAO-25-107604"), ("GAO-25-107604", "donor", None))
EXTRACTION = (
    "gao_long_prose_v1: start at first >=200-character narrative MsoBodyText "
    "after h2 Letter, excluding The Honorable addressees; preserve chapter "
    "order through appendix III and stop before appendix IV; retain non-table "
    "MsoBodyText/MsoListBullet/MsoListBullet2 paragraphs and h2-h5 headings; "
    "omit numeric footnote anchors, captions, images, tables and sidebars; "
    "skip distribution/signature closing until next h2, Contact Information "
    "headings and email-contact paragraphs, standard audit boilerplate; "
    "collapse intra-block whitespace and join with two newlines."
)
CLOSINGS = (
    "We are sending copies of this report",
    "As agreed with your offices",
    "If you or your staff have any questions",
    "This report was prepared under",
)


def extract_long(raw: bytes) -> tuple[str, dict]:
    soup = BeautifulSoup(raw, "html.parser")
    letter = next(h for h in soup.find_all("h2") if normalized_text(h) == "Letter")
    blocks = []
    started = False
    closing = False
    stop = None
    for element in letter.find_all_next(["h2", "h3", "h4", "h5", "p"]):
        text = normalized_text(element)
        if element.name == "h2":
            if text.startswith("Appendix IV:"):
                stop = text
                break
            closing = False
        if closing or element.find_parent("table") is not None:
            continue
        classes = set(element.get("class", []))
        if element.name == "p" and not classes.intersection(
            {"MsoBodyText", "MsoListBullet", "MsoListBullet2"}
        ):
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
            closing = True
            continue
        if (
            text == "Contact Information"
            or "@gao.gov" in text.lower()
            or text.startswith("We conducted this performance audit")
        ):
            continue
        if text:
            blocks.append(text)
    text = "\n\n".join(blocks) + "\n"
    return text, {
        "extraction": EXTRACTION,
        "detected_encoding": soup.original_encoding,
        "stop_before": stop,
        "character_count": len(text),
        "word_count": len(text.split()),
        "prose_blocks": len(blocks),
        "text_sha256": sha256(text.encode("utf-8")),
        "replacement_character_count": text.count("\ufffd"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = try_to_load_from_cache(TOKENIZER_ID, "tokenizer.json", revision=TOKENIZER_REVISION)
    path = (
        cached
        if isinstance(cached, str)
        else hf_hub_download(
            TOKENIZER_ID,
            "tokenizer.json",
            revision=TOKENIZER_REVISION,
            cache_dir=args.output_dir / "tokenizer_cache",
        )
    )
    tokenizer = Tokenizer.from_file(path)
    rows, token_lists = [], []
    for report_id, role, donor in REPORTS:
        raw, source = fetch_source(raw_dir, report_id, args.refresh)
        text, extraction = extract_long(raw)
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        source.update(
            extraction,
            tokenizer_id=TOKENIZER_ID,
            tokenizer_revision=TOKENIZER_REVISION,
            native_tokens=len(token_ids),
            special_tokens_added=False,
        )
        row = {
            "document_id": report_id,
            "role": role,
            "split": "development",
            "text": text,
            "source": source,
        }
        if donor:
            row["donor_id"] = donor
        rows.append(row)
        token_lists.append(token_ids)
        print(report_id, len(text), "characters;", len(token_ids), "native tokens")
    output = args.output_dir / "development.jsonl"
    output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    prefix_texts = [tokenizer.decode(ids[:n]) for ids, n in zip(token_lists, (32768, 29696))]
    shingles = [word_shingles(text) for text in prefix_texts]
    words = [re.findall(r"\w+", text.lower()) for text in prefix_texts]
    shared = shingles[0] & shingles[1]
    positions = []
    for sequence in words:
        locations = {}
        for index in range(len(sequence) - 12):
            key = tuple(sequence[index : index + 13])
            if key in shared:
                locations.setdefault(key, []).append(index)
        positions.append(locations)
    matches = set()
    for key in shared:
        for left in positions[0][key]:
            for right in positions[1][key]:
                a, b, size = left, right, 13
                while a and b and words[0][a - 1] == words[1][b - 1]:
                    a -= 1
                    b -= 1
                    size += 1
                while (
                    a + size < len(words[0])
                    and b + size < len(words[1])
                    and words[0][a + size] == words[1][b + size]
                ):
                    size += 1
                matches.add((a, b, size))
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "one-target development feasibility pilot; not held-out reporting",
        "pilot": {"E": 32768, "C": 2048, "T": 1024, "donor_prefix_tokens": 29696},
        "jsonl_sha256": sha256(output.read_bytes()),
        "documents": [{k: v for k, v in row.items() if k != "text"} for row in rows],
        "prefix_overlap": {
            "scope": "target first 32768 native tokens vs donor first 29696",
            "shared_13_word_shingles": len(shared),
            "jaccard_13_word_shingles": len(shared) / len(shingles[0] | shingles[1]),
            "largest_matching_word_blocks": [
                {"words": size, "text": " ".join(words[0][a : a + size])}
                for a, b, size in sorted(matches, key=lambda m: (-m[2], m[0], m[1]))[:8]
            ],
        },
        "target_scored_text": tokenizer.decode(token_lists[0][31744:32768]),
        "target_prefix_start": prefix_texts[0][:1200],
        "donor_prefix_start": prefix_texts[1][:1200],
        "donor_prefix_end": prefix_texts[1][-3000:],
    }
    (args.output_dir / "preparation.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Prefix overlap:", len(shared), "shared 13-word shingles;")
    print("Wrote", output)


if __name__ == "__main__":
    main()
