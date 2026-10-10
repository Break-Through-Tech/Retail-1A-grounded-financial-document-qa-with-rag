"""Turn extracted page text into chunks with metadata.

Run src/extract_text.py first to create data/processed/pages.jsonl.

Strategies:
  page            one chunk per page
  words           fixed-size word windows inside each page, with overlap

Usage: python src/chunk.py            (writes data/processed/chunks_page.jsonl etc.)
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "data" / "processed" / "pages.jsonl"
OUT_DIR = ROOT / "data" / "processed"
DATASET = ROOT / "data" / "financebench_merged.jsonl"

# Repeated page furniture that adds noise to every chunk.
JUNK = [
    re.compile(r"^table of contents$", re.I),
    re.compile(r"^page \d+ of \d+$", re.I),
    re.compile(r"^\d{1,3}$"),
]

# SEC section headings, e.g. "Item 7." or "Item 1A."
SECTION = re.compile(r"^\s*(Item\s+\d+[A-Z]?)\.?\s*(.{0,60})", re.I)


def clean(text):
    """Drop repeated headers/footers and collapse blank lines."""
    lines = [l.strip() for l in text.splitlines()]
    kept = [l for l in lines if l and not any(p.match(l) for p in JUNK)]
    return "\n".join(kept)


def find_section(text, current):
    """Return the section heading this page belongs to."""
    for line in text.splitlines()[:15]:
        m = SECTION.match(line)
        if m:
            return f"{m.group(1)} {m.group(2)}".strip()
    return current


def doc_metadata():
    """doc_name -> company, filing period, type and source URL, taken from the dataset."""
    meta = {}
    for line in DATASET.open():
        r = json.loads(line)
        meta[r["doc_name"]] = {
            "company": r["company"],
            "doc_type": r["doc_type"],
            "doc_period": r["doc_period"],
            "doc_link": r["doc_link"],
        }
    return meta


def load_pages():
    """Group cleaned pages by document, tagging each with its section."""
    docs = {}
    for line in PAGES.open():
        r = json.loads(line)
        docs.setdefault(r["doc_name"], []).append(r)
    for name, pages in docs.items():
        pages.sort(key=lambda r: r["page_num"])
        section = ""
        for p in pages:
            p["text"] = clean(p["text"])
            section = find_section(p["text"], section)
            p["section"] = section
    return docs


def chunk_docs(docs, strategy="page", size=300, overlap=50):
    """Build chunks. Each chunk keeps the metadata needed for citations."""
    chunks = []
    meta = doc_metadata()
    for name, pages in docs.items():
        for p in pages:
            if not p["text"].strip():
                continue
            if strategy == "page":
                pieces = [p["text"]]
            elif strategy == "words":
                words = p["text"].split()
                step = size - overlap
                pieces = [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)]
                pieces = [x for x in pieces if x]
            else:
                raise ValueError(strategy)

            for j, text in enumerate(pieces):
                chunks.append({
                    "chunk_id": f"{name}_p{p['page_num']}_{j}",
                    "doc_name": name,
                    "page_num": p["page_num"],
                    "section": p["section"],
                    "text": text,
                    **meta.get(name, {}),
                })
    return chunks


def main():
    docs = load_pages()
    for tag, kw in [
        ("page", dict(strategy="page")),
        ("w300", dict(strategy="words", size=300, overlap=50)),
        ("w600", dict(strategy="words", size=600, overlap=100)),
    ]:
        chunks = chunk_docs(docs, **kw)
        out = OUT_DIR / f"chunks_{tag}.jsonl"
        with out.open("w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"{tag}: {len(chunks)} chunks -> {out.name}")


if __name__ == "__main__":
    main()
