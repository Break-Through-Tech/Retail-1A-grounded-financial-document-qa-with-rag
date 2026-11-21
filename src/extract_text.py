"""Extract page text from every PDF in data/pdfs/ into data/processed/pages.jsonl.

One JSON object per page: doc_name, page_num (0-based, matches the dataset), text.

Usage: python src/extract_text.py
"""

import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
OUT = ROOT / "data" / "processed" / "pages.jsonl"


def extract_pages(pdf_dir: Path = PDF_DIR, output: Path = OUT) -> int:
    """Extract every PDF in ``pdf_dir`` to ``output`` without fixed-path state."""
    output.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0
    with output.open("w", encoding="utf-8") as f:
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            reader = PdfReader(pdf)
            for i, page in enumerate(reader.pages):
                rec = {
                    "doc_name": pdf.stem,
                    "page_num": i,
                    "text": page.extract_text() or "",
                }
                f.write(json.dumps(rec) + "\n")
            print(f"{pdf.stem}: {len(reader.pages)} pages")
            page_count += len(reader.pages)
    return page_count


def main():
    extract_pages()


if __name__ == "__main__":
    main()
