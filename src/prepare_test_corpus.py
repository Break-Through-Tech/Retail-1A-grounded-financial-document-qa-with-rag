"""Prepare an isolated frozen-test corpus from already available PDFs.

This workflow never reads ``data/pdfs`` or writes ``data/processed``. PDFs must be
placed by a separate, approved process in ``data/test_pdfs`` (or a caller-supplied
directory). It reuses the page extraction and chunk construction functions used by
the development workflow.

Usage::

    python -m src.prepare_test_corpus --pdf-dir data/test_pdfs \
        --out-dir data/test_processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.chunk import chunk_docs, load_pages
from src.data.split import DATASET_SHA256, DEFAULT_DATASET, sha256_of
from src.extract_text import extract_pages
from src.retrieval.queries import DEFAULT_TEST_IDS, load_test_queries

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_DIR = ROOT / "data" / "test_pdfs"
DEFAULT_OUT_DIR = ROOT / "data" / "test_processed"


def test_filings(dataset: Path = DEFAULT_DATASET, ids_path: Path = DEFAULT_TEST_IDS) -> set[str]:
    """Return the source filings named by the frozen test questions."""
    ids = {q.financebench_id for q in load_test_queries(dataset, ids_path)}
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    return {row["doc_name"] for row in rows if row["financebench_id"] in ids}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def _reject_development_path(path: Path, label: str) -> None:
    resolved = path.resolve()
    protected = (
        (ROOT / "data" / "pdfs").resolve(),
        (ROOT / "data" / "processed").resolve(),
    )
    if any(resolved == directory or directory in resolved.parents for directory in protected):
        raise ValueError(f"{label} may not be a development artifact directory: {path}")


def select_test_docs(docs: dict[str, list[dict[str, object]]], required: set[str]) -> dict[str, list[dict[str, object]]]:
    """Keep only required test filings when an extracted page file is broader."""
    return {name: pages for name, pages in docs.items() if name in required}


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_test_corpus(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    dataset: Path = DEFAULT_DATASET,
    ids_path: Path = DEFAULT_TEST_IDS,
) -> dict[str, object]:
    """Extract and chunk only test PDFs into a separate output directory."""
    _reject_development_path(pdf_dir, "pdf_dir")
    _reject_development_path(out_dir, "out_dir")
    required = test_filings(dataset, ids_path)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    present = {pdf.stem for pdf in pdfs}
    missing = sorted(required - present)
    if missing:
        raise FileNotFoundError(f"missing frozen-test PDFs: {missing}")
    unexpected = sorted(present - required)
    if unexpected:
        raise ValueError(f"test PDF directory contains non-test filings: {unexpected}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pages_path = out_dir / "pages.jsonl"
    extract_pages(pdf_dir=pdf_dir, output=pages_path)
    docs = select_test_docs(load_pages(pages_path=pages_path), required)
    missing_extracted = sorted(required - set(docs))
    empty_extracted = sorted(
        name for name in required
        if name in docs and not any(page["text"].strip() for page in docs[name])
    )
    if missing_extracted or empty_extracted:
        raise ValueError(
            f"incomplete extracted corpus; missing={missing_extracted}, "
            f"empty={empty_extracted}"
        )

    outputs: list[Path] = [pages_path]
    chunk_counts: dict[str, int] = {}
    for tag, options in (
        ("page", {"strategy": "page"}),
        ("w300", {"strategy": "words", "size": 300, "overlap": 50}),
        ("w600", {"strategy": "words", "size": 600, "overlap": 100}),
    ):
        chunks = chunk_docs(docs, **options)
        chunk_docs_present = {chunk["doc_name"] for chunk in chunks if chunk["text"].strip()}
        if chunk_docs_present != required or not chunks:
            raise ValueError(
                f"incomplete {tag} chunks; missing={sorted(required - chunk_docs_present)}"
            )
        chunk_counts[tag] = len(chunks)
        path = out_dir / f"chunks_{tag}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        outputs.append(path)

    manifest = {
        "schema_version": 1,
        "purpose": "frozen-test-corpus",
        "git": {"commit": _commit(), "dirty": _dirty()},
        "roots": {
            "pdf_dir": str(pdf_dir.resolve()),
            "output_dir": str(out_dir.resolve()),
        },
        "dataset": {"path": str(dataset), "sha256": sha256_of(dataset)},
        "split": {"ids_path": str(ids_path), "ids_sha256": sha256_of(ids_path)},
        "filings": sorted(required),
        "pdfs": [_file_record(path, pdf_dir) for path in pdfs],
        "artifacts": [_file_record(path, out_dir) for path in outputs],
        "chunking": {"page": {"strategy": "page"}, "w300": {"size": 300, "overlap": 50},
                     "w600": {"size": 600, "overlap": 100}},
        "chunk_counts": chunk_counts,
    }
    if manifest["dataset"]["sha256"] != DATASET_SHA256:
        raise ValueError("dataset hash does not match the committed FinanceBench anchor")
    manifest_path = out_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    manifest = prepare_test_corpus(args.pdf_dir, args.out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
