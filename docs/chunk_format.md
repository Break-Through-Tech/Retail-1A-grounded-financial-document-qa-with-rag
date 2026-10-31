# Chunk format

This is the one input that BM25, dense retrieval, the pipeline and evaluation all read. If you change a field here, tell everyone who reads it.

## How to build it

From the repository root:

```
python src/download_pdfs.py     # PDFs into data/pdfs/
python src/extract_text.py      # page text into data/processed/pages.jsonl
python src/chunk.py             # chunks into data/processed/
```

`chunk.py` writes three files, one per chunking strategy:

| File | One chunk is | Chunks now |
|---|---|---|
| `chunks_page.jsonl` | one whole page (the team default) | 1,610 |
| `chunks_w300.jsonl` | 300 words, 50 words of overlap | 3,748 |
| `chunks_w600.jsonl` | 600 words, 100 words of overlap | 2,286 |

The counts are for the 12 dev filings downloaded so far. They grow as more filings are added.

## One line, one chunk

Each file is JSON Lines: one JSON object per line.

```json
{
  "chunk_id": "AMD_2022_10K_p46_0",
  "doc_name": "AMD_2022_10K",
  "page_num": 46,
  "section": "ITEM 7 MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION",
  "text": "The most significant assumptions utilized in the determination of the estimated ...",
  "company": "AMD",
  "doc_type": "10k",
  "doc_period": 2022,
  "doc_link": "https://ir.amd.com/sec-filings/content/0000002488-23-000047/0000002488-23-000047.pdf"
}
```

| Field | Type | Meaning |
|---|---|---|
| `chunk_id` | text | Unique ID: `{doc_name}_p{page_num}_{n}`. `n` counts the chunks on that page, from 0. A page chunk always ends in `_0`. |
| `doc_name` | text | Filing name, the same as `doc_name` in the dataset. |
| `page_num` | number | Page in the PDF, counted from 0, like the dataset's `evidence_page_num`. Add 1 to find it in a PDF viewer. |
| `section` | text | The last SEC heading seen, such as "Item 7". Empty if none was found yet. |
| `text` | text | The cleaned page text, or a piece of it. |
| `company` | text | Company name, from the dataset. |
| `doc_type` | text | `10k`, `10q`, `8k` or `Earnings`, from the dataset. |
| `doc_period` | number | Year of the filing, from the dataset. |
| `doc_link` | text | Link to the original filing, from the dataset. |

## Rules

- **Dev filings only.** Only filings named by `data/splits/dev_ids.txt` are chunked. The test ID file is never opened.
- **No chunk crosses a page.** Word chunks are cut inside one page, so every chunk belongs to exactly one page and can be cited by page.
- **IDs are stable.** Rebuilding from the same PDFs gives the same chunks and the same IDs. A different pypdf version could change the text slightly, which could shift word-chunk IDs. Page-chunk IDs stay the same.
- **Matching to the dataset.** The dataset calls the page `evidence_page_num`. Chunks call it `page_num`. Rename it when scoring with `src/evaluation/metrics.py`.

## Reading it

```python
import json
from src.retrieval.bm25 import BM25Index

chunks = [json.loads(line) for line in open("data/processed/chunks_page.jsonl")]
index = BM25Index.build((c["chunk_id"], c["text"]) for c in chunks)
```
