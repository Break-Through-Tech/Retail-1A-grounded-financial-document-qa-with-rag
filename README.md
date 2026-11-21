# Grounded Financial Document QA with RAG

**Break Through Tech AI Studio, Fall 2026 | Retail 1A**

This repository implements a grounded question-answering system for public financial filings. It combines document ingestion, page-level chunking, BM25 and dense retrieval, reranking, source-aware response formatting, and insufficient-evidence abstention for the FinanceBench benchmark.

## Team Members

| Name | GitHub Handle | Repository contributions |
|---|---|---|
| Benita Benjamin | [@benitabenjamin-cmd](https://github.com/benitabenjamin-cmd) | Dense retrieval, reranking, and answer-model experiments |
| Marceliott Encarnacion | [@marceliottencarnacion](https://github.com/marceliottencarnacion) | Data verification, pipeline construction, and tuning |
| Kerwyn Jean | [@kjean230](https://github.com/kjean230) | Baseline computation, vector indexing, and training set creation |
| Anjena Raja | [@anjenaraja](https://github.com/anjenaraja) | Repository coordination and evaluation infrastructure |
| Shruti Senthilram | [@shrutisenthilram](https://github.com/shrutisenthilram) | PDF parsing while maintaining document structure and format context |
| Maanveer Singh | [@ms04](https://github.com/ms04) | Initial search experiments, chunking comparisons, and logging |
| Kanishka Yadav | [@kanishkay](https://github.com/kanishkay) | Retrieval metrics and search evaluation |


## Project Highlights

- Created a deterministic, filing-level development/test split: 111 development questions across 58 filings and 39 frozen-test questions across 26 filings.
- Built a reproducible ingestion and chunking pipeline for PDF pages and overlapping 300-word/600-word chunks.
- Implemented Lucene-style BM25, exact FAISS cosine-similarity dense retrieval, hybrid reciprocal-rank fusion, and cross-encoder reranking.
- Compared retrieval models and chunk sizes using saved, per-question evaluation artifacts.
- Added grounded response structures and a two-stage abstention mechanism that checks evidence scope and confidence before and after generation.
- Added frozen-test evaluation infrastructure that requires a clean worktree and validates corpus, split, configuration, and artifact hashes.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/Break-Through-Tech/Retail-1A-grounded-financial-document-qa-with-rag.git
cd Retail-1A-grounded-financial-document-qa-with-rag
```

### 2. Create a Python environment

Python 3.10 or newer is recommended. A virtual environment keeps the retrieval and test dependencies isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The Python dependencies include pandas, NumPy, scikit-learn, Sentence Transformers, FAISS, pypdf, BeautifulSoup, Jupyter, and pytest.

### 3. Use the included dataset

The repository already includes `data/financebench_merged.jsonl`, a 150-row public FinanceBench sample. It is pinned by SHA-256 and loaded by the split module. Verify it with:

```bash
shasum -a 256 data/financebench_merged.jsonl
```

Expected SHA-256:

```text
7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b
```

See [data/README.md](data/README.md) for dataset provenance.

### 4. Download and process source PDFs

The public source PDFs are downloaded through the approved FinanceBench repository. The script reads only the development filing list and writes PDFs under `data/pdfs/`:

```bash
python src/download_pdfs.py
python src/extract_text.py
python src/chunk.py
```

This produces `data/processed/pages.jsonl` and page, 300-word, and 600-word chunk variants. The chunking stage only indexes development filings and never opens the frozen-test ID file.

### 5. Rebuild the development/test split

The committed split is reproducible and may be regenerated with:

```bash
python -m src.data.split --dry-run
python -m src.data.split
```

The split uses seed 42 and groups rows by `doc_name`, keeping all questions about the same filing on one side.

### 6. Run retrieval comparisons

Run one or more searches after processing the PDFs:

```bash
# BM25
python -m src.task10_compare --chunks page --retriever bm25

# Dense retrieval with a pinned embedding model
python -m src.task10_compare --chunks page --retriever dense \
  --model BAAI/bge-base-en-v1.5 \
  --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a \
  --device cpu

# Hybrid BM25 + dense retrieval
python -m src.task10_compare --chunks page --retriever hybrid \
  --model BAAI/bge-base-en-v1.5 \
  --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a \
  --device cpu

# Reranking
python -m src.task10_compare --chunks page --retriever rerank \
  --model BAAI/bge-small-en-v1.5 \
  --revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a \
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank-revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --device cpu
```

Use `--device mps` on Apple Silicon or `--device cuda` on CUDA-enabled hardware. Retrieval results are appended to [results/task10-comparison.csv](results/task10-comparison.csv).

To explore the dataset interactively, open [notebooks/01_explore_data.ipynb](notebooks/01_explore_data.ipynb).

### 7. Run tests

```bash
pytest -q
```

The test suite covers BM25, dense retrieval, embedding provenance, reranking, metrics, split isolation, response structure, abstention policy/tuning, and frozen-test evaluation.

### Optional answer-generation setup

Answer-model experiments used Ollama with Qwen3 models:

```bash
ollama pull qwen3:8b
ollama pull qwen3:14b
```

The answer experiments compared BM25 and dense top-5 evidence on the 111-question development set. Ollama is an external runtime; it is not installed by `requirements.txt`.

## Project Overview

This project is part of the Break Through Tech AI Studio Retail 1A challenge. The objective is to build a retrievable, reproducible, and grounded QA system that helps users obtain answers from long public financial filings without inventing unsupported facts.

The working scope is single-document questions over filing text and text extracted from tables. The system accepts a natural-language question and returns either:

- A grounded answer, cited by filing and 0-based PDF page number, or
- An insufficient-evidence response with a structured abstention reason.

The implemented architecture separates:

1. Dataset splitting and validation
2. PDF extraction and chunk construction
3. Keyword, dense, hybrid, and reranked retrieval
4. Evidence-scope and confidence checks
5. Answer generation and citation verification
6. Development metrics and frozen-test evaluation

This separation makes retrieval failures distinguishable from answer-generation failures and prevents frozen-test IDs or filings from entering development-only tuning.

## Data Exploration

The primary dataset is the public 150-row FinanceBench JSONL sample. A row includes:

- Stable question ID
- Company, filing name, period, and filing type
- Question and question category
- Gold answer and human justification
- Gold evidence passage and 0-based page number
- Source-document URL
- GICS sector

The split contains 111 development rows and 39 frozen-test rows across 84 source filings. It is partitioned by `doc_name`; no filing appears on both sides. The split uses `question_type` and `doc_type` as balancing strata and has a fixed SHA-256, seed, and 40-row target test size.

| Split | Questions | Filings |
|---|---:|---:|
| Development | 111 | 58 |
| Frozen test | 39 | 26 |

Question distribution in the development split:

- 38 novel-generated
- 37 metrics-generated
- 36 domain-relevant
- 82 10-K questions
- 23 questions requiring evidence from two passages
- 3 questions requiring evidence from three passages

The exploration notebook [notebooks/01_explore_data.ipynb](notebooks/01_explore_data.ipynb) inspects the JSONL data and can be run with Jupyter.

### Chunking and preprocessing

The ingestion pipeline removes repeated table-of-contents headings and page furniture, preserves SEC section context, and produces chunks that do not cross PDF page boundaries. Every chunk retains its filing name, page, section, company, filing type, period, and source URL.

| Chunking strategy | Chunk definition | Development chunks |
|---|---|---:|
| Page | One complete extracted PDF page | 1,610 |
| 300-word | 300-word windows with 50-word overlap | 3,748 |
| 600-word | 600-word windows with 100-word overlap | 2,286 |

The chunk format and reconstruction rules are documented in [docs/chunk_format.md](docs/chunk_format.md).

### Key data findings

- Gold evidence for 105 of 111 development questions is present in the extracted PDF text.
- BM25 performs well for questions using filing terminology but misses paraphrases such as “quick ratio,” “healthy liquidity profile,” and “customer concentration.”
- Numerical and logical-reasoning questions are difficult because the query often does not name the pages that contain the needed values.
- Page-level chunks were selected as the default: page chunks matched or beat 300/600-word chunks in the BM25 chunking comparison.
- Multi-page evidence remains a weakness. In a 48-question filing-downloaded subset, no tested setup found every gold page for every multi-page question.

## Model Development

### Retrieval

The baseline is Lucene-style Okapi BM25:

- K1: 1.2
- B: 0.75
- Lucene-style positive IDF
- NFKC normalization and case folding
- No stop-word removal or stemming
- Deterministic `(-score, chunk_id)` tie-breaking

Dense retrieval uses exact cosine similarity over a FAISS `IndexFlatIP` index. The evaluated dense model was `BAAI/bge-base-en-v1.5`, pinned to revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.

Additional evaluated approaches were:

- BM25 + dense reciprocal-rank fusion
- BM25 + dense candidate generation followed by `cross-encoder/ms-marco-MiniLM-L-6-v2`
- 300-word and 600-word chunk variants
- Company and fiscal-year metadata filtering

The reranker did not improve over dense search in the 111-question development comparison.

### Answer generation and abstention

The grounded pipeline:

1. Retrieves candidate passages.
2. Optionally reranks them.
3. Filters evidence by named company and fiscal year.
4. Abstains if retrieval is empty, evidence is outside the question's scope, or the reranker score is below a development-tuned threshold.
5. Generates an answer only from eligible evidence.
6. Abstains again if the generator declines or returns an answer without a citation.

The abstention policy is separate from the generator and is pinned to one scorer and retrieval pipeline. It is tuned only on development data, with constructed withheld-filing cases for unsupported evidence. Development metrics include unsupported-answer rate, answer coverage, abstention precision, and abstention recall.

The response format is documented in [docs/response_structure.md](docs/response_structure.md).

## Results and Key Findings

All headline retrieval numbers below are development-set results, not frozen-test results.

### Retrieval comparison

| Method | Model | Hit@1 | Hit@5 | MRR@10 |
|---|---|---:|---:|---:|
| BM25 | — | 11.7% | 23.4% | 17.9% |
| Dense | `bge-small-en-v1.5` | 26.1% | 49.5% | 35.2% |
| Dense | `bge-base-en-v1.5` | 27.9% | 50.5% | 37.6% |
| Hybrid RRF | `bge-small` + BM25 | 18.0% | 38.7% | 27.9% |
| Rerank | `bge-small` + MS MARCO MiniLM | 18.9% | 46.8% | 31.0% |

The selected development search configuration is dense retrieval with `bge-base-en-v1.5`, one-page chunks, and company/year filtering. It found a gold evidence page in the top five for 56 of 111 development questions.

In an earlier 48-question subset, BM25 with company/year filtering achieved:

- Hit@1: 22.9%
- Hit@5: 47.9%
- MRR@10: 32.2%

This subset is smaller and is not directly comparable to the 111-question Task 10 run.

### Answer-model comparison

The answer stage was tested with BM25 and dense top-5 retrieval using qwen3 through Ollama. Hand grading was limited to the 56 questions for which dense search actually retrieved the gold page.

| Answer model | Gold page retrieved | Gold page cited | Answered | Abstained | Correct | Partly correct | Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3:8b | 56/111 | 53/111 | 106 | 5 | 29/56 | 6/56 | 21/56 |
| qwen3:14b | 56/111 | 46/111 | 85 | 26 | 35/56 | 3/56 | 11/56 |

Observed development findings:

- Dense retrieval increased gold-page retrieval from 23.4% to 50.5% compared with BM25.
- The 14B model was more accurate within retrieved-evidence questions: 62.5% correct versus 51.8% for 8B.
- The 14B model abstained more and was slower, averaging about 24 seconds per question compared with 12.7 seconds for 8B.
- Both models struggled with multi-step arithmetic and ratios such as days payable outstanding, inventory turnover, and cash conversion cycle.
- The 8B model sometimes returned a dollar figure instead of the margin requested.
- Three long prompts exceeded or approached the configured 8,192-token context limit.

No frozen-test headline metrics or answer-accuracy score are currently committed. The repository includes the infrastructure to perform that evaluation once rankings and the isolated test corpus are prepared.

### Results against challenge targets

The 111-question development retrieval result has not reached the 70% Hit@5 challenge target. Dense `bge-base-en-v1.5` reached 50.5%, a 27.1 percentage-point improvement over the 23.4% BM25 baseline.

The 55% answer-accuracy and 85% citation-correctness targets cannot yet be reported as final project metrics because answer grading was performed only in the retrieved-evidence subset and no complete frozen-test answer evaluation is saved. The 14B model's 62.5% development correctness among retrieved-evidence questions is promising but is not a defensible full-sample answer-accuracy result.

## Next Steps

1. Prepare and validate the 26 frozen-test PDFs and isolated test corpus.
2. Run the separately frozen retrieval configuration once on the 39 frozen-test questions.
3. Report and save frozen-test Hit@1, Hit@5, MRR@10, and all-pages Hit@5.
4. Complete human answer and citation grading using the frozen test set.
5. Improve retrieval beyond 50.5% Hit@5, especially for semantic paraphrases, calculations, and multi-page questions.
6. Evaluate stronger rerankers and table-aware chunking while preserving page citations.
7. Tune and validate abstention thresholds on development data only; measure coverage, unsupported-answer rate, precision, recall, and confidence calibration.
8. Extend the pipeline to support a complete model run through a documented command and add latency and cost reporting to output artifacts.
9. Add a lightweight demo for browsing retrieved passages, citations, confidence, and abstention decisions.
10. Finalize a license and add project-specific citations, acknowledgements, and a reproducible demo notebook.

## License

No repository license is currently declared. Licensing should be confirmed with the Challenge Advisor before publishing or redistributing the project.

## References

- [FinanceBench: A New Benchmark for Financial Question Answering](https://arxiv.org/abs/2311.11944)
- [Patronus AI FinanceBench dataset](https://huggingface.co/datasets/PatronusAI/financebench)
- [Patronus AI FinanceBench source repository](https://github.com/patronus-ai/financebench)
- [Sentence Transformers semantic search](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Sentence Transformers retrieve-and-rerank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

## Acknowledgements

We thank Challenge Advisor SaiSandeep Kantareddy, AI Coach Alexandra Ladyzhensky, and Break Through Tech for project guidance and support.
