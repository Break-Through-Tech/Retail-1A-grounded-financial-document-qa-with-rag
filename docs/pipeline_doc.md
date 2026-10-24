# RAG pipeline (Issue #23)

The first end-to-end pipeline. A development question goes in; an answer with citations comes out, in the format of [`response_structure.md`](response_structure.md).

## How it works

1. **Search.** BM25 (`src/retrieval/bm25.py`) ranks every page chunk. Only pages from the company and year named in the question are kept, using `allowed()` from `src/eval_retrieval.py`. The top 5 go on.
2. **Prompt.** The 5 pages are numbered `[1]`–`[5]`, best first. The model must answer only from them, show any calculation step by step, and cite each source it uses by number with a short exact quote. If the pages would not fit in the context window, the page that overflows is cut short and lower-ranked pages are left out (`estimate_tokens()` counts every digit as a token, as Qwen does).
3. **Model.** A local model served by Ollama (default `qwen3:8b`): temperature 0, seed 42, an 8,192-token context window, replies capped at 1,024 tokens, thinking off, and JSON output forced into a fixed shape. `--llm fake` (the default) runs everything with a stand-in model, for tests and plumbing checks.
4. **Format.** Each citation number is turned into the filing name, page, company and link from the chunk metadata, so the model cannot invent a page. Numbers that don't match a source are dropped.

Page numbers are 0-based, like the dataset. Add 1 to find a page in a PDF viewer.

## Setup (once)

```powershell
pip install -r requirements.txt
python src/download_pdfs.py
python src/extract_text.py
python src/chunk.py
```

Then install Ollama from https://ollama.com and run `ollama pull qwen3:8b` (about 5 GB).

## Run

```powershell
python -m src.pipeline --id financebench_id_00705                # one question, fake model
python -m src.pipeline --id financebench_id_00705 --llm ollama   # one question, real model
python -m src.pipeline --all-dev --llm ollama                    # all dev questions we have filings for
python -m src.check_pipeline_run                                 # score the run against the dev answer key
```

`--all-dev` writes `results/pipeline_dev_run.jsonl` plus `results/pipeline_dev_run.meta.json`, which records the model, settings and git commit of the run. Only development IDs run; frozen test IDs are refused.

### Why the context window is 8,192 tokens

At 8,192, `qwen3:8b` fits entirely on an 8 GB graphics card. With a larger window part of the model spills onto the CPU, and Ollama on Windows then crashes during warm-up with `CUDA error: shared object initialization failed` (exit status `0xc0000409`; see [ollama/ollama#17380](https://github.com/ollama/ollama/issues/17380)). On a card with more memory, pass `--num-ctx 16384` to send longer pages untrimmed.

### Windows: use Ollama's CUDA 12 engine

On an RTX 3070 laptop with Ollama 0.40.10, the default CUDA 13 engine wrote garbage (repeated commas) once a prompt passed about 2,500 tokens. Ollama's bundled CUDA 12 engine answers correctly. Switch to it once, then quit and reopen Ollama:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_LLM_LIBRARY", "cuda_v12", "User")
```

Even on CUDA 12, Ollama occasionally crashes with `CUDA error: an illegal memory access was encountered` and restarts its engine. The pipeline retries each model call up to 3 times; a question that still fails gets an `error` field and the run moves on.

## Where the next tasks plug in

| Task | Where |
|---|---|
| Insufficient evidence (#24) | `abstained` and `confidence` in `to_response()`. `answer_question()` already skips the model when search finds nothing. |
| Rerank (#22), dense retrieval (#18, #19) | Replace `retrieve()`; everything after it stays the same. Fill `rerank_score`. |
| Answer and citation format (#25) | `to_response()`. `check_pipeline_run` reports how many quotes really appear on the page they cite. |
| Configuration comparison (#26) | `--k`, `--model`, `--num-ctx`, and the chunk file. |

## Known limits

- Search finds a gold evidence page for about half of the dev questions (Hit@5 = 0.50, same as the BM25 baseline). The model cannot answer correctly when the evidence wasn't retrieved.
- Only the 16 downloaded filings are searchable, which covers 48 of the 111 dev questions.
- With the 8,192-token window, the lowest-ranked page is cut short or left out when the 5 pages are long. `retrieved_but_unused` still lists pages that were left out.
- A model reply that isn't valid JSON, or a model call that fails 3 times, is recorded with an extra `error` field, which is not in the response spec yet.
- `retrieved_but_unused` lists pages as `"DOC_NAME p<page>"`.
