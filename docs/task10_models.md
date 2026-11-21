# Task 10: retrieval and model comparison (development set)

All numbers are on the 111 development questions and the 58 development filings, with 1-page passages and the company/year filter on. Source: `results/task10-comparison.csv`, produced by `src/task10_compare.py`.

## Results (Hit@5 = the right evidence page is in the top 5)

| Method | Model | Hit@1 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| BM25 (keyword) | none | 0.117 | 0.234 | 0.179 |
| Dense | BAAI/bge-small-en-v1.5 | 0.261 | 0.495 | 0.352 |
| Dense | BAAI/bge-base-en-v1.5 | 0.279 | 0.505 | 0.376 |
| Hybrid (BM25 + dense, RRF) | bge-small | 0.180 | 0.387 | 0.279 |
| Rerank (BM25 + dense, then cross-encoder) | bge-small + ms-marco-MiniLM-L-6-v2 | 0.189 | 0.468 | 0.310 |

Passage size (1 page, 300 words, 600 words) barely changed any result, so 1 page is used.

## What we found

- Dense search roughly doubles keyword search (Hit@5 about 50% against 23%).
- A larger embedding model adds little (bge-base finds one more question than bge-small).
- Combining BM25 with dense, or reranking with the MS MARCO cross-encoder, did not beat dense alone.
- The extracted PDF text contains the gold evidence for 105 of 111 questions, so text extraction is not the bottleneck.
- Best Hit@5 is 50.5%. The challenge target is 70%.

## Models and pinned revisions

The repo did not name any models, so these were chosen for the comparison. The code requires a full 40-character revision.

| Model | Role | Revision | Run? |
|---|---|---|---|
| BAAI/bge-small-en-v1.5 | dense embedding | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a | yes |
| BAAI/bge-base-en-v1.5 | dense embedding (best so far) | a5beb1e3e68b9ab74eb54cfd186867f64f240e1a | yes |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | reranker | 233902d25c440f23af6f7d6e94d2946bac0bee0a | yes, did not help |
| BAAI/bge-large-en-v1.5 | dense embedding | d4aa6901d3a41ba39fb536a557fa166f842b0e09 | not run |
| BAAI/bge-reranker-base | reranker | 2cfc18c9415c912f9d8155881c133215df768a70 | not run |

Dense queries use the prompt `Represent this sentence for searching relevant passages: `.

## Recommendation

Use dense search with BAAI/bge-base-en-v1.5, with qwen3:14b as the answer model (more accurate than qwen3:8b, but about twice as slow; see "Answer models" below).

## How to reproduce

```
python src/download_pdfs.py
python src/extract_text.py
python src/chunk.py
python -m src.task10_compare --chunks page --retriever dense --model BAAI/bge-base-en-v1.5 --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a --query-prompt "Represent this sentence for searching relevant passages: " --device cpu
```

Use `--device cuda` on a GPU or `--device mps` on an Apple chip. 

## Answer models (runs 15–17)

Same pipeline and prompt as Task 9, all 111 dev questions, top 5 pages, company/year filter on, qwen3 via Ollama (temperature 0, seed 42, 8192 context). Only the search and the answer model change.

| Run | Search | Answer model | Gold page retrieved | Gold page cited | Quotes on page | Answered / abstained | Seconds per question |
|---|---|---|---|---|---|---|---|
| 15 | BM25 | qwen3:8b | 26/111 (23%) | 21/111 (19%) | 77/102 | 85 / 26 | 13.1 |
| 16 | dense bge-base | qwen3:8b | 56/111 (50%) | 53/111 (48%) | 96/131 | 106 / 5 | 12.7 |
| 17 | dense bge-base | qwen3:14b | 56/111 (50%) | 46/111 (41%) | 86/114 | 85 / 26 | 24.0 |

Answer correctness was graded by hand against the answer key, on the 56 questions where search found the gold page (the only questions that can be answered correctly from the pages given). The first pass was done by Claude and spot-checked by Benita.

| Run | Correct | Partly | Wrong | Abstained |
|---|---|---|---|---|
| 16 (8B) | 29 (52%) | 6 | 21 | 0 |
| 17 (14B) | 35 (62%) | 3 | 11 | 7 |

Findings:

- Dense search is the largest single improvement: gold page retrieved rose from 23% to 50%, and abstentions fell from 26 to 5 with the same 8B model.
- The 14B model is more accurate (62% against 52% of the 56) and wrong less often (11 against 21). Its extra abstentions mostly avoided wrong answers: the 8B answered those 7 questions with 1 right and 6 wrong.
- The 8B often fails at multi-step arithmetic and sometimes reports a dollar figure instead of the margin asked for. Both models fail on the hardest ratios (days payable outstanding, inventory turnover, cash conversion cycle).
- The 14B takes about twice as long per question and cites the gold page less often (41% against 48%) because it abstains more.
- Lower bound for answer accuracy over all 111 questions: 8B 29/111 (26%), 14B 35/111 (32%), against the 55% target. About half the questions still never get the gold page, so search is the main limit.
- Three questions (02608, 03838, 04417) used 7,190–7,836 of 8,192 tokens, so their prompts or replies may be cut off.

Reproduce:

    python -m src.pipeline --all-dev --llm ollama --model qwen3:8b --retriever dense --device cuda --out results/run2_dense_qwen3-8b.jsonl
    python -m src.pipeline --all-dev --llm ollama --model qwen3:14b --retriever dense --device cuda --out results/run3_dense_qwen3-14b.jsonl
    python -m src.check_pipeline_run --run results/run2_dense_qwen3-8b.jsonl --show 0