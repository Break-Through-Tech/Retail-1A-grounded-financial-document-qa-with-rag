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

Use dense search with BAAI/bge-base-en-v1.5. Reranking is not ruled out in general, only for the MS MARCO model that was tried.

## How to reproduce

```
python src/download_pdfs.py
python src/extract_text.py
python src/chunk.py
python -m src.task10_compare --chunks page --retriever dense --model BAAI/bge-base-en-v1.5 --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a --query-prompt "Represent this sentence for searching relevant passages: " --device cpu
```

Use `--device cuda` on a GPU or `--device mps` on an Apple chip. The answer-model half of Task 10 (comparing language models on answer accuracy) is not done yet.