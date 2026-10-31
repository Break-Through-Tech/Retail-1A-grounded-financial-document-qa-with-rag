# October Task #3: Dataset exploration and search test

Maanveer Singh and Kanishka | Retail 1A | October 16, 2026

Only development questions are used here. The frozen test questions are never read.

To rerun: `python -m src.task3_search_test` from the repository root. Add `--log` to also write the BM25 runs to the Task #9 log. It needs `data/processed/chunks_page.jsonl`, which comes from `src/extract_text.py` and then `src/chunk.py`.

## Part 1: What the dev questions look like

There are 111 dev questions over 58 filings from 29 companies.

| Kind | Questions |
|---|---|
| Question type: novel-generated | 38 |
| Question type: metrics-generated | 37 |
| Question type: domain-relevant | 36 |
| Document: 10-K | 82 |
| Document: 10-Q | 11 |
| Document: earnings release | 11 |
| Document: 8-K | 7 |
| Needs 1 evidence passage | 85 |
| Needs 2 passages | 23 |
| Needs 3 passages | 3 |

Most questions come from 10-Ks. About a quarter need evidence from more than one page. Gold evidence is usually short: the median is 196 words and the longest is 717.

## Part 2: Search test

The test uses the 48 dev questions whose filing we have downloaded, covering 12 filings. It uses page chunks and two retrievers: TF-IDF from Task #5, and the team's BM25 from `src/retrieval/bm25.py`. Each one runs with the metadata filter off and on. Hit@1, Hit@5 and MRR@10 come from Kanishka's `src/evaluation/metrics.py`.

| Retriever | Filter | Hit@1 | Hit@5 | Text Hit@5 | MRR@10 |
|---|---|---|---|---|---|
| TF-IDF | off | 0.021 | 0.188 | 0.208 | 0.087 |
| TF-IDF | on | 0.021 | 0.208 | 0.208 | 0.113 |
| BM25 | off | 0.188 | 0.333 | 0.354 | 0.243 |
| BM25 | on | 0.229 | 0.500 | 0.562 | 0.322 |

BM25 with the filter is the best setup, finding the right page in the top 5 for half the questions. The TF-IDF numbers match runs 19 and 20 in the Task #9 log. The BM25 runs are logged as runs 25 and 26.

### Hit@5 by kind of question (BM25, filter on)

| Kind | Questions | Hit@5 |
|---|---|---|
| novel-generated | 27 | 0.556 |
| domain-relevant | 19 | 0.421 |
| metrics-generated | 2 | 0.500 |
| Information extraction | 10 | 0.600 |
| Numerical reasoning | 6 | 0.333 |
| Logical reasoning | 5 | 0.200 |
| 8-K | 5 | 1.000 |
| 10-Q | 8 | 0.625 |
| Earnings release | 9 | 0.444 |
| 10-K | 26 | 0.385 |

The full table, with all four setups, is in `results/task3/hit5_by_group.csv`.

## What search does well and badly on

**Does well:** questions that use the same words as the filing. Examples are "Which debt securities are registered to trade on a national securities exchange", which is cover-page wording, "What are the major products and services that AMD sells", and "change in the number of Best Buy stores". Short filings also do well: 8-Ks have only a few pages, so the right one is easy to find.

**Does badly:** questions that use analyst words the filing never says. Examples are "quick ratio", "healthy liquidity profile", "customer concentration", "cyclicality", "materially important ongoing legal battles", and "If JPM went bankrupted... how much could each shareholder get?". Questions that need a calculation, such as numerical and logical reasoning, also do badly, because their numbers sit on several pages and the question names none of them. Long 10-Ks are the hardest document type, since they have the most pages to search through.

This is the same wording-mismatch problem from Task #5. Dense retrieval (Tasks #4 and #5) is meant to fix exactly this kind of miss.

## Questions that need several pages

Kanishka pointed out that Hit@5 counts a question as found if any one gold page is in the top 5, even when the answer needs several pages. So we added all@5, which asks whether every gold page is in the top 5. For single-page questions, the two are the same.

Only 5 of our 48 questions have evidence on more than one page.

| Retriever | Filter | Hit@5 (any page) | all@5 (every page) |
|---|---|---|---|
| TF-IDF | off | 0.0 | 0.0 |
| TF-IDF | on | 0.2 | 0.0 |
| BM25 | off | 0.0 | 0.0 |
| BM25 | on | 0.4 | 0.0 |

No setup found all the pages for any of these 5 questions. BM25 with the filter found one page for 2 of them, which Hit@5 counts as a hit, but the model would still be missing numbers. Five questions is very few, but it backs up Kanishka's point that Hit@5 overstates search on multi-page questions. Details are in `results/task3/multi_page_any_vs_all.csv`.

## Rechecking the Task #5 chunking choice with BM25

Task #5 picked page chunks using TF-IDF only. Because chunk size and retriever affect each other, we reran all three chunk types with BM25 (`src/bm25_chunking_check.py`).

| Chunking | Filter | Hit@1 | Hit@5 | Text Hit@5 | MRR@10 | Run |
|---|---|---|---|---|---|---|
| Page | off | 0.188 | 0.333 | 0.354 | 0.243 | 25 |
| Page | on | 0.229 | 0.500 | 0.562 | 0.322 | 26 |
| 300 / 50 | off | 0.125 | 0.333 | 0.271 | 0.212 | 27 |
| 300 / 50 | on | 0.167 | 0.438 | 0.354 | 0.285 | 28 |
| 600 / 100 | off | 0.188 | 0.354 | 0.375 | 0.241 | 29 |
| 600 / 100 | on | 0.229 | 0.479 | 0.521 | 0.314 | 30 |

Page chunks are still the best, or tied for best, on every measure, so the Task #5 choice holds. Page and 600-word chunks are one question apart, which is inside the noise. 300-word chunks are a little worse.

## Limitations

- At 48 questions, one question is worth about 2 points. Several groups are tiny, with only 2 metrics-generated questions and 5 8-Ks, so treat those rows as hints, not results.
- For novel-generated questions the dataset gives no reasoning label, so the "none given" reasoning row is the same 27 questions.
- The corpus still includes the 4 downloaded test filings. Only their questions are left out, and extra pages make search harder, not easier.

## Files

- `src/task3_search_test.py`: the script.
- `results/task3/dataset_overview.csv`: the Part 1 counts.
- `results/task3/per_question.csv`: one row per question per setup, with hit or miss.
- `results/task3/hit5_by_group.csv`: Hit@5 by kind of question.
- `results/task3/multi_page_any_vs_all.csv`: any page vs all pages, for multi-page questions.
- `src/bm25_chunking_check.py`: the BM25 chunking recheck.
- `results/task9-experiment-log.csv`: runs 25 to 30 are BM25.
