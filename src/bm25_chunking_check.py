"""Recheck the Task #5 chunking choice with BM25 instead of TF-IDF.

Task #5 compared page, 300-word and 600-word chunks using TF-IDF only.
Chunk size and retriever affect each other, so this reruns the 300- and
600-word chunks with BM25. Page chunks with BM25 are already runs 25-26.

Usage: python -m src.bm25_chunking_check     (from the repository root)
"""

import json
from datetime import date

from src.eval_retrieval import (
    COVERAGE, PROC, TOP_K,
    allowed, ceiling, dev_questions, log_run, next_run_number, shell, tokens,
)
from src.evaluation.metrics import hit_at_k, reciprocal_rank_at_k
from src.task3_search_test import bm25_ranker

SETTINGS = [
    # tag, size, overlap
    ("w300", 300, 50),
    ("w600", 600, 100),
]


def score(chunks, rank, questions, use_filter):
    chunk_tokens = [tokens(c["text"]) for c in chunks]
    totals = {"hit@1": 0, "hit@5": 0, "thit@5": 0, "mrr@10": 0.0}
    for q in questions:
        order = rank(q["question"])
        if use_filter:
            keep = allowed(q["question"], chunks)
            order = [i for i in order if keep[i]]
        top = order[:TOP_K]

        results = [{"doc_name": chunks[i]["doc_name"],
                    "evidence_page_num": chunks[i]["page_num"]} for i in top]
        gold = [tokens(e["evidence_text"]) for e in q["evidence"]]
        totals["hit@1"] += hit_at_k(results, q["evidence"], 1)
        totals["hit@5"] += hit_at_k(results, q["evidence"], 5)
        totals["thit@5"] += any(len(g & chunk_tokens[i]) / len(g) >= COVERAGE
                                for g in gold if g for i in top[:5])
        totals["mrr@10"] += reciprocal_rank_at_k(results, q["evidence"], TOP_K)
    return {k: round(v / len(questions), 3) for k, v in totals.items()}


def main():
    questions, n_filings = dev_questions()
    run = next_run_number()
    who = shell(["git", "config", "user.name"])
    commit = shell(["git", "rev-parse", "--short", "HEAD"])
    dirty = shell(["git", "status", "--porcelain", "--untracked-files=no"]) != ""

    for tag, size, overlap in SETTINGS:
        chunks = [json.loads(l) for l in (PROC / f"chunks_{tag}.jsonl").open()]
        rank = bm25_ranker(chunks)
        cap = ceiling(f"chunks_{tag}.jsonl", questions)
        for use_filter in [False, True]:
            scores = score(chunks, rank, questions, use_filter)
            log_run({
                "run": run, "date": date.today().isoformat(), "who": who,
                "commit": commit + ("+dirty" if dirty else ""),
                "split": "dev", "questions": len(questions), "filings": n_filings,
                "chunking": "words", "size": size, "overlap": overlap,
                "header": "removed", "filter": "on" if use_filter else "off",
                "retriever": "bm25", "top_k": TOP_K,
                **scores, "ceiling": cap,
                "note": "bm25 chunking recheck; dev questions; dev-filings-only corpus",
            })
            print(f"run {run}: {tag}, filter={'on' if use_filter else 'off'}", scores)
            run += 1


if __name__ == "__main__":
    main()
