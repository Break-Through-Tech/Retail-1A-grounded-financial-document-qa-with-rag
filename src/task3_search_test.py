"""October Task #3: explore the dev questions, then run a search test.

Part 1 describes the 111 development questions.
Part 2 runs TF-IDF and BM25 over page chunks for the dev questions whose filing
we have downloaded, then breaks Hit@5 down by kind of question, to see which
questions search does well and badly on.

The frozen test questions are never read.

Usage: python -m src.task3_search_test     (from the repository root)
"""

import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.eval_retrieval import (
    COVERAGE, PROC, ROOT, SPLITS, TOP_K,
    allowed, ceiling, dev_questions, log_run, next_run_number, shell, tokens,
)
from src.evaluation.metrics import hit_at_k, reciprocal_rank_at_k
from src.retrieval.bm25 import BM25Index

OUT = ROOT / "results" / "task3"
CHUNKS = "chunks_page.jsonl"
GROUPS = ["question_type", "reasoning", "doc_type"]


def short_reasoning(label):
    """'Numerical reasoning OR Logical reasoning' -> 'Numerical reasoning'."""
    return label.split(" OR ")[0] if label else "none given"


# ---------- Part 1: explore ----------

def explore():
    dev = set((SPLITS / "dev_ids.txt").read_text().split())
    df = pd.read_json(ROOT / "data" / "financebench_merged.jsonl", lines=True)
    df = df[df.financebench_id.isin(dev)].copy()
    df["reasoning"] = df.question_reasoning.map(short_reasoning)
    df["passages"] = df.evidence.map(len)
    words = df.evidence.map(lambda ev: sum(len(e["evidence_text"].split()) for e in ev))

    rows = [
        {"group": "total", "value": "questions", "questions": len(df)},
        {"group": "total", "value": "companies", "questions": df.company.nunique()},
        {"group": "total", "value": "filings", "questions": df.doc_name.nunique()},
        {"group": "evidence words", "value": "median", "questions": int(words.median())},
        {"group": "evidence words", "value": "max", "questions": int(words.max())},
    ]
    for col in GROUPS + ["passages"]:
        for value, n in df[col].value_counts().items():
            rows.append({"group": col, "value": value, "questions": n})

    overview = pd.DataFrame(rows)
    overview.to_csv(OUT / "dataset_overview.csv", index=False)
    print(overview.to_string(index=False))


# ---------- Part 2: search test ----------

def tfidf_ranker(chunks):
    """Same TF-IDF settings as src/eval_retrieval.py."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vec.fit_transform(c["text"] for c in chunks)

    def rank(question):
        sims = cosine_similarity(vec.transform([question]), matrix)[0]
        return np.argsort(-sims, kind="stable").tolist()
    return rank


def bm25_ranker(chunks):
    """The team's BM25 (src/retrieval/bm25.py), ranking every chunk."""
    index = BM25Index.build((c["chunk_id"], c["text"]) for c in chunks)
    position = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    def rank(question):
        return [position[r.chunk_id] for r in index.retrieve(question, len(chunks))]
    return rank


def search_test():
    chunks = [json.loads(l) for l in (PROC / CHUNKS).open()]
    chunk_tokens = [tokens(c["text"]) for c in chunks]
    questions, n_filings = dev_questions()
    rankers = {"tfidf": tfidf_ranker(chunks), "bm25": bm25_ranker(chunks)}

    rows = []
    for name, rank in rankers.items():
        for use_filter in [False, True]:
            for q in questions:
                order = rank(q["question"])
                if use_filter:
                    keep = allowed(q["question"], chunks)
                    order = [i for i in order if keep[i]]
                top = order[:TOP_K]

                # the shape Kanishka's metrics expect
                results = [{"doc_name": chunks[i]["doc_name"],
                            "evidence_page_num": chunks[i]["page_num"]} for i in top]
                gold = [tokens(e["evidence_text"]) for e in q["evidence"]]
                text_hit = any(len(g & chunk_tokens[i]) / len(g) >= COVERAGE
                               for g in gold if g for i in top[:5])

                rows.append({
                    "financebench_id": q["financebench_id"],
                    "retriever": name,
                    "filter": "on" if use_filter else "off",
                    "question_type": q["question_type"],
                    "reasoning": short_reasoning(q["question_reasoning"]),
                    "doc_type": q["doc_type"],
                    "doc_name": q["doc_name"],
                    "hit@1": hit_at_k(results, q["evidence"], 1),
                    "hit@5": hit_at_k(results, q["evidence"], 5),
                    "thit@5": int(text_hit),
                    "mrr@10": round(reciprocal_rank_at_k(results, q["evidence"], TOP_K), 3),
                    "question": q["question"],
                })

    per_q = pd.DataFrame(rows)
    per_q.to_csv(OUT / "per_question.csv", index=False)

    # overall score for each of the four setups
    metrics = ["hit@1", "hit@5", "thit@5", "mrr@10"]
    overall = per_q.groupby(["retriever", "filter"], sort=False)[metrics].mean().round(3)
    print(f"\n{len(questions)} dev questions over {n_filings} filings\n")
    print(overall.to_string())

    # Hit@5 by kind of question: the "does well / does badly" table
    one_setup = per_q[(per_q.retriever == "tfidf") & (per_q["filter"] == "off")]
    tables = []
    for col in GROUPS:
        t = per_q.pivot_table(index=col, columns=["retriever", "filter"],
                              values="hit@5", aggfunc="mean", sort=False).round(3)
        t.columns = [f"{r} filter {f}" for r, f in t.columns]
        t.insert(0, "questions", one_setup.groupby(col).size())
        t.insert(0, "group", col)
        tables.append(t.rename_axis("value").reset_index()[["group", "value", *t.columns[1:]]])
    by_group = pd.concat(tables)
    by_group.to_csv(OUT / "hit5_by_group.csv", index=False)
    print("\nHit@5 by kind of question\n")
    print(by_group.to_string(index=False))

    log_bm25_runs(overall, len(questions), n_filings, ceiling(CHUNKS, questions))


def log_bm25_runs(overall, n_questions, n_filings, cap):
    """Add the BM25 runs to the Task #9 log (TF-IDF is already runs 19-20)."""
    run = next_run_number()
    who = shell(["git", "config", "user.name"])
    commit = shell(["git", "rev-parse", "--short", "HEAD"])
    dirty = shell(["git", "status", "--porcelain"]) != ""
    for use_filter in ["off", "on"]:
        log_run({
            "run": run, "date": date.today().isoformat(), "who": who,
            "commit": commit + ("+dirty" if dirty else ""),
            "split": "dev", "questions": n_questions, "filings": n_filings,
            "chunking": "page", "size": "1 page", "overlap": "none",
            "header": "removed", "filter": use_filter,
            "retriever": "bm25", "top_k": TOP_K,
            **overall.loc[("bm25", use_filter)].to_dict(), "ceiling": cap,
            "note": "task3 search test; development questions only",
        })
        run += 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    explore()
    search_test()


if __name__ == "__main__":
    main()
