"""Score a chunking strategy with a TF-IDF retriever.

Only questions whose filing we have downloaded are scored.

Metrics (a question counts as a hit if any of its gold passages is found):
  hit@1, hit@5   page rule: retrieved chunk is on the gold evidence page
  thit@5         text rule: chunk contains >= 80% of the evidence's tokens
  mrr@10         1/rank of the first correct chunk

Usage: python src/eval_retrieval.py
"""

import csv
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
LOG = ROOT / "results" / "task9-experiment-log.csv"
COVERAGE = 0.8
TOP_K = 10

# One row per pipeline run, in four groups:
# what the run was | what data it ran on | settings | results
COLUMNS = [
    "run", "date", "who", "commit",
    "split", "questions", "filings",
    "chunking", "size", "overlap", "header", "filter", "retriever", "top_k",
    "hit@1", "hit@5", "thit@5", "mrr@10", "ceiling",
    "note",
]


def tokens(text):
    return set(re.findall(r"[a-z0-9][a-z0-9,.]*", text.lower()))


def allowed(question, chunks):
    """Mask of chunks whose company (and year, if named) matches the question."""
    q = question.lower()
    companies = {c["company"] for c in chunks
                 if c["company"].lower() in q or c["company"].split()[0].lower() in q}
    years = {int(y) for y in re.findall(r"20\d\d", q)}

    keep = [
        (not companies or c["company"] in companies)
        and (not years or c["doc_period"] in years)
        for c in chunks
    ]
    return keep if any(keep) else [True] * len(chunks)


def ceiling(chunk_file, questions):
    """Chunking on its own: does ANY chunk hold a whole gold passage?

    No searching involved. If a question fails here, the boundaries cut its
    evidence apart and no retriever could ever recover it.
    """
    chunks = [json.loads(l) for l in (PROC / chunk_file).open()]
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_name"], []).append(tokens(c["text"]))

    found = 0
    for q in questions:
        gold = [tokens(e["evidence_text"]) for e in q["evidence"]]
        cand = by_doc.get(q["doc_name"], [])
        if any(len(g & ct) / len(g) >= COVERAGE for g in gold if g for ct in cand):
            found += 1
    return round(found / len(questions), 3)


def evaluate(chunk_file, questions, use_metadata=False):
    chunks = [json.loads(l) for l in (PROC / chunk_file).open()]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vec.fit_transform(c["text"] for c in chunks)
    chunk_tokens = [tokens(c["text"]) for c in chunks]

    scores = {"hit@1": 0, "hit@5": 0, "thit@5": 0, "mrr@10": 0.0}
    for q in questions:
        sims = cosine_similarity(vec.transform([q["question"]]), matrix)[0]
        if use_metadata:
            keep = allowed(q["question"], chunks)
            sims = [s if k else -1 for s, k in zip(sims, keep)]
        top = pd.Series(sims).nlargest(10).index.tolist()

        gold_pages = {(e["doc_name"], e["evidence_page_num"]) for e in q["evidence"]}
        gold_tokens = [tokens(e["evidence_text"]) for e in q["evidence"]]

        page_ok, text_ok, rank = [], [], None
        for pos, i in enumerate(top, start=1):
            c = chunks[i]
            on_page = (c["doc_name"], c["page_num"]) in gold_pages
            covers = any(
                len(g & chunk_tokens[i]) / len(g) >= COVERAGE for g in gold_tokens if g
            )
            page_ok.append(on_page)
            text_ok.append(covers)
            if on_page and rank is None:
                rank = pos

        scores["hit@1"] += page_ok[0]
        scores["hit@5"] += any(page_ok[:5])
        scores["thit@5"] += any(text_ok[:5])
        scores["mrr@10"] += 1 / rank if rank else 0.0

    n = len(questions)
    return {k: round(v / n, 3) for k, v in scores.items()}


def shell(cmd, default="unknown"):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return default


def next_run_number():
    if not LOG.exists():
        return 1
    with LOG.open() as f:
        return sum(1 for _ in csv.DictReader(f)) + 1


def log_run(row):
    """Append one run to the log. Existing rows are never touched."""
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


SETTINGS = [
    # tag, chunking strategy, size, overlap
    ("page", "page", "1 page", "none"),
    ("w300", "words", 300, 50),
    ("w600", "words", 600, 100),
]
NOTE = "development questions only; frozen test IDs excluded"


def dev_questions():
    """Dev-split questions whose filing we have downloaded.

    The frozen test IDs are never scored here.
    """
    dev = set((SPLITS / "dev_ids.txt").read_text().split())
    have = {p.stem for p in (ROOT / "data" / "pdfs").glob("*.pdf")}
    df = pd.read_json(ROOT / "data" / "financebench_merged.jsonl", lines=True)
    keep = df[df.financebench_id.isin(dev) & df.doc_name.isin(have)]
    return keep.to_dict("records"), len(set(keep.doc_name))


def main():
    questions, n_filings = dev_questions()
    print(f"{len(questions)} dev questions over {n_filings} filings")

    run = next_run_number()
    who = shell(["git", "config", "user.name"])
    commit = shell(["git", "rev-parse", "--short", "HEAD"])
    dirty = shell(["git", "status", "--porcelain"]) != ""

    for tag, strategy, size, overlap in SETTINGS:
        cap = ceiling(f"chunks_{tag}.jsonl", questions)
        print(f"\n{tag}: evidence kept whole in some chunk = {cap}")
        for use_filter in [False, True]:
            scores = evaluate(f"chunks_{tag}.jsonl", questions, use_filter)
            log_run({
                "run": run, "date": date.today().isoformat(), "who": who,
                "commit": commit + ("+dirty" if dirty else ""),
                "split": "dev", "questions": len(questions), "filings": n_filings,
                "chunking": strategy, "size": size, "overlap": overlap,
                "header": "removed", "filter": "on" if use_filter else "off",
                "retriever": "tfidf", "top_k": TOP_K,
                **scores, "ceiling": cap, "note": NOTE,
            })
            print(f"  run {run}: {tag}, filter={'on' if use_filter else 'off'}", scores)
            run += 1


if __name__ == "__main__":
    main()
