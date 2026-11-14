"""Task 10, search stage: compare passage sizes and search types on the dev questions.

For each setting it builds an index over the dev chunks, searches with every dev question
(the question text only), scores the ranking against the gold evidence page and appends one
row to results/task10-comparison.csv.

Metrics (page rule, same as src/evaluation/metrics.py):
  hit@1, hit@5   a gold evidence page is among the top 1 / top 5 results
  mrr@10         1 / rank of the first gold page within the top 10

Usage:
  python -m src.task10_compare --chunks page --retriever bm25
  python -m src.task10_compare --all-bm25                       # 3 passage sizes, BM25
  python -m src.task10_compare --chunks page --retriever hybrid --model ID --revision HASH
                                                                # BM25 + dense fused, no reranker
  python -m src.task10_compare --chunks page --retriever dense  --model ID --revision HASH
  python -m src.task10_compare --chunks page --retriever rerank --model ID --revision HASH \
        --rerank-model ID --rerank-revision HASH   (rerank = BM25 + dense fused, then reranked)

The company/year filter is applied to the full ranking before the top results are taken.

Needs data/processed/chunks_*.jsonl (python src/chunk.py). Only dev questions are used.
"""

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import date
from pathlib import Path

from src.evaluation.metrics import evaluate_rankings
from src.retrieval.bm25 import BM25Index
from src.retrieval.queries import load_dev_queries

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DATASET = ROOT / "data" / "financebench_merged.jsonl"
LOG = ROOT / "results" / "task10-comparison.csv"

CHUNK_FILES = {
    "page": ("chunks_page.jsonl", "page", "1 page", "none"),
    "w300": ("chunks_w300.jsonl", "words", "300", "50"),
    "w600": ("chunks_w600.jsonl", "words", "600", "100"),
}

COLUMNS = [
    "run", "stage", "status", "date", "who", "commit", "split", "questions", "filings",
    "chunking", "size", "overlap", "filter", "retriever", "top_k", "model", "prompt",
    "hit@1", "hit@5", "mrr@10", "gold_page_retrieved", "gold_page_cited", "quotes_on_page",
    "answered", "abstained", "correct_by_hand", "partly_correct_by_hand",
    "wrong_or_abstained_by_hand", "latency_s", "note",
]


def allowed(question, chunks):
    """Indexes of chunks whose company (and year, if named) fits the question.

    Same rule as src/eval_retrieval.py: if nothing fits, search everything.
    """
    q = question.lower()
    companies = {c["company"] for c in chunks
                 if c["company"].lower() in q or c["company"].split()[0].lower() in q}
    years = {int(y) for y in re.findall(r"20\d\d", q)}
    keep = [
        i for i, c in enumerate(chunks)
        if (not companies or c["company"] in companies)
        and (not years or c["doc_period"] in years)
    ]
    return keep or list(range(len(chunks)))


def load_gold():
    gold = {}
    for line in DATASET.open():
        r = json.loads(line)
        gold[r["financebench_id"]] = [
            {"doc_name": e["doc_name"],
                "evidence_page_num": e["evidence_page_num"]}
            for e in r["evidence"]
        ]
    return gold


class Filtered:
    """Wraps a retriever so it only returns chunks the company/year filter allows.

    `keep` (a set of chunk_ids) is set per question before searching. The wrapped retriever
    ranks every chunk and the disallowed ones are skipped, as in src/pipeline.py.
    """

    def __init__(self, inner, n_chunks):
        self.inner = inner
        self.n = n_chunks
        self.keep = None

    def retrieve(self, query, k):
        hits = self.inner.retrieve(query, self.n)
        if self.keep is not None:
            hits = [h for h in hits if h.chunk_id in self.keep]
        return hits[:k]


class Hybrid:
    """BM25 and dense results fused with reciprocal rank fusion (no reranker)."""

    def __init__(self, parts, depth=100):
        self.parts = parts
        self.depth = depth

    def retrieve(self, query, k):
        from src.rerank import rrf_fuse
        lists = [list(p.retrieve(query, self.depth)) for p in self.parts]
        return rrf_fuse([l for l in lists if l])[:k]


def build_retriever(kind, chunks, args):
    """Return (retriever, info, filters). `filters` must get `.keep` set for each question."""
    docs = [(c["chunk_id"], c["text"]) for c in chunks]
    n = len(docs)
    if kind == "bm25":
        f = Filtered(BM25Index.build(docs), n)
        return f, {"model": "none"}, [f]

    from src.retrieval.dense import DenseRetriever
    from src.retrieval.embeddings import SentenceTransformerEncoder

    if not (args.model and args.revision):
        raise SystemExit(
            "dense, hybrid and rerank need --model and --revision (a 40-character hash)")
    encoder = SentenceTransformerEncoder.load(
        args.model, args.revision, device=args.device, query_prompt=args.query_prompt)
    dense = Filtered(DenseRetriever.build(docs, encoder), n)
    if kind == "dense":
        return dense, {"model": args.model}, [dense]

    bm25 = Filtered(BM25Index.build(docs), n)
    if kind == "hybrid":
        return Hybrid([bm25, dense], depth=args.candidate_k), \
            {"model": f"{args.model} + bm25 (RRF)"}, [bm25, dense]

    from src.rerank import CrossEncoderScorer, RetrieveAndRerank
    if not (args.rerank_model and args.rerank_revision):
        raise SystemExit("rerank needs --rerank-model and --rerank-revision")
    scorer = CrossEncoderScorer.load(
        args.rerank_model, args.rerank_revision, device=args.device)
    pipe = RetrieveAndRerank({"bm25": bm25, "dense": dense}, dict(docs), scorer,
                             candidate_k=args.candidate_k)
    return pipe, {"model": f"{args.model} + {args.rerank_model}"}, [bm25, dense]


def search(kind, retriever, question, k, keep_ids):
    """Return chunk_ids, best first. The wrappers already apply the filter; this re-checks it."""
    if kind == "rerank":
        out = retriever.search(question, top_k=k)
        return [r.chunk_id for r in out.results if r.chunk_id in keep_ids][:k]
    hits = retriever.retrieve(question, k)
    return [h.chunk_id for h in hits if h.chunk_id in keep_ids][:k]


def run_one(chunk_key, kind, args, number):
    fname, chunking, size, overlap = CHUNK_FILES[chunk_key]
    chunks = [json.loads(l) for l in (PROC / fname).open()]
    by_id = {c["chunk_id"]: c for c in chunks}
    queries = load_dev_queries()
    gold = load_gold()

    t0 = time.time()
    retriever, info, filters = build_retriever(kind, chunks, args)
    rankings = {}
    for q in queries:
        keep = allowed(q.question, chunks) if args.filter == "on" else range(
            len(chunks))
        keep_ids = {chunks[i]["chunk_id"] for i in keep}
        for f in filters:
            f.keep = keep_ids
        ids = search(kind, retriever, q.question, 10, keep_ids)
        rankings[q.financebench_id] = [
            {"doc_name": by_id[i]["doc_name"],
                "evidence_page_num": by_id[i]["page_num"]}
            for i in ids
        ]
    elapsed = time.time() - t0

    scores = evaluate_rankings(rankings, {q.financebench_id: gold[q.financebench_id]
                                          for q in queries})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=ROOT, text=True).strip()
    except Exception:
        commit = ""
    row = {
        "run": number, "stage": args.stage, "status": "done", "date": date.today().isoformat(),
        "who": args.who, "commit": commit, "split": "dev", "questions": len(queries),
        "filings": len({c["doc_name"] for c in chunks}),
        "chunking": chunking, "size": size, "overlap": overlap, "filter": args.filter,
        "retriever": kind, "top_k": 10, "model": info["model"], "prompt": "n/a",
        "hit@1": round(scores["hit@1"], 3), "hit@5": round(scores["hit@5"], 3),
        "mrr@10": round(scores["mrr@10"], 3), "latency_s": round(elapsed, 1),
        "note": args.note,
    }
    return row


def append(row):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})


def next_run():
    if not LOG.exists():
        return 1
    with LOG.open() as f:
        return sum(1 for _ in csv.DictReader(f)) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", choices=list(CHUNK_FILES), default="page")
    p.add_argument(
        "--retriever", choices=["bm25", "dense", "hybrid", "rerank"], default="bm25")
    p.add_argument("--all-bm25", action="store_true",
                   help="run BM25 on all three passage sizes")
    p.add_argument("--filter", choices=["on", "off"], default="on")
    p.add_argument("--stage", default="A search only")
    p.add_argument("--who", default="Benita")
    p.add_argument("--note", default="")
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--rerank-model")
    p.add_argument("--rerank-revision")
    p.add_argument("--query-prompt")
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    plan = [(k, "bm25") for k in CHUNK_FILES] if args.all_bm25 else [
        (args.chunks, args.retriever)]
    for chunk_key, kind in plan:
        row = run_one(chunk_key, kind, args, next_run())
        append(row)
        print(f"{chunk_key:5s} {kind:6s} hit@1={row['hit@1']} hit@5={row['hit@5']} "
              f"mrr@10={row['mrr@10']}  ({row['latency_s']}s)")


if __name__ == "__main__":
    main()
