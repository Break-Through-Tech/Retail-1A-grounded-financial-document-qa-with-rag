"""Check a pipeline run against the development answer key

Reads the file written by ``python -m src.pipeline --all-dev`` and reports:
  answered / abstained / errors
  gold page retrieved   search put a gold evidence page among the pages sent to the model
  gold page cited       the answer cites a gold evidence page
  quotes on their page  each citation's quote really appears on the page it cites
Then prints model answers next to gold answers so you can judge them by hand.

Only development IDs are scored; anything else in the file is refused.

Usage: python -m src.check_pipeline_run [--run results/pipeline_dev_run.jsonl] [--show 10]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.data.split import load_rows
from src.evaluation.metrics import is_gold_match
from src.pipeline import CHUNKS, DATASET, DEFAULT_OUT, DEV_IDS, load_chunks
from src.retrieval.queries import load_dev_queries


def squash(text: str) -> str:
    """Lowercase and collapse whitespace, so line breaks in the PDF text don't matter."""
    return re.sub(r"\s+", " ", text).strip().lower()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a pipeline run on development questions.")
    parser.add_argument("--run", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--show", type=int, default=10, help="answers to print for reading by hand")
    args = parser.parse_args(argv)

    results = [json.loads(l) for l in args.run.open(encoding="utf-8") if l.strip()]
    dev = {q.financebench_id for q in load_dev_queries(DATASET, DEV_IDS)}
    outside = [r["financebench_id"] for r in results if r["financebench_id"] not in dev]
    if outside:
        parser.error(f"{len(outside)} results are not development questions, e.g. {outside[0]}")
    gold = {row["financebench_id"]: row for row in load_rows(DATASET) if row["financebench_id"] in dev}
    _, by_id = load_chunks(CHUNKS)
    page_text = {(c["doc_name"], c["page_num"]): squash(c["text"]) for c in by_id.values()}

    counts = Counter()
    for r in results:
        evidence = gold[r["financebench_id"]]["evidence"]
        retrieved = [{"doc_name": c["doc_name"], "evidence_page_num": c["evidence_page_num"]}
                     for c in r["citations"]]
        for item in r["retrieved_but_unused"]:  # "DOC_NAME p47"
            doc, page = item.rsplit(" p", 1)
            retrieved.append({"doc_name": doc, "evidence_page_num": int(page)})

        counts["error" if "error" in r else "abstained" if r["abstained"] else "answered"] += 1
        counts["gold page retrieved"] += any(is_gold_match(x, evidence) for x in retrieved)
        counts["gold page cited"] += any(is_gold_match(c, evidence) for c in r["citations"])
        for c in r["citations"]:
            counts["citations"] += 1
            text = page_text.get((c["doc_name"], c["evidence_page_num"]), "")
            counts["quotes on their page"] += bool(c["evidence_text"].strip()) and squash(c["evidence_text"]) in text

    n = len(results)
    print(f"{n} development questions in {args.run.name}\n")
    for key in ["answered", "abstained", "error", "gold page retrieved", "gold page cited"]:
        print(f"  {key:<22}{counts[key]:>4} / {n}  ({counts[key] / n:.0%})")
    if counts["citations"]:
        share = counts["quotes on their page"] / counts["citations"]
        print(f"  {'quotes on their page':<22}{counts['quotes on their page']:>4} / {counts['citations']}  ({share:.0%})")

    print(f"\nFirst {min(args.show, n)} answers next to the answer key:")
    for r in results[: args.show]:
        row = gold[r["financebench_id"]]
        print(f"\n{r['financebench_id']} ({row['question_type']})")
        print(f"  Q:     {row['question']}")
        print(f"  model: {r['answer']}")
        print(f"  gold:  {row['answer']}")


if __name__ == "__main__":
    main()