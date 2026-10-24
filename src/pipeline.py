"""First end-to-end RAG pipeline (Issue #23).

One development question goes in; one answer in the team's response format
(docs/response_structure.md) comes out:

    question -> BM25 search with the company/year filter -> top-k pages
             -> prompt with numbered sources, trimmed to fit the model's context window
             -> LLM -> JSON reply -> answer with citations filled in from chunk metadata

The model cites sources by NUMBER only. The filing name, page, company and link in
each citation come from the chunk metadata, so the model can never invent a page.

Usage, from the repository root:
    python -m src.pipeline --id financebench_id_00705                # one question, fake LLM
    python -m src.pipeline --id financebench_id_00705 --llm ollama   # one question, real model
    python -m src.pipeline --all-dev --llm ollama                    # every dev question we have filings for

Only development questions are accepted. Frozen test IDs are refused.
Page numbers are 0-based, like the dataset: add 1 to find a page in a PDF viewer.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import date
from pathlib import Path

from src.data.split import load_rows
from src.eval_retrieval import allowed
from src.retrieval.bm25 import BM25Index, config_record
from src.retrieval.queries import load_dev_queries

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "financebench_merged.jsonl"
DEV_IDS = ROOT / "data" / "splits" / "dev_ids.txt"
CHUNKS = ROOT / "data" / "processed" / "chunks_page.jsonl"
DEFAULT_OUT = ROOT / "results" / "pipeline_dev_run.jsonl"

TOP_K = 5                    # pages retrieved for the model
DEFAULT_MODEL = "qwen3:8b"   # Ollama model tag
# Context window in tokens. At 8,192, qwen3:8b fits entirely on an 8 GB graphics card.
# When part of the model spills onto the CPU, Ollama on Windows can crash with
# "CUDA error: shared object initialization failed" (ollama/ollama#17380).
# Bigger cards can raise it with --num-ctx.
NUM_CTX = 8192
MAX_REPLY_TOKENS = 1024      # longest reply the model may write
RESERVED_TOKENS = 1500       # room kept for the instructions, the question and the reply
MAX_ATTEMPTS = 3             # tries per question before recording an error and moving on

SYSTEM_PROMPT = """You answer questions about company financial filings.
Use ONLY the numbered sources in the user's message. Never use outside knowledge.

Rules:
1. For every source you rely on, give its number and copy one short exact quote from it (a sentence or a table row).
2. If the question needs a calculation, write each step with the numbers you used in "calculation_steps".
3. Put the final answer in "answer", with units (for example "$1,577 million" or "24.5%").
4. If the sources do not contain what you need, set "abstained" to true, "answer" to "" and "citations" to [].

Reply with JSON only, in exactly this form:
{"citations": [{"source": 1, "quote": "..."}], "calculation_steps": ["..."], "answer": "...", "abstained": false}"""

# The JSON shape Ollama is forced to follow. Evidence comes first and the answer last,
# so the model quotes and calculates before it commits to an answer.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["source", "quote"],
            },
        },
        "calculation_steps": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
        "abstained": {"type": "boolean"},
    },
    "required": ["citations", "calculation_steps", "answer", "abstained"],
}


# ---------------------------------------------------------------- 1. data and search

def load_chunks(path: Path = CHUNKS) -> tuple[list[dict], dict[str, dict]]:
    """Read the chunk file. Returns the chunks and a chunk_id -> chunk lookup."""
    with path.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    return chunks, {c["chunk_id"]: c for c in chunks}


def build_index(chunks: list[dict]) -> BM25Index:
    """Build the team's BM25 index over every chunk (Issue #9 configuration)."""
    return BM25Index.build((c["chunk_id"], c["text"]) for c in chunks)


def retrieve(question: str, index: BM25Index, chunks: list[dict],
             by_id: dict[str, dict], k: int = TOP_K) -> list[tuple[dict, float]]:
    """Top-k (chunk, BM25 score) pairs, searching only the company/year the question names.

    BM25 ranks every chunk, then chunks outside the filter are skipped. This matches
    how src/eval_retrieval.py applies the same filter.
    """
    keep = allowed(question, chunks)
    allowed_ids = {c["chunk_id"] for c, ok in zip(chunks, keep) if ok}
    ranked = index.retrieve(question, k=index.n_docs)
    top = [r for r in ranked if r.chunk_id in allowed_ids][:k]
    return [(by_id[r.chunk_id], r.score) for r in top]


# ---------------------------------------------------------------- 2. prompt

def estimate_tokens(text: str) -> int:
    """A cautious guess at how many tokens a Qwen model will see.

    Qwen gives every digit its own token and financial tables are full of digits and
    symbols, so those count one each; ordinary text runs about 4 characters per token.
    """
    digits = sum(ch.isdigit() for ch in text)
    symbols = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    return digits + symbols + (len(text) - digits - symbols) // 4 + 1


def build_prompt(question: str, retrieved: list[tuple[dict, float]],
                 max_tokens: int = NUM_CTX - RESERVED_TOKENS) -> tuple[str, str, int]:
    """Return (system message, user message, number of sources sent).

    Sources are numbered [1]..[n], best first. If they would not fit in ``max_tokens``,
    the page that overflows is cut short and lower-ranked pages are left out, so the
    prompt never outgrows the model's context window.
    """
    sources, left = [], max_tokens
    for n, (chunk, _score) in enumerate(retrieved, start=1):
        source = f"[{n}] {chunk['doc_name']}, page {chunk['page_num']}\n{chunk['text']}"
        cost = estimate_tokens(source)
        if cost > left:
            if left < 150:  # too little room left for a useful excerpt
                break
            source = source[: len(source) * left // cost] + " [...]"
            left = 0
        else:
            left -= cost
        sources.append(source)
        if left == 0:
            break
    user = "Sources:\n\n" + "\n\n".join(sources) + f"\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user, len(sources)


# ---------------------------------------------------------------- 3. language models

class FakeLLM:
    """Stands in for a real model: always returns the same valid reply.

    Lets the whole pipeline run and be tested with no model installed.
    """

    name = "fake"
    num_ctx = NUM_CTX

    def generate(self, system: str, user: str) -> str:
        return json.dumps({
            "citations": [{"source": 1, "quote": "(fake quote)"}],
            "calculation_steps": [],
            "answer": "FAKE ANSWER - no model was called",
            "abstained": False,
        })


class OllamaLLM:
    """A local open model served by Ollama (https://ollama.com).

    No API key, and questions never leave the laptop. Temperature 0 and a fixed
    seed make runs repeatable. A failed call is retried, because Ollama's GPU engine
    occasionally crashes ("CUDA error: an illegal memory access") and restarts itself.
    """

    def __init__(self, model: str = DEFAULT_MODEL, num_ctx: int = NUM_CTX):
        import ollama  # imported here so the fake LLM and the tests never need it

        self.client = ollama.Client()
        self.name = model
        self.num_ctx = num_ctx
        self.options = {"temperature": 0, "seed": 42, "num_ctx": num_ctx,
                        "num_predict": MAX_REPLY_TOKENS}
        self.last_prompt_tokens = None
        self.last_calls = 0     # model calls the last question needed, retries included
        self.retry_wait = 10    # seconds to wait before retrying

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.last_calls = attempt
            try:
                response = self.client.chat(
                    model=self.name,
                    messages=messages,
                    format=REPLY_SCHEMA,  # forces valid JSON in exactly this shape
                    options=self.options,
                    think=False,          # qwen3's "thinking" text would come before the JSON
                )
            except Exception as error:
                if attempt == MAX_ATTEMPTS:
                    raise
                print(f"  model call failed ({str(error).splitlines()[0][:80]}); "
                      f"retrying in {self.retry_wait} s")
                time.sleep(self.retry_wait)
                continue
            if response.message.content or attempt == MAX_ATTEMPTS:
                break
            print(f"  empty reply; retrying in {self.retry_wait} s")
            time.sleep(self.retry_wait)

        self.last_prompt_tokens = response.prompt_eval_count
        if (self.last_prompt_tokens or 0) > self.num_ctx - MAX_REPLY_TOKENS:
            # Ollama cuts an over-long prompt without raising an error.
            print(f"  warning: the prompt used {self.last_prompt_tokens} of {self.num_ctx} tokens, "
                  "so it or the reply may have been cut")
        return response.message.content or ""


# ---------------------------------------------------------------- 4. reply -> team format

def parse_reply(raw: str) -> dict | None:
    """The model's reply as a dict, or None if it is not a JSON object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())  # drop ```json fences
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        return None
    return reply if isinstance(reply, dict) else None


def to_response(financebench_id: str, meta: dict, retrieved: list[tuple[dict, float]],
                reply: dict | None, raw: str, latency_ms: int, model_calls: int = 1,
                sent: int | None = None) -> dict:
    """Build one answer in the format of docs/response_structure.md.

    ``sent`` is how many of the retrieved pages were in the prompt (all, if None);
    a citation can only point at one of those. ``confidence`` and ``rerank_score``
    stay None: they belong to the abstention (#24) and reranking (#22) tasks. A reply
    that is not valid JSON gets an extra ``error`` field instead of crashing the run.
    """
    error = None
    if reply is None:
        error = f"model reply was not valid JSON: {raw[:300]!r}"
        reply = {}
    sent = len(retrieved) if sent is None else sent

    citations, used = [], set()
    for item in reply.get("citations") or []:
        number = item.get("source") if isinstance(item, dict) else None
        # Keep only numbers that point at a source we actually sent.
        if isinstance(number, bool) or not isinstance(number, int):
            continue
        if not 1 <= number <= sent or number in used:
            continue
        used.add(number)
        chunk, score = retrieved[number - 1]
        citations.append({
            "doc_name": chunk["doc_name"],
            "company": chunk["company"],
            "doc_type": chunk["doc_type"],
            "doc_period": chunk["doc_period"],
            "evidence_page_num": chunk["page_num"],
            "evidence_text": str(item.get("quote", "")),
            "doc_link": chunk["doc_link"],
            "retrieval_score": round(score, 4),
            "rerank_score": None,
        })

    steps = [str(s) for s in (reply.get("calculation_steps") or []) if str(s).strip()]
    answer = reply.get("answer")
    abstained = reply.get("abstained") is True
    response = {
        "financebench_id": financebench_id,
        "question_type": meta.get("question_type"),
        "question_reasoning": meta.get("question_reasoning"),
        "answer": None if abstained or not answer else str(answer),
        "confidence": None,
        "abstained": abstained,
        "calculation": {"shown": bool(steps), "steps": steps},
        "citations": citations,
        "retrieved_but_unused": [
            f"{chunk['doc_name']} p{chunk['page_num']}"
            for n, (chunk, _score) in enumerate(retrieved, start=1)
            if n not in used
        ],
        "latency_ms": latency_ms,
        "model_calls": model_calls,
    }
    if error:
        response["error"] = error
    return response


# ---------------------------------------------------------------- 5. one question, end to end

def answer_question(financebench_id: str, question: str, meta: dict, index: BM25Index,
                    chunks: list[dict], by_id: dict[str, dict], llm, k: int = TOP_K) -> dict:
    """Run one question through search, prompt, model and formatting."""
    start = time.perf_counter()
    failure = None
    retrieved = retrieve(question, index, chunks, by_id, k)
    if not retrieved:  # nothing matched: don't ask the model to answer from nothing
        reply, raw, calls, sent = {"abstained": True}, "", 0, 0
    else:
        system, user, sent = build_prompt(question, retrieved, llm.num_ctx - RESERVED_TOKENS)
        try:
            raw = llm.generate(system, user)
            reply = parse_reply(raw)
        except Exception as error:  # the model kept failing: record it and move on
            raw, reply = "", None
            failure = f"model call failed: {str(error)[:300]}"
        calls = getattr(llm, "last_calls", 1)
    latency_ms = round((time.perf_counter() - start) * 1000)
    response = to_response(financebench_id, meta, retrieved, reply, raw, latency_ms, calls, sent)
    if failure:
        response["error"] = failure
    return response


# ---------------------------------------------------------------- 6. command line

def git_commit() -> str:
    """Current commit, marked "+dirty" if the code has uncommitted changes (as in eval_retrieval.py)."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True, cwd=ROOT).stdout.strip()
        dirty = git("status", "--porcelain") != ""
        return git("rev-parse", "--short", "HEAD") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer development questions with the RAG pipeline.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--id", help="one development financebench_id")
    which.add_argument("--all-dev", action="store_true",
                       help="every development question whose filing is downloaded")
    parser.add_argument("--llm", choices=["fake", "ollama"], default="fake")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--k", type=int, default=TOP_K, help="pages retrieved for the model")
    parser.add_argument("--num-ctx", type=int, default=NUM_CTX,
                        help="model context window in tokens (8192 fits qwen3:8b on an 8 GB card)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results file for --all-dev")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.num_ctx < 2 * RESERVED_TOKENS:
        parser.error(f"--num-ctx must be at least {2 * RESERVED_TOKENS}")

    # Questions come only from the dev split. The test ID file is never opened.
    dev = {q.financebench_id: q.question for q in load_dev_queries(DATASET, DEV_IDS)}
    if args.id and args.id not in dev:
        parser.error(f"{args.id} is not a development question; test questions are never run here")

    # Question type and filing name only. Answers and evidence are never read.
    meta = {
        row["financebench_id"]: {
            "question_type": row["question_type"],
            "question_reasoning": row["question_reasoning"],
            "doc_name": row["doc_name"],
        }
        for row in load_rows(DATASET)
        if row["financebench_id"] in dev
    }

    if not CHUNKS.exists():
        parser.error(f"{CHUNKS} not found; run download_pdfs.py, extract_text.py and chunk.py first")
    chunks, by_id = load_chunks()
    have = {c["doc_name"] for c in chunks}
    ids = [args.id] if args.id else [i for i in sorted(dev) if meta[i]["doc_name"] in have]
    if args.id and meta[args.id]["doc_name"] not in have:
        parser.error(f"the filing for {args.id} ({meta[args.id]['doc_name']}) is not downloaded")

    index = build_index(chunks)
    llm = FakeLLM() if args.llm == "fake" else OllamaLLM(args.model, args.num_ctx)

    if args.id:
        result = answer_question(args.id, dev[args.id], meta[args.id], index, chunks, by_id, llm, args.k)
        print(json.dumps(result, indent=2))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for n, fid in enumerate(ids, start=1):
            result = answer_question(fid, dev[fid], meta[fid], index, chunks, by_id, llm, args.k)
            f.write(json.dumps(result) + "\n")
            status = "ERROR" if "error" in result else "abstained" if result["abstained"] else "answered"
            tokens = getattr(llm, "last_prompt_tokens", None)
            print(f"[{n}/{len(ids)}] {fid}  {status}  citations={len(result['citations'])}  "
                  f"{result['latency_ms']} ms" + (f"  prompt_tokens={tokens}" if tokens else ""))

    run_record = {
        "date": date.today().isoformat(),
        "commit": git_commit(),
        "llm": args.llm,
        "model": llm.name,
        "options": getattr(llm, "options", None),
        "top_k": args.k,
        "chunks": CHUNKS.name,
        "retriever": "bm25 + company/year filter",
        "bm25_config": config_record(),
        "questions": len(ids),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(ids)} answers to {args.out}")


if __name__ == "__main__":
    main()
