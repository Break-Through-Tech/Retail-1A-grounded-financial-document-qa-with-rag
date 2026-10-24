"""Tests for the first end-to-end RAG pipeline (Issue #23).

Every test uses tiny made-up chunks and the fake LLM: no PDFs, no model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.pipeline import (
    FakeLLM,
    OllamaLLM,
    answer_question,
    build_index,
    build_prompt,
    estimate_tokens,
    main,
    parse_reply,
    retrieve,
    to_response,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "docs" / "response_structure.md").read_text(encoding="utf-8"))
META = {"question_type": "metrics-generated", "question_reasoning": "Information extraction"}


def make_chunk(doc: str, company: str, year: int, page: int, text: str) -> dict:
    return {
        "chunk_id": f"{doc}_p{page}_0",
        "doc_name": doc,
        "page_num": page,
        "section": "",
        "text": text,
        "company": company,
        "doc_type": "10k",
        "doc_period": year,
        "doc_link": f"https://example.com/{doc}.pdf",
    }


CHUNKS = [
    make_chunk("ACME_2022_10K", "Acme", 2022, 10, "Acme capital expenditure was $120 million in FY2022."),
    make_chunk("ACME_2022_10K", "Acme", 2022, 11, "Acme revenue was $900 million in FY2022."),
    make_chunk("ACME_2022_10K", "Acme", 2022, 12, "Acme research and development expense was $90 million."),
    make_chunk("GLOBEX_2022_10K", "Globex", 2022, 5, "Globex capital expenditure was $75 million in FY2022."),
]
BY_ID = {c["chunk_id"]: c for c in CHUNKS}
INDEX = build_index(CHUNKS)
QUESTION = "What was Acme's FY2022 capital expenditure?"


def acme_sources() -> list[tuple[dict, float]]:
    return retrieve(QUESTION, INDEX, CHUNKS, BY_ID, k=3)


def test_retrieve_searches_only_the_company_named():
    retrieved = acme_sources()
    assert retrieved, "expected at least one Acme page"
    assert all(chunk["company"] == "Acme" for chunk, _ in retrieved)
    assert retrieved[0][0]["page_num"] == 10  # the capex page ranks first


def test_retrieve_respects_k():
    assert len(retrieve(QUESTION, INDEX, CHUNKS, BY_ID, k=1)) == 1


def test_prompt_numbers_every_source_and_ends_with_the_question():
    retrieved = acme_sources()
    system, user, sent = build_prompt(QUESTION, retrieved)
    assert sent == len(retrieved)
    for n, (chunk, _) in enumerate(retrieved, start=1):
        assert f"[{n}] {chunk['doc_name']}, page {chunk['page_num']}" in user
        assert chunk["text"] in user
    assert user.rstrip().endswith(f"Question: {QUESTION}")
    assert "ONLY the numbered sources" in system


def test_digits_count_as_one_token_each():
    assert estimate_tokens("2,107 1,577") >= 10
    assert estimate_tokens("revenue grew") < 6


def test_long_sources_are_cut_to_fit_the_budget():
    pages = [make_chunk("BIG_2022_10K", "Big", 2022, p, "Revenue 1,234,567. " * 300) for p in range(3)]
    retrieved = [(chunk, 1.0) for chunk in pages]
    _, user, sent = build_prompt("What was Big's revenue?", retrieved, max_tokens=4000)
    assert sent < len(retrieved)              # the lowest-ranked page was left out
    assert "[1] BIG_2022_10K, page 0" in user  # the best page always goes in
    assert "[...]" in user                     # the page that overflowed was cut short
    assert estimate_tokens(user) <= 4000 + 50  # sources plus the question line


def test_citation_to_a_page_that_was_not_sent_is_dropped():
    retrieved = acme_sources()
    reply = {"citations": [{"source": 1, "quote": "a"}, {"source": 2, "quote": "b"}],
             "calculation_steps": [], "answer": "x", "abstained": False}
    response = to_response("fb_1", META, retrieved, reply, "", latency_ms=5, sent=1)
    assert [c["evidence_page_num"] for c in response["citations"]] == [retrieved[0][0]["page_num"]]
    assert len(response["retrieved_but_unused"]) == len(retrieved) - 1


def test_citation_number_maps_to_that_sources_metadata():
    retrieved = acme_sources()
    reply = {"citations": [{"source": 2, "quote": "q"}], "calculation_steps": [],
             "answer": "x", "abstained": False}
    response = to_response("fb_1", META, retrieved, reply, "", latency_ms=5)
    chunk, score = retrieved[1]
    citation = response["citations"][0]
    assert citation["doc_name"] == chunk["doc_name"]
    assert citation["evidence_page_num"] == chunk["page_num"]
    assert citation["doc_link"] == chunk["doc_link"]
    assert citation["retrieval_score"] == round(score, 4)
    unused = [f"{c['doc_name']} p{c['page_num']}" for i, (c, _) in enumerate(retrieved, 1) if i != 2]
    assert response["retrieved_but_unused"] == unused


def test_source_numbers_that_were_never_sent_are_dropped():
    retrieved = acme_sources()
    bad = [{"source": 0, "quote": ""}, {"source": 99, "quote": ""},
           {"source": "1", "quote": ""}, {"source": True, "quote": ""}, "not a dict"]
    reply = {"citations": bad, "calculation_steps": [], "answer": "x", "abstained": False}
    assert to_response("fb_1", META, retrieved, reply, "", latency_ms=5)["citations"] == []


def test_response_has_exactly_the_fields_in_the_spec():
    retrieved = acme_sources()
    reply = json.loads(FakeLLM().generate("", ""))
    response = to_response("fb_1", META, retrieved, reply, "", latency_ms=5)
    assert set(response) == set(SPEC)
    assert set(response["calculation"]) == set(SPEC["calculation"])
    assert set(response["citations"][0]) == set(SPEC["citations"][0])


def test_abstention_clears_the_answer():
    reply = {"citations": [], "calculation_steps": [], "answer": "", "abstained": True}
    response = to_response("fb_1", META, acme_sources(), reply, "", latency_ms=5)
    assert response["abstained"] is True
    assert response["answer"] is None


def test_fenced_json_is_accepted_and_garbage_is_not():
    assert parse_reply('```json\n{"answer": "1"}\n```') == {"answer": "1"}
    assert parse_reply("Sorry, I cannot help with that.") is None
    assert parse_reply("[1, 2]") is None


def test_garbled_reply_is_recorded_not_crashed():
    response = to_response("fb_1", META, acme_sources(), None, "not json", latency_ms=5)
    assert response["answer"] is None
    assert response["citations"] == []
    assert "not valid JSON" in response["error"]


def test_end_to_end_with_the_fake_llm():
    response = answer_question("fb_1", QUESTION, META, INDEX, CHUNKS, BY_ID, FakeLLM(), k=3)
    assert response["model_calls"] == 1
    assert response["citations"][0]["company"] == "Acme"
    assert isinstance(response["latency_ms"], int)


def test_no_matching_pages_means_no_model_call():
    class ExplodingLLM:
        def generate(self, system, user):
            raise AssertionError("the model must not be called without sources")

    response = answer_question("fb_1", "zzz qqq", META, INDEX, CHUNKS, BY_ID, ExplodingLLM())
    assert response["abstained"] is True
    assert response["model_calls"] == 0


def test_ollama_retries_errors_and_empty_replies():
    class FlakyClient:  # fails, then answers with nothing, then works
        def __init__(self):
            self.replies = [RuntimeError("CUDA error: an illegal memory access"), "", '{"answer": "1"}']

        def chat(self, **kwargs):
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return SimpleNamespace(message=SimpleNamespace(content=reply), prompt_eval_count=100)

    llm = OllamaLLM()
    llm.client, llm.retry_wait = FlakyClient(), 0
    assert llm.generate("system", "user") == '{"answer": "1"}'
    assert llm.last_calls == 3


def test_a_model_that_keeps_failing_is_recorded_not_fatal():
    class BrokenLLM:
        num_ctx, last_calls = 8192, 3

        def generate(self, system, user):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    response = answer_question("fb_1", QUESTION, META, INDEX, CHUNKS, BY_ID, BrokenLLM(), k=3)
    assert response["answer"] is None
    assert response["model_calls"] == 3
    assert response["error"].startswith("model call failed: CUDA error")


def test_frozen_test_question_is_refused():
    test_id = (ROOT / "data" / "splits" / "test_ids.txt").read_text(encoding="utf-8").split()[0]
    with pytest.raises(SystemExit):
        main(["--id", test_id])
