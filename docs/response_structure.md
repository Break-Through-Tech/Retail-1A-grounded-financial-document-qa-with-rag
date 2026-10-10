{
  "financebench_id": "string",
  "question_type": "string (dataset: metrics-generated | domain-relevant | novel-generated)",
  "question_reasoning": "string (dataset: e.g. Information extraction, Numerical reasoning, Logical reasoning)",
  "answer": "string or null",
  "confidence": "float 0-1",
  "abstained": "boolean",
  "calculation": {
    "shown": "boolean",
    "steps": ["string"]
  },
  "citations": [
    {
      "doc_name": "string",
      "company": "string",
      "doc_type": "string",
      "doc_period": "int",
      "evidence_page_num": "int",
      "evidence_text": "string (verbatim excerpt used)",
      "doc_link": "string",
      "retrieval_score": "float",
      "rerank_score": "float"
    }
  ],
  "retrieved_but_unused": ["array of doc_name + page, for audit trail"],
  "latency_ms": "int",
  "model_calls": "int"
}
