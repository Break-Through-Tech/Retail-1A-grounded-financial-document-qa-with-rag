# Frozen-test evaluation infrastructure

Issue #33 uses a separate corpus and consumes rankings from a previously frozen
retrieval configuration. It does not select models, tune parameters, download
documents, or modify the benchmark JSONL or committed split files.

## Prepare the isolated corpus

Place exactly the 26 required test PDFs in `data/test_pdfs/` using an approved
download process. Then run:

```bash
python -m src.prepare_test_corpus \
  --pdf-dir data/test_pdfs \
  --out-dir data/test_processed
```

The command writes `pages.jsonl`, the three chunk variants, and
`corpus_manifest.json` under `data/test_processed`. It never reads or writes the
development PDF/chunk directories. It fails unless every required filing has
non-empty extracted text and usable chunks. The manifest records the PDF/output
roots, file sizes, SHA-256 hashes, dataset/test-ID hashes, and Git provenance.

## Evaluate frozen rankings

The retrieval system should first emit one JSONL record per frozen question:

```json
{"financebench_id":"financebench_id_...","rankings":[
  {"doc_name":"FILING","evidence_page_num":12,"score":0.5}
]}
```

With a separately frozen configuration JSON, run:

`retrieval_config.json` must contain at least:

```json
{"retriever":"bm25","chunking":{"strategy":"page"},"retrieval_depth":10}
```

```bash
python -m src.evaluation.evaluate_test \
  --rankings rankings.jsonl \
  --config retrieval_config.json \
  --corpus-manifest data/test_processed/corpus_manifest.json \
  --output-dir results/final-test
```

The output directory records the full Git commit, dataset hash, retrieval
configuration, copied corpus manifest, per-question rankings and metrics. The
headline metrics are Hit@1, Hit@5 and MRR@10. `all@5` is supplementary and
requires every gold evidence page for a question to appear in the top five. The
evaluator requires a clean Git worktree and refuses to overwrite a non-empty
output directory.
