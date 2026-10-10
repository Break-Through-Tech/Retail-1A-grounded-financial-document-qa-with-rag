"""Tests for the Sentence Transformers adapter (issue #18).

No real model is loaded. A stub stands in for a loaded model, and a fake
``sentence_transformers`` module is placed in ``sys.modules`` when :meth:`load` is
exercised. Nothing here touches the network, a model cache, or a GPU.

The load-bearing test is :func:`test_load_requires_a_pinned_commit_revision`: it runs
with ``sentence_transformers`` made unimportable, so it proves a bad revision is
rejected before anything is imported or downloaded.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.retrieval.dense import DenseRetriever
from src.retrieval.embeddings import REVISION_PATTERN, SentenceTransformerEncoder

ROOT = Path(__file__).resolve().parents[1]

#: A syntactically valid 40-character commit hash. It names no real model revision.
REVISION = "0123456789abcdef0123456789abcdef01234567"
MODEL_ID = "example-org/example-model"

#: Text -> fixed 2-D vector, so a stub model behaves deterministically.
VECTORS = {"east": [1.0, 0.0], "north": [0.0, 1.0], "q": [2.0, 1.0]}


class StubModel:
    """Stands in for a loaded SentenceTransformer and records every encode call."""

    max_seq_length = 512

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, Any]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def _encode(self, kind: str, sentences: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append((kind, list(sentences), kwargs))
        return np.array([VECTORS[text] for text in sentences], dtype=np.float32)

    def encode_document(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        return self._encode("document", sentences, **kwargs)

    def encode_query(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        return self._encode("query", sentences, **kwargs)


def _encoder(**overrides: Any) -> tuple[SentenceTransformerEncoder, StubModel]:
    model = StubModel()
    settings: dict[str, Any] = {"model_id": MODEL_ID, "revision": REVISION, **overrides}
    return SentenceTransformerEncoder(model, **settings), model


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Put a fake ``sentence_transformers`` module in ``sys.modules``; return its call log."""
    constructions: list[dict[str, Any]] = []

    class FakeSentenceTransformer(StubModel):
        def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
            super().__init__()
            constructions.append({"model_name_or_path": model_name_or_path, **kwargs})

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return constructions


# --- revision pinning and loading -----------------------------------------------


@pytest.mark.parametrize(
    "revision",
    ["main", "v1.0", "5c38ec7", "0123456789ABCDEF0123456789ABCDEF01234567", REVISION + "0", ""],
)
def test_load_requires_a_pinned_commit_revision(
    monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    # Making the module unimportable proves validation happens before the import.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ValueError):
        SentenceTransformerEncoder.load(MODEL_ID, revision)


@pytest.mark.parametrize("revision", [None, 5, b"0123456789abcdef0123456789abcdef01234567"])
def test_non_string_revision_is_rejected(revision: object) -> None:
    with pytest.raises(TypeError):
        SentenceTransformerEncoder.load(MODEL_ID, revision)  # type: ignore[arg-type]


def test_revision_pattern_accepts_only_a_full_lowercase_commit_hash() -> None:
    assert REVISION_PATTERN.fullmatch(REVISION)
    assert not REVISION_PATTERN.fullmatch("main")
    assert not REVISION_PATTERN.fullmatch(REVISION[:7])


def test_load_passes_revision_device_and_disables_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions = _install_fake_sentence_transformers(monkeypatch)
    encoder = SentenceTransformerEncoder.load(MODEL_ID, REVISION, device="cpu", batch_size=8)
    assert constructions == [
        {
            "model_name_or_path": MODEL_ID,
            "revision": REVISION,
            "device": "cpu",
            "trust_remote_code": False,
        }
    ]
    assert encoder.describe()["revision"] == REVISION


def test_importing_the_adapter_does_not_load_sentence_transformers_or_torch() -> None:
    code = (
        "import sys, src.retrieval.embeddings, src.retrieval.dense; "
        "print(sorted(m for m in ('sentence_transformers', 'torch') if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"


# --- encoding -------------------------------------------------------------------


def test_encode_documents_uses_encode_document_without_normalization() -> None:
    encoder, model = _encoder(batch_size=16)
    vectors = encoder.encode_documents(("east", "north"))
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert model.calls == [
        (
            "document",
            ["east", "north"],
            {
                "batch_size": 16,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "show_progress_bar": False,
            },
        )
    ]


def test_encode_queries_uses_encode_query_without_a_prompt_by_default() -> None:
    encoder, model = _encoder()
    encoder.encode_queries(["q"])
    kind, texts, kwargs = model.calls[0]
    assert (kind, texts) == ("query", ["q"])
    assert kwargs["normalize_embeddings"] is False
    assert "prompt" not in kwargs


def test_encode_queries_passes_the_configured_query_prompt() -> None:
    encoder, model = _encoder(query_prompt="Represent this question: ")
    encoder.encode_queries(["q"])
    assert model.calls[0][2]["prompt"] == "Represent this question: "


def test_describe_reports_model_revision_dimension_and_max_seq_length() -> None:
    encoder, _ = _encoder(device="cpu", batch_size=4)
    record = encoder.describe()
    assert record["encoder"] == "sentence-transformers"
    assert record["model_id"] == MODEL_ID
    assert record["revision"] == REVISION
    assert record["device"] == "cpu"
    assert record["batch_size"] == 4
    assert record["query_prompt"] is None
    assert record["normalize_embeddings"] is False
    assert record["embedding_dimension"] == 2
    assert record["max_seq_length"] == 512
    assert set(record) >= {"sentence_transformers_version", "torch_version"}


def test_adapter_works_as_a_text_encoder() -> None:
    encoder, model = _encoder()
    retriever = DenseRetriever.build([("b", "north"), ("a", "east")], encoder)
    results = retriever.retrieve("q", 2)
    assert [r.chunk_id for r in results] == ["a", "b"]
    assert [kind for kind, _, _ in model.calls] == ["document", "query"]
    assert retriever.config_record()["encoder"]["revision"] == REVISION


# --- settings validation --------------------------------------------------------


@pytest.mark.parametrize("batch_size", [0, -4])
def test_batch_size_below_one_is_rejected(batch_size: int) -> None:
    with pytest.raises(ValueError):
        _encoder(batch_size=batch_size)
    with pytest.raises(ValueError):
        SentenceTransformerEncoder.load(MODEL_ID, REVISION, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [1.5, "8", True, None])
def test_non_integer_batch_size_is_rejected(batch_size: object) -> None:
    with pytest.raises(TypeError):
        _encoder(batch_size=batch_size)


@pytest.mark.parametrize("field, value", [("model_id", ""), ("device", " "), ("query_prompt", "")])
def test_empty_settings_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        _encoder(**{field: value})


def test_model_without_encode_methods_is_rejected() -> None:
    with pytest.raises(TypeError, match="missing required method"):
        SentenceTransformerEncoder(object(), model_id=MODEL_ID, revision=REVISION)
