"""Sentence Transformers adapter for the dense retriever (issue #18).

Wraps a pinned Sentence Transformers model so it satisfies
:class:`src.retrieval.dense.TextEncoder`. No model is chosen here: the model ID and an
exact 40-character commit revision are required, so every run is tied to one
immutable set of weights. A branch name such as ``main`` or a short hash is rejected
because it does not pin the weights.

``sentence_transformers``, and with it ``torch``, is imported only inside
:meth:`SentenceTransformerEncoder.load`. Importing this module therefore loads no
neural-network library. The adapter never normalizes embeddings; the vector index
does that for documents and queries alike.
"""

from __future__ import annotations

import re
from importlib import metadata
from typing import Any, Sequence

import numpy as np

__all__ = ["REVISION_PATTERN", "SentenceTransformerEncoder"]

#: A full Hugging Face commit hash, the only revision form that pins model weights.
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")

_MODEL_METHODS = ("encode_document", "encode_query")


def _check_text(name: str, value: object) -> str:
    """Return ``value`` if it is a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _check_settings(
    model_id: object,
    revision: object,
    device: object,
    query_prompt: object,
    batch_size: object,
) -> None:
    """Validate the adapter settings shared by ``__init__`` and :meth:`load`."""
    _check_text("model_id", model_id)
    _check_text("revision", revision)
    if not REVISION_PATTERN.fullmatch(revision):  # type: ignore[arg-type]
        raise ValueError(
            f"revision must be a full 40-character lowercase commit hash, got {revision!r}"
        )
    _check_text("device", device)
    if query_prompt is not None:
        _check_text("query_prompt", query_prompt)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be int, got {type(batch_size).__name__}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; got {batch_size}")


def _package_version(name: str) -> str | None:
    """Return the installed version of ``name`` without importing it, or ``None``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class SentenceTransformerEncoder:
    """A :class:`~src.retrieval.dense.TextEncoder` backed by a Sentence Transformers model.

    Documents go through the model's ``encode_document`` and questions through
    ``encode_query``. When ``query_prompt`` is ``None`` no prompt is passed, so the
    model applies only the query prompt its own configuration defines, if any.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        query_prompt: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Wrap an already-loaded model.

        Args:
            model: A loaded ``SentenceTransformer``, or any object with
                ``encode_document`` and ``encode_query`` methods.
            model_id: Hugging Face model ID the weights came from.
            revision: Full 40-character commit hash the weights came from.
            device: Device the model runs on, recorded for provenance.
            query_prompt: Prompt prepended to every question, or ``None``.
            batch_size: Texts encoded per forward pass.

        Raises:
            TypeError: If ``model`` lacks a required method or a setting has the wrong
                type.
            ValueError: If a setting is empty, the revision is not a full commit hash,
                or ``batch_size`` is below 1.
        """
        _check_settings(model_id, revision, device, query_prompt, batch_size)
        missing = [name for name in _MODEL_METHODS if not callable(getattr(model, name, None))]
        if missing:
            raise TypeError(f"model is missing required method(s): {missing}")
        self._model = model
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._query_prompt = query_prompt
        self._batch_size = batch_size

    @classmethod
    def load(
        cls,
        model_id: str,
        revision: str,
        *,
        device: str = "cpu",
        query_prompt: str | None = None,
        batch_size: int = 32,
    ) -> SentenceTransformerEncoder:
        """Load a pinned Sentence Transformers model and wrap it.

        Every argument is validated before ``sentence_transformers`` is imported, so a
        bad revision fails without loading anything. Remote code is never trusted.

        Args:
            model_id: Hugging Face model ID.
            revision: Full 40-character commit hash to load.
            device: Device to run on.
            query_prompt: Prompt prepended to every question, or ``None``.
            batch_size: Texts encoded per forward pass.

        Returns:
            The wrapped model.

        Raises:
            TypeError: If a setting has the wrong type.
            ValueError: If a setting is empty, the revision is not a full commit hash,
                or ``batch_size`` is below 1.
        """
        _check_settings(model_id, revision, device, query_prompt, batch_size)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_id, revision=revision, device=device, trust_remote_code=False
        )
        return cls(
            model,
            model_id=model_id,
            revision=revision,
            device=device,
            query_prompt=query_prompt,
            batch_size=batch_size,
        )

    def _encode_options(self) -> dict[str, Any]:
        """Options shared by document and query encoding; never normalizes."""
        return {
            "batch_size": self._batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": False,
            "show_progress_bar": False,
        }

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return one unnormalized embedding row per chunk text, via ``encode_document``."""
        return self._model.encode_document(list(texts), **self._encode_options())

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Return one unnormalized embedding row per question, via ``encode_query``."""
        options = self._encode_options()
        if self._query_prompt is not None:
            options["prompt"] = self._query_prompt
        return self._model.encode_query(list(texts), **options)

    def describe(self) -> dict[str, Any]:
        """Return the model identity and encoding settings, for provenance.

        Returns:
            A JSON-serializable dict. Library versions are read from package metadata,
            so this does not import them.
        """
        dimension = getattr(self._model, "get_sentence_embedding_dimension", None)
        return {
            "encoder": "sentence-transformers",
            "model_id": self._model_id,
            "revision": self._revision,
            "device": self._device,
            "batch_size": self._batch_size,
            "query_prompt": self._query_prompt,
            "normalize_embeddings": False,
            "embedding_dimension": dimension() if callable(dimension) else None,
            "max_seq_length": getattr(self._model, "max_seq_length", None),
            "sentence_transformers_version": _package_version("sentence-transformers"),
            "torch_version": _package_version("torch"),
        }
