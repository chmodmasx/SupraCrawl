from __future__ import annotations

import pytest

from supracrawl.config import Settings
from supracrawl.embeddings import EmbeddingBackendError
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.indexer import Indexer


class _Store:
    def __init__(self) -> None:
        self.lexical_calls = 0
        self.vector_calls = 0
        self.vector_vectors: list[list[float]] = []

    async def index_document(self, **kwargs):
        self.lexical_calls += 1
        return "doc-1", len(kwargs["chunks"])

    async def index_vector_document(self, **kwargs):
        self.vector_calls += 1
        self.vector_vectors = kwargs["vectors"]
        return "doc-1", len(kwargs["chunks"])


class _Embedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.passages: list[str] = []

    async def embed_passages(self, passages):
        self.calls += 1
        self.passages = list(passages)
        if self.fail:
            raise EmbeddingBackendError("embedding failed")
        return [[0.1, 0.2, 0.3] for _ in passages]


def _fixture() -> tuple[FetchResult, Extraction]:
    fetched = FetchResult(
        fetch_url="https://example.com/article",
        final_url="https://example.com/article",
        status_code=200,
        content_type="text/html",
        html="",
        fetched_at="2026-08-27T12:00:00+00:00",
    )
    extraction = Extraction(
        title="Hybrid article",
        markdown="# Search\n\nHybrid retrieval combines lexical and semantic ranking.",
        canonical_url="https://example.com/article",
        extractor="fixture",
        quality=1.0,
        rendered=False,
    )
    return fetched, extraction


@pytest.mark.asyncio
async def test_dense_enabled_indexes_lexical_then_vector_with_passage_prefix() -> None:
    settings = Settings(
        dense_enabled=True,
        dense_dimension=3,
        dense_passage_prefix="passage: ",
    )
    store = _Store()
    embedder = _Embedder()
    indexer = Indexer(settings, object(), store, embedder=embedder)  # type: ignore[arg-type]
    fetched, extraction = _fixture()

    outcome = await indexer.index_extraction(fetched, extraction)

    assert outcome.indexed is True
    assert outcome.document_id == "doc-1"
    assert outcome.chunks_indexed >= 1
    assert outcome.vector_indexed is True
    assert outcome.vector_chunks_indexed == outcome.chunks_indexed
    assert outcome.vector_error is None
    assert store.lexical_calls == 1
    assert store.vector_calls == 1
    assert embedder.calls == 1
    assert embedder.passages[0].startswith("passage: Hybrid article\nSearch\n")
    assert len(store.vector_vectors) == outcome.chunks_indexed


@pytest.mark.asyncio
async def test_vector_failure_preserves_successful_lexical_index() -> None:
    settings = Settings(dense_enabled=True, dense_dimension=3)
    store = _Store()
    embedder = _Embedder(fail=True)
    indexer = Indexer(settings, object(), store, embedder=embedder)  # type: ignore[arg-type]
    fetched, extraction = _fixture()

    outcome = await indexer.index_extraction(fetched, extraction)

    assert outcome.indexed is True
    assert outcome.document_id == "doc-1"
    assert outcome.chunks_indexed >= 1
    assert outcome.vector_indexed is False
    assert outcome.vector_chunks_indexed == 0
    assert outcome.vector_error == "embedding failed"
    assert outcome.error is None
    assert store.lexical_calls == 1
    assert store.vector_calls == 0


@pytest.mark.asyncio
async def test_dense_disabled_never_attempts_vector_indexing() -> None:
    settings = Settings(dense_enabled=False)
    store = _Store()
    embedder = _Embedder()
    indexer = Indexer(settings, object(), store, embedder=embedder)  # type: ignore[arg-type]
    fetched, extraction = _fixture()

    outcome = await indexer.index_extraction(fetched, extraction)

    assert outcome.indexed is True
    assert outcome.vector_indexed is None
    assert outcome.vector_chunks_indexed == 0
    assert outcome.vector_error is None
    assert store.lexical_calls == 1
    assert store.vector_calls == 0
    assert embedder.calls == 0
