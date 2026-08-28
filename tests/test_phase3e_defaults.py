from __future__ import annotations

import pytest

from supracrawl.config import Settings
from supracrawl.embeddings import EmbeddingBackendError
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.indexer import Indexer
from supracrawl.retrieval import SearchService


def _result(document_id: str, position: int) -> dict:
    return {
        "title": document_id,
        "url": f"https://example.com/{document_id}",
        "description": f"content for {document_id}",
        "position": position,
        "score": float(10 - position),
        "metadata": {"document_id": document_id},
    }


class _Embedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.query_calls = 0
        self.passage_calls = 0

    async def embed_query(self, _query: str) -> list[float]:
        self.query_calls += 1
        if self.fail:
            raise EmbeddingBackendError("embedding failed")
        return [0.1, 0.2, 0.3]

    async def embed_passages(self, passages) -> list[list[float]]:
        self.passage_calls += 1
        if self.fail:
            raise EmbeddingBackendError("embedding failed")
        return [[0.1, 0.2, 0.3] for _ in passages]


class _SearchStore:
    def __init__(self, lexical: list[dict]) -> None:
        self.lexical = lexical
        self.search_calls = 0

    async def search(self, _query: str, _limit: int) -> list[dict]:
        self.search_calls += 1
        return self.lexical


class _IndexStore:
    def __init__(self) -> None:
        self.lexical_calls = 0
        self.vector_calls = 0

    async def index_document(self, **kwargs):
        self.lexical_calls += 1
        return "doc-1", len(kwargs["chunks"])

    async def index_vector_document(self, **kwargs):
        self.vector_calls += 1
        return "doc-1", len(kwargs["chunks"])


def _index_fixture() -> tuple[FetchResult, Extraction]:
    url = "https://example.com/phase3e"
    return (
        FetchResult(
            fetch_url=url,
            final_url=url,
            status_code=200,
            content_type="text/html",
            html="",
            fetched_at="2026-08-28T00:00:00+00:00",
        ),
        Extraction(
            title="Phase 3E default",
            markdown="# Hybrid default\n\nHybrid retrieval is the promoted default path.",
            canonical_url=url,
            extractor="fixture",
            quality=1.0,
            rendered=False,
        ),
    )


def test_phase3e_settings_promote_hybrid_and_dense_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SUPRACRAWL_SEARCH_MODE", raising=False)
    monkeypatch.delenv("SUPRACRAWL_DENSE_ENABLED", raising=False)

    settings = Settings(_env_file=None)

    assert settings.search_mode == "hybrid"
    assert settings.dense_enabled is True


def test_operator_environment_can_opt_out_to_bm25(monkeypatch) -> None:
    monkeypatch.setenv("SUPRACRAWL_SEARCH_MODE", "bm25")
    monkeypatch.setenv("SUPRACRAWL_DENSE_ENABLED", "false")

    settings = Settings(_env_file=None)

    assert settings.search_mode == "bm25"
    assert settings.dense_enabled is False


@pytest.mark.asyncio
async def test_default_request_uses_hybrid_when_vector_path_is_healthy(monkeypatch) -> None:
    settings = Settings(dense_enabled=True, hybrid_rrf_k=60)
    lexical = [_result("exact", 1), _result("semantic", 2)]
    dense = [_result("semantic", 1), _result("dense-only", 2)]
    store = _SearchStore(lexical)
    embedder = _Embedder()
    service = SearchService(settings, store, embedder)  # type: ignore[arg-type]

    async def fake_dense(_query: str, _limit: int) -> list[dict]:
        embedder.query_calls += 1
        return dense

    async def current(results: list[dict]) -> list[dict]:
        return results

    monkeypatch.setattr(service, "_dense_search", fake_dense)
    monkeypatch.setattr(service, "_filter_current_dense_results", current)

    execution = await service.search("query", 3)

    assert execution.mode_requested == "hybrid"
    assert execution.mode_used == "hybrid"
    assert execution.degraded is False
    assert [item["metadata"]["document_id"] for item in execution.results] == [
        "semantic",
        "exact",
        "dense-only",
    ]
    assert embedder.query_calls == 1


@pytest.mark.asyncio
async def test_explicit_bm25_never_touches_embedder_under_hybrid_defaults(monkeypatch) -> None:
    settings = Settings(dense_enabled=True)
    store = _SearchStore([_result("doc-a", 1)])
    embedder = _Embedder(fail=True)
    service = SearchService(settings, store, embedder)  # type: ignore[arg-type]

    async def unexpected_dense(*_args, **_kwargs):
        raise AssertionError("explicit BM25 must not enter the dense path")

    monkeypatch.setattr(service, "_dense_search", unexpected_dense)
    execution = await service.search("query", 5, mode="bm25")

    assert execution.mode_requested == "bm25"
    assert execution.mode_used == "bm25"
    assert execution.degraded is False
    assert execution.results[0]["metadata"]["retrieval_mode"] == "bm25"
    assert embedder.query_calls == 0


@pytest.mark.asyncio
async def test_default_hybrid_degrades_when_dense_capability_is_disabled() -> None:
    settings = Settings(dense_enabled=False)
    store = _SearchStore([_result("doc-a", 1)])
    embedder = _Embedder()
    service = SearchService(settings, store, embedder)  # type: ignore[arg-type]

    execution = await service.search("query", 5)

    assert execution.mode_requested == "hybrid"
    assert execution.mode_used == "bm25"
    assert execution.degraded is True
    assert "disabled" in (execution.degradation_reason or "")
    assert embedder.query_calls == 0


@pytest.mark.asyncio
async def test_default_indexing_attempts_vector_write() -> None:
    settings = Settings(dense_dimension=3)
    store = _IndexStore()
    embedder = _Embedder()
    indexer = Indexer(settings, object(), store, embedder=embedder)  # type: ignore[arg-type]
    fetched, extraction = _index_fixture()

    outcome = await indexer.index_extraction(fetched, extraction)

    assert outcome.indexed is True
    assert outcome.vector_indexed is True
    assert outcome.vector_chunks_indexed == outcome.chunks_indexed
    assert outcome.vector_error is None
    assert store.lexical_calls == 1
    assert store.vector_calls == 1
    assert embedder.passage_calls == 1
