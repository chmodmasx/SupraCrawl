from __future__ import annotations

import httpx
import pytest

from supracrawl.config import Settings
from supracrawl.embeddings import EmbeddingBackendError
from supracrawl.retrieval import SearchService
from supracrawl.search import OpenSearchStore, SearchBackendError


def _result(document_id: str, position: int, *, url: str | None = None) -> dict:
    return {
        "title": document_id,
        "url": url or f"https://example.com/{document_id}",
        "description": f"content for {document_id}",
        "position": position,
        "score": float(10 - position),
        "metadata": {"document_id": document_id},
    }


class _Embedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed_query(self, _query: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2, 0.3]


class _Store:
    def __init__(self, lexical: list[dict] | None = None) -> None:
        self.lexical = lexical or []
        self.search_calls = 0

    async def search(self, _query: str, _limit: int) -> list[dict]:
        self.search_calls += 1
        return self.lexical


@pytest.mark.asyncio
async def test_default_bm25_does_not_touch_embedder(monkeypatch) -> None:
    settings = Settings(search_mode="bm25", dense_enabled=True)
    store = _Store([_result("doc-a", 1)])
    embedder = _Embedder()
    service = SearchService(settings, store, embedder)  # type: ignore[arg-type]

    async def unexpected_dense(*_args, **_kwargs):
        raise AssertionError("BM25 must not enter the dense path")

    monkeypatch.setattr(service, "_dense_search", unexpected_dense)
    execution = await service.search("query", 5)

    assert execution.mode_requested == "bm25"
    assert execution.mode_used == "bm25"
    assert execution.degraded is False
    assert execution.results[0]["metadata"]["retrieval_mode"] == "bm25"
    assert embedder.calls == 0


@pytest.mark.asyncio
async def test_explicit_hybrid_with_dense_disabled_degrades_to_bm25() -> None:
    settings = Settings(search_mode="bm25", dense_enabled=False)
    store = _Store([_result("doc-a", 1)])
    embedder = _Embedder()
    service = SearchService(settings, store, embedder)  # type: ignore[arg-type]

    execution = await service.search("query", 5, mode="hybrid")

    assert execution.mode_requested == "hybrid"
    assert execution.mode_used == "bm25"
    assert execution.degraded is True
    assert "disabled" in (execution.degradation_reason or "")
    assert embedder.calls == 0


@pytest.mark.asyncio
async def test_hybrid_fuses_rankings_with_deterministic_provenance(monkeypatch) -> None:
    settings = Settings(search_mode="bm25", dense_enabled=True, hybrid_rrf_k=60)
    lexical = [_result("exact", 1), _result("semantic", 2)]
    dense = [_result("semantic", 1), _result("dense-only", 2)]
    store = _Store(lexical)
    service = SearchService(settings, store, _Embedder())  # type: ignore[arg-type]

    async def fake_dense(_query: str, _limit: int) -> list[dict]:
        return dense

    async def current(results: list[dict]) -> list[dict]:
        return results

    monkeypatch.setattr(service, "_dense_search", fake_dense)
    monkeypatch.setattr(service, "_filter_current_dense_results", current)

    execution = await service.search("query", 3, mode="hybrid")

    assert execution.mode_used == "hybrid"
    assert execution.degraded is False
    assert [item["position"] for item in execution.results] == [1, 2, 3]
    assert [_result_id(item) for item in execution.results] == [
        "semantic",
        "exact",
        "dense-only",
    ]
    semantic = execution.results[0]
    assert semantic["metadata"]["retrieval_mode"] == "hybrid"
    assert semantic["metadata"]["lexical_rank"] == 2
    assert semantic["metadata"]["dense_rank"] == 1
    assert semantic["score"] == pytest.approx(1 / 62 + 1 / 61)


def _result_id(result: dict) -> str:
    return result["metadata"]["document_id"]


@pytest.mark.asyncio
async def test_vector_failure_degrades_without_masking_lexical_results(monkeypatch) -> None:
    settings = Settings(dense_enabled=True)
    store = _Store([_result("doc-a", 1)])
    service = SearchService(settings, store, _Embedder())  # type: ignore[arg-type]

    async def broken_dense(_query: str, _limit: int) -> list[dict]:
        raise EmbeddingBackendError("model unavailable")

    monkeypatch.setattr(service, "_dense_search", broken_dense)
    execution = await service.search("query", 5, mode="hybrid")

    assert execution.mode_used == "bm25"
    assert execution.degraded is True
    assert "model unavailable" in (execution.degradation_reason or "")
    assert [_result_id(item) for item in execution.results] == ["doc-a"]


@pytest.mark.asyncio
async def test_vector_value_error_degrades_without_masking_lexical_results(monkeypatch) -> None:
    settings = Settings(dense_enabled=True)
    store = _Store([_result("doc-a", 1)])
    service = SearchService(settings, store, _Embedder())  # type: ignore[arg-type]

    async def invalid_dense(_query: str, _limit: int) -> list[dict]:
        raise ValueError("query must not be empty")

    monkeypatch.setattr(service, "_dense_search", invalid_dense)
    execution = await service.search("   ", 5, mode="hybrid")

    assert execution.mode_used == "bm25"
    assert execution.degraded is True
    assert "query must not be empty" in (execution.degradation_reason or "")
    assert [_result_id(item) for item in execution.results] == ["doc-a"]


@pytest.mark.asyncio
async def test_lexical_failure_is_not_hidden_by_hybrid() -> None:
    class BrokenStore(_Store):
        async def search(self, _query: str, _limit: int) -> list[dict]:
            raise SearchBackendError("lexical backend unavailable")

    settings = Settings(dense_enabled=True)
    service = SearchService(settings, BrokenStore(), _Embedder())  # type: ignore[arg-type]

    with pytest.raises(SearchBackendError, match="lexical backend unavailable"):
        await service.search("query", 5, mode="hybrid")


@pytest.mark.asyncio
async def test_current_hash_filter_discards_stale_dense_candidates(monkeypatch) -> None:
    settings = Settings(
        opensearch_url="http://opensearch:9200",
        opensearch_documents_index="docs-v1",
    )
    store = OpenSearchStore(settings)
    service = SearchService(settings, store, _Embedder())  # type: ignore[arg-type]
    captured: dict = {}

    async def fake_request(method: str, path: str, **kwargs):
        captured.update({"method": method, "path": path, "body": kwargs["json"]})
        return httpx.Response(
            200,
            json={
                "docs": [
                    {"_id": "stale", "found": True, "_source": {"content_hash": "new"}},
                    {"_id": "current", "found": True, "_source": {"content_hash": "same"}},
                ]
            },
        )

    monkeypatch.setattr(store, "_request_with_index_recovery", fake_request)
    results = [
        {
            **_result("stale", 1),
            "metadata": {"document_id": "stale", "content_hash": "old"},
        },
        {
            **_result("current", 2),
            "metadata": {"document_id": "current", "content_hash": "same"},
        },
    ]

    filtered = await service._filter_current_dense_results(results)

    assert [_result_id(item) for item in filtered] == ["current"]
    assert captured == {
        "method": "POST",
        "path": "/docs-v1/_mget",
        "body": {"ids": ["stale", "current"]},
    }


@pytest.mark.asyncio
async def test_hash_validation_item_error_degrades_with_operation_context(monkeypatch) -> None:
    settings = Settings(
        opensearch_url="http://opensearch:9200",
        opensearch_documents_index="docs-v1",
        dense_enabled=True,
    )
    store = OpenSearchStore(settings)
    service = SearchService(settings, store, _Embedder())
    lexical = [_result("doc-a", 1)]
    dense = [
        {
            **_result("doc-a", 1),
            "metadata": {"document_id": "doc-a", "content_hash": "hash-a"},
        }
    ]

    async def fake_search(_query: str, _limit: int) -> list[dict]:
        return lexical

    async def fake_dense(_query: str, _limit: int) -> list[dict]:
        return dense

    async def item_error_request(*_args, **_kwargs):
        return httpx.Response(
            200,
            json={
                "docs": [
                    {
                        "_id": "doc-a",
                        "error": {
                            "type": "cluster_block_exception",
                            "reason": "index blocked by: [FORBIDDEN/5/index read-only]",
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(store, "search", fake_search)
    monkeypatch.setattr(service, "_dense_search", fake_dense)
    monkeypatch.setattr(store, "_request_with_index_recovery", item_error_request)

    execution = await service.search("query", 5, mode="hybrid")

    assert execution.mode_used == "bm25"
    assert execution.degraded is True
    assert "current-content validation" in (execution.degradation_reason or "")
    assert [_result_id(item) for item in execution.results] == ["doc-a"]


@pytest.mark.asyncio
async def test_hash_validation_transport_failure_keeps_operation_context(monkeypatch) -> None:
    settings = Settings(
        opensearch_url="http://opensearch:9200",
        opensearch_documents_index="docs-v1",
        dense_enabled=True,
    )
    store = OpenSearchStore(settings)
    service = SearchService(settings, store, _Embedder())
    lexical = [_result("doc-a", 1)]
    dense = [
        {
            **_result("doc-a", 1),
            "metadata": {"document_id": "doc-a", "content_hash": "hash-a"},
        }
    ]

    async def fake_search(_query: str, _limit: int) -> list[dict]:
        return lexical

    async def fake_dense(_query: str, _limit: int) -> list[dict]:
        return dense

    async def broken_request(*_args, **_kwargs):
        raise SearchBackendError("OpenSearch request failed: ")

    monkeypatch.setattr(store, "search", fake_search)
    monkeypatch.setattr(service, "_dense_search", fake_dense)
    monkeypatch.setattr(store, "_request_with_index_recovery", broken_request)

    execution = await service.search("query", 5, mode="hybrid")

    assert execution.mode_used == "bm25"
    assert execution.degraded is True
    assert "current-content validation" in (execution.degradation_reason or "")
    assert "OpenSearch request failed" in (execution.degradation_reason or "")
    assert [_result_id(item) for item in execution.results] == ["doc-a"]


@pytest.mark.asyncio
async def test_hash_validation_failure_degrades_to_bm25(monkeypatch) -> None:
    settings = Settings(dense_enabled=True)
    store = _Store([_result("doc-a", 1)])
    service = SearchService(settings, store, _Embedder())  # type: ignore[arg-type]

    async def fake_dense(_query: str, _limit: int) -> list[dict]:
        return [_result("doc-a", 1)]

    async def broken_validation(_results: list[dict]) -> list[dict]:
        raise SearchBackendError("hash validation unavailable")

    monkeypatch.setattr(service, "_dense_search", fake_dense)
    monkeypatch.setattr(service, "_filter_current_dense_results", broken_validation)

    execution = await service.search("query", 5, mode="hybrid")

    assert execution.mode_used == "bm25"
    assert execution.degraded is True
    assert "hash validation unavailable" in (execution.degradation_reason or "")
