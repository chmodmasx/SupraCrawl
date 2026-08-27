from __future__ import annotations

import json
import math

import httpx
import pytest

from supracrawl.chunking import Chunk
from supracrawl.config import Settings
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.search import OpenSearchStore, SearchBackendError, document_id


def _settings() -> Settings:
    return Settings(
        opensearch_url="http://opensearch:9200",
        dense_model_name="test/e5-small",
        dense_dimension=3,
        opensearch_vector_chunks_index="test-vector-chunks-v1",
    )


def _mapping_payload(store: OpenSearchStore) -> dict:
    definition = store._vector_chunks_mapping()
    return {
        store.settings.opensearch_vector_chunks_index: {
            "mappings": definition["mappings"],
        }
    }


def _fixture() -> tuple[FetchResult, Extraction, list[Chunk]]:
    fetched = FetchResult(
        fetch_url="https://example.com/vector",
        final_url="https://example.com/vector",
        status_code=200,
        content_type="text/html",
        html="",
        fetched_at="2026-08-27T12:00:00+00:00",
    )
    extraction = Extraction(
        title="Vector article",
        markdown="# Dense\n\nSemantic body",
        canonical_url="https://example.com/vector",
        extractor="fixture",
        quality=1.0,
        rendered=False,
    )
    chunks = [
        Chunk(
            ordinal=0,
            section_path=("Dense",),
            text="Semantic body",
            approx_tokens=4,
        )
    ]
    return fetched, extraction, chunks


def test_vector_mapping_is_exact_lucene_cosine_with_model_provenance() -> None:
    store = OpenSearchStore(_settings())
    mapping = store._vector_chunks_mapping()

    assert mapping["settings"]["index"]["knn"] is True
    meta = mapping["mappings"]["_meta"]
    assert meta == {
        "embedding_model": "test/e5-small",
        "embedding_dimension": 3,
        "vector_engine": "lucene",
        "vector_method": "flat",
        "vector_space": "cosinesimil",
    }
    vector = mapping["mappings"]["properties"]["embedding"]
    assert vector["type"] == "knn_vector"
    assert vector["dimension"] == 3
    assert vector["method"] == {
        "name": "flat",
        "engine": "lucene",
        "space_type": "cosinesimil",
    }


def test_vector_mapping_model_mismatch_fails_closed() -> None:
    store = OpenSearchStore(_settings())
    payload = _mapping_payload(store)
    payload[store.settings.opensearch_vector_chunks_index]["mappings"]["_meta"][
        "embedding_model"
    ] = "other/model"

    with pytest.raises(SearchBackendError, match="embedding_model"):
        store._validate_vector_mapping_payload(payload)


def test_vector_mapping_dimension_mismatch_fails_closed() -> None:
    store = OpenSearchStore(_settings())
    payload = _mapping_payload(store)
    payload[store.settings.opensearch_vector_chunks_index]["mappings"]["properties"][
        "embedding"
    ]["dimension"] = 4

    with pytest.raises(SearchBackendError, match="dimension"):
        store._validate_vector_mapping_payload(payload)


@pytest.mark.parametrize(
    "vector",
    [
        [0.1, 0.2],
        [0.1, math.inf, 0.3],
        [0.0, 0.0, 0.0],
    ],
)
def test_vector_validation_rejects_wrong_or_unsafe_vectors(vector) -> None:
    store = OpenSearchStore(_settings())
    with pytest.raises(SearchBackendError):
        store._validated_vector(vector)


@pytest.mark.asyncio
async def test_vector_index_writes_provenance_and_removes_stale_chunks(monkeypatch) -> None:
    store = OpenSearchStore(_settings())
    fetched, extraction, chunks = _fixture()
    calls: list[tuple[str, str, dict]] = []

    async def fake_request(method: str, path: str, **kwargs):
        calls.append((method, path, kwargs))
        if method == "HEAD":
            return httpx.Response(200)
        if method == "GET" and path.endswith("/_mapping"):
            return httpx.Response(200, json=_mapping_payload(store))
        if path.startswith("/_bulk"):
            return httpx.Response(200, json={"errors": False, "items": []})
        if "_delete_by_query" in path:
            return httpx.Response(200, json={"deleted": 1})
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(store, "_request", fake_request)

    doc_id, count = await store.index_vector_document(
        fetched=fetched,
        extraction=extraction,
        chunks=chunks,
        content_hash="a" * 64,
        vectors=[[0.1, 0.2, 0.3]],
    )

    assert doc_id == document_id("https://example.com/vector")
    assert count == 1
    bulk = next(call for call in calls if call[1].startswith("/_bulk"))
    lines = bulk[2]["content"].decode("utf-8").strip().splitlines()
    assert len(lines) == 2
    source = json.loads(lines[1])
    assert source["embedding_model"] == "test/e5-small"
    assert source["embedding_dimension"] == 3
    assert source["embedding"] == pytest.approx([0.1, 0.2, 0.3])
    mapping_reads = [
        call for call in calls if call[0] == "GET" and call[1].endswith("/_mapping")
    ]
    assert len(mapping_reads) == 2
    cleanup = next(call for call in calls if "_delete_by_query" in call[1])
    assert cleanup[2]["json"]["query"]["bool"]["must_not"] == [
        {"term": {"content_hash": "a" * 64}}
    ]


@pytest.mark.asyncio
async def test_vector_mapping_race_after_bulk_fails_closed(monkeypatch) -> None:
    store = OpenSearchStore(_settings())
    fetched, extraction, chunks = _fixture()
    mapping_reads = 0

    async def fake_request(method: str, path: str, **_kwargs):
        nonlocal mapping_reads
        if method == "HEAD":
            return httpx.Response(200)
        if method == "GET" and path.endswith("/_mapping"):
            mapping_reads += 1
            payload = _mapping_payload(store)
            if mapping_reads == 2:
                payload[store.settings.opensearch_vector_chunks_index]["mappings"][
                    "_meta"
                ]["embedding_model"] = "auto-created/incompatible"
            return httpx.Response(200, json=payload)
        if path.startswith("/_bulk"):
            return httpx.Response(200, json={"errors": False, "items": []})
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(store, "_request", fake_request)

    with pytest.raises(SearchBackendError, match="embedding_model"):
        await store.index_vector_document(
            fetched=fetched,
            extraction=extraction,
            chunks=chunks,
            content_hash="a" * 64,
            vectors=[[0.1, 0.2, 0.3]],
        )

    assert mapping_reads == 2


@pytest.mark.asyncio
async def test_vector_count_mismatch_fails_before_opensearch(monkeypatch) -> None:
    store = OpenSearchStore(_settings())
    fetched, extraction, chunks = _fixture()

    async def unexpected_request(*_args, **_kwargs):
        raise AssertionError("OpenSearch must not be contacted for invalid vectors")

    monkeypatch.setattr(store, "_request", unexpected_request)

    with pytest.raises(SearchBackendError, match="vector count mismatch"):
        await store.index_vector_document(
            fetched=fetched,
            extraction=extraction,
            chunks=chunks,
            content_hash="a" * 64,
            vectors=[],
        )


@pytest.mark.asyncio
async def test_dense_search_uses_knn_collapse_and_returns_vector_provenance(monkeypatch) -> None:
    store = OpenSearchStore(_settings())
    store._vector_index_ready = True
    captured: dict = {}

    async def fake_request(method: str, path: str, **kwargs):
        captured.update({"method": method, "path": path, "body": kwargs["json"]})
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_score": 0.95,
                            "_source": {
                                "document_id": "doc-1",
                                "url": "https://example.com/vector",
                                "title": "Vector article",
                                "section_path": "Dense",
                                "text": "Semantic body",
                                "ordinal": 0,
                                "fetched_at": "2026-08-27T12:00:00+00:00",
                                "embedding_model": "test/e5-small",
                                "embedding_dimension": 3,
                            },
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr(store, "_request", fake_request)

    results = await store.dense_search([0.1, 0.2, 0.3], limit=5)

    assert captured["method"] == "POST"
    assert captured["path"] == "/test-vector-chunks-v1/_search"
    assert captured["body"]["collapse"] == {"field": "document_id"}
    knn = captured["body"]["query"]["knn"]["embedding"]
    assert knn["vector"] == pytest.approx([0.1, 0.2, 0.3])
    assert knn["k"] >= 5
    assert results[0]["metadata"]["embedding_model"] == "test/e5-small"
    assert results[0]["metadata"]["embedding_dimension"] == 3


@pytest.mark.asyncio
async def test_bm25_search_does_not_touch_vector_index(monkeypatch) -> None:
    settings = _settings()
    assert settings.dense_enabled is False
    store = OpenSearchStore(settings)
    store._indices_ready = True
    paths: list[str] = []

    async def fake_request(method: str, path: str, **_kwargs):
        paths.append(path)
        assert method == "POST"
        return httpx.Response(200, json={"hits": {"hits": []}})

    monkeypatch.setattr(store, "_request", fake_request)

    assert await store.search("lexical", limit=5) == []
    assert paths == [f"/{settings.opensearch_chunks_index}/_search"]
