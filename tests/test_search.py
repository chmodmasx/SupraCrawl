from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from supracrawl.chunking import Chunk
from supracrawl.config import Settings
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.search import OpenSearchStore, canonical_identity_url, document_id


def test_document_id_uses_normalized_url_identity() -> None:
    first = document_id("HTTPS://Example.COM:443/path?b=2&a=1#fragment")
    second = document_id("https://example.com/path?a=1&b=2")
    assert first == second


def test_same_host_canonical_can_define_document_identity() -> None:
    identity = canonical_identity_url(
        "https://example.com/article?utm_source=test&id=7",
        "https://example.com/article?id=7",
    )
    assert identity == "https://example.com/article?id=7"


def test_cross_origin_canonical_cannot_take_over_document_identity() -> None:
    identity = canonical_identity_url(
        "https://example.com/article?id=7",
        "https://attacker.example/other-document",
    )
    assert identity == "https://example.com/article?id=7"


@pytest.mark.asyncio
async def test_index_document_writes_document_and_chunks_then_removes_stale_chunks(
    monkeypatch,
) -> None:
    settings = Settings(opensearch_url="http://opensearch:9200")
    store = OpenSearchStore(settings)
    store._indices_ready = True
    calls: list[tuple[str, str, dict]] = []

    async def fake_request(method: str, path: str, **kwargs):
        calls.append((method, path, kwargs))
        if method == "HEAD":
            return httpx.Response(200)
        if path.startswith("/_bulk"):
            return httpx.Response(200, json={"errors": False, "items": []})
        if "_delete_by_query" in path:
            return httpx.Response(200, json={"deleted": 1})
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(store, "_request", fake_request)

    fetched = FetchResult(
        fetch_url="https://example.com/article",
        final_url="https://example.com/article",
        status_code=200,
        content_type="text/html",
        html="<html></html>",
        fetched_at="2026-08-27T12:00:00+00:00",
    )
    extraction = Extraction(
        title="Article",
        markdown="# Heading\n\nUseful text",
        canonical_url="https://example.com/article",
        extractor="readability",
        quality=0.9,
        rendered=False,
    )
    chunks = [
        Chunk(
            ordinal=0,
            section_path=("Heading",),
            text="# Heading\n\nUseful text",
            approx_tokens=8,
        )
    ]

    doc_id, count = await store.index_document(
        fetched=fetched,
        extraction=extraction,
        chunks=chunks,
        content_hash="a" * 64,
    )

    assert doc_id == document_id("https://example.com/article")
    assert count == 1
    assert len(calls) == 4
    assert calls[0][:2] == ("HEAD", f"/{settings.opensearch_documents_index}")
    assert calls[1][:2] == ("HEAD", f"/{settings.opensearch_chunks_index}")

    bulk_method, bulk_path, bulk_kwargs = calls[2]
    assert bulk_method == "POST"
    assert bulk_path == "/_bulk?refresh=wait_for"
    bulk_lines = bulk_kwargs["content"].decode("utf-8").strip().splitlines()
    assert len(bulk_lines) == 4
    document_action = json.loads(bulk_lines[0])
    document_source = json.loads(bulk_lines[1])
    chunk_action = json.loads(bulk_lines[2])
    chunk_source = json.loads(bulk_lines[3])
    assert document_action["index"]["_index"] == settings.opensearch_documents_index
    assert document_source["content_hash"] == "a" * 64
    assert document_source["identity_url"] == "https://example.com/article"
    assert chunk_action["index"]["_index"] == settings.opensearch_chunks_index
    assert chunk_source["document_id"] == doc_id
    assert chunk_source["section_path"] == "Heading"

    cleanup_method, cleanup_path, cleanup_kwargs = calls[3]
    assert cleanup_method == "POST"
    assert cleanup_path.startswith(f"/{settings.opensearch_chunks_index}/_delete_by_query")
    must_not = cleanup_kwargs["json"]["query"]["bool"]["must_not"]
    assert must_not == [{"term": {"content_hash": "a" * 64}}]


@pytest.mark.asyncio
async def test_concurrent_reindexes_of_same_document_are_serialized(monkeypatch) -> None:
    settings = Settings(opensearch_url="http://opensearch:9200")
    store = OpenSearchStore(settings)
    store._indices_ready = True
    first_bulk_started = asyncio.Event()
    release_first_bulk = asyncio.Event()
    operations: list[str] = []
    bulk_count = 0

    async def fake_request(method: str, path: str, **kwargs):
        nonlocal bulk_count
        if method == "HEAD":
            return httpx.Response(200)
        if path.startswith("/_bulk"):
            bulk_count += 1
            lines = kwargs["content"].decode("utf-8").strip().splitlines()
            digest = json.loads(lines[1])["content_hash"]
            operations.append(f"bulk:{digest[0]}")
            if bulk_count == 1:
                first_bulk_started.set()
                await release_first_bulk.wait()
            return httpx.Response(200, json={"errors": False, "items": []})
        if "_delete_by_query" in path:
            digest = kwargs["json"]["query"]["bool"]["must_not"][0]["term"]["content_hash"]
            operations.append(f"cleanup:{digest[0]}")
            return httpx.Response(200, json={"deleted": 1})
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(store, "_request", fake_request)

    fetched = FetchResult(
        fetch_url="https://example.com/article",
        final_url="https://example.com/article",
        status_code=200,
        content_type="text/html",
        html="<html></html>",
        fetched_at="2026-08-27T12:00:00+00:00",
    )
    extraction = Extraction(
        title="Article",
        markdown="# Heading\n\nUseful text",
        canonical_url="https://example.com/article",
        extractor="readability",
        quality=0.9,
        rendered=False,
    )
    chunks = [
        Chunk(
            ordinal=0,
            section_path=("Heading",),
            text="# Heading\n\nUseful text",
            approx_tokens=8,
        )
    ]

    first = asyncio.create_task(
        store.index_document(fetched, extraction, chunks, content_hash="a" * 64)
    )
    await first_bulk_started.wait()
    second = asyncio.create_task(
        store.index_document(fetched, extraction, chunks, content_hash="b" * 64)
    )

    await asyncio.sleep(0.05)
    assert operations == ["bulk:a"]

    release_first_bulk.set()
    await asyncio.gather(first, second)

    assert operations == ["bulk:a", "cleanup:a", "bulk:b", "cleanup:b"]
    assert store._document_locks == {}


@pytest.mark.asyncio
async def test_search_uses_bm25_fields_and_collapses_by_document(monkeypatch) -> None:
    settings = Settings(opensearch_url="http://opensearch:9200")
    store = OpenSearchStore(settings)
    store._indices_ready = True
    captured: dict = {}
    expected_text = "BM25 returns the most relevant passage for the indexed page."

    async def fake_request(method: str, path: str, **kwargs):
        captured.update({"method": method, "path": path, "body": kwargs["json"]})
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_score": 4.25,
                            "_source": {
                                "document_id": "doc-1",
                                "url": "https://example.com/article",
                                "title": "SupraCrawl article",
                                "section_path": "Architecture > Search",
                                "text": expected_text,
                                "ordinal": 2,
                                "fetched_at": "2026-08-27T12:00:00+00:00",
                            },
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr(store, "_request", fake_request)

    results = await store.search("BM25 relevant passage", 5)

    assert captured["method"] == "POST"
    assert captured["path"] == f"/{settings.opensearch_chunks_index}/_search"
    assert captured["body"]["collapse"] == {"field": "document_id"}
    fields = captured["body"]["query"]["bool"]["should"][0]["multi_match"]["fields"]
    assert fields == ["title^4", "section_path^2", "text"]
    assert results == [
        {
            "title": "SupraCrawl article",
            "url": "https://example.com/article",
            "description": expected_text,
            "position": 1,
            "score": 4.25,
            "metadata": {
                "document_id": "doc-1",
                "section_path": "Architecture > Search",
                "chunk_ordinal": 2,
                "fetched_at": "2026-08-27T12:00:00+00:00",
            },
        }
    ]


@pytest.mark.asyncio
async def test_search_recreates_missing_index_after_ready_state(monkeypatch) -> None:
    settings = Settings(opensearch_url="http://opensearch:9200")
    store = OpenSearchStore(settings)
    store._indices_ready = True
    calls: list[tuple[str, str]] = []
    first_search = True

    async def fake_request(method: str, path: str, **_kwargs):
        nonlocal first_search
        calls.append((method, path))
        search_path = f"/{settings.opensearch_chunks_index}/_search"
        if path == search_path and first_search:
            first_search = False
            return httpx.Response(404, json={"error": "index_not_found_exception"})
        if method == "HEAD":
            if path == f"/{settings.opensearch_documents_index}":
                return httpx.Response(200)
            if path == f"/{settings.opensearch_chunks_index}":
                return httpx.Response(404)
        if method == "PUT" and path == f"/{settings.opensearch_chunks_index}":
            return httpx.Response(200, json={"acknowledged": True})
        if path == search_path:
            return httpx.Response(200, json={"hits": {"hits": []}})
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(store, "_request", fake_request)

    assert await store.search("anything", 5) == []
    assert calls.count(("POST", f"/{settings.opensearch_chunks_index}/_search")) == 2
    assert ("PUT", f"/{settings.opensearch_chunks_index}") in calls
    assert store._indices_ready is True
