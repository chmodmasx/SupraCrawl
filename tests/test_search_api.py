from __future__ import annotations

import httpx
import pytest

import supracrawl.app as app_module
from supracrawl.crawler import CrawlOutcome
from supracrawl.indexer import IndexOutcome
from supracrawl.search import SearchBackendError


@pytest.mark.asyncio
async def test_search_endpoint_returns_ranked_results(monkeypatch) -> None:
    async def fake_search(query: str, limit: int):
        assert query == "supra crawl"
        assert limit == 3
        return [
            {
                "title": "Result",
                "url": "https://example.com/result",
                "description": "Relevant indexed passage",
                "position": 1,
                "score": 2.5,
                "metadata": {"document_id": "doc-1"},
            }
        ]

    monkeypatch.setattr(app_module.search_store, "search", fake_search)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/search",
            json={"query": "supra crawl", "limit": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["results"][0]["url"] == "https://example.com/result"
    assert body["results"][0]["position"] == 1


@pytest.mark.asyncio
async def test_search_endpoint_returns_503_when_backend_is_unavailable(monkeypatch) -> None:
    async def fail_search(_query: str, _limit: int):
        raise SearchBackendError("OpenSearch request failed: fixture")

    monkeypatch.setattr(app_module.search_store, "search", fail_search)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/search",
            json={"query": "anything", "limit": 5},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "OpenSearch request failed: fixture"


@pytest.mark.asyncio
async def test_index_endpoint_preserves_per_url_failures(monkeypatch) -> None:
    async def fake_index(url: str) -> IndexOutcome:
        if "bad" in url:
            return IndexOutcome(url=url, indexed=False, error="Fetch failed: fixture")
        return IndexOutcome(
            url=url,
            indexed=True,
            document_id="doc-1",
            content_hash="a" * 64,
            chunks_indexed=2,
        )

    monkeypatch.setattr(app_module.indexer, "index_url", fake_index)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/index",
            json={
                "urls": [
                    "https://example.com/good",
                    "https://example.com/bad",
                ]
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["items"][0]["indexed"] is True
    assert body["items"][1]["indexed"] is False
    assert body["items"][1]["error"] == "Fetch failed: fixture"


@pytest.mark.asyncio
async def test_crawl_endpoint_reports_visited_and_indexed_counts(monkeypatch) -> None:
    async def fake_crawl(**_kwargs):
        return [
            CrawlOutcome(
                url="https://example.com/",
                depth=0,
                indexed=True,
                document_id="doc-1",
                content_hash="a" * 64,
                chunks_indexed=2,
            ),
            CrawlOutcome(
                url="https://example.com/fail",
                depth=1,
                indexed=False,
                error="Fetch failed: fixture",
            ),
        ]

    monkeypatch.setattr(app_module.crawler, "crawl", fake_crawl)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/crawl",
            json={
                "seeds": ["https://example.com/"],
                "max_pages": 5,
                "max_depth": 1,
                "same_origin": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["pages_visited"] == 2
    assert body["pages_indexed"] == 1
    assert body["pages"][1]["error"] == "Fetch failed: fixture"
