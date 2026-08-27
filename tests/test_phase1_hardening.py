from __future__ import annotations

import httpx
import pytest

import supracrawl.app as app_module
import supracrawl.extractor as extractor_module
from supracrawl.config import Settings
from supracrawl.extractor import Extraction, Extractor
from supracrawl.fetcher import FetchResult, HttpFetcher


@pytest.mark.asyncio
async def test_schema_invalid_cache_object_is_treated_as_cache_miss(monkeypatch) -> None:
    async def fake_cache_get(_key: str) -> dict:
        # Valid JSON/dict, but not a valid ExtractedDocument.
        return {"metadata": {"extractor": "broken"}}

    async def fake_cache_set(_key: str, _value: dict) -> None:
        return None

    async def fake_extract(url: str):
        fetched = FetchResult(
            fetch_url=url,
            final_url=url,
            status_code=200,
            content_type="text/html",
            html="<html></html>",
            fetched_at="2026-08-27T12:00:00+00:00",
        )
        extraction = Extraction(
            title="Recovered",
            markdown="# Recovered\n\nfresh extraction after invalid cache object",
            canonical_url=url,
            extractor="fixture",
            quality=1.0,
            rendered=False,
        )
        return fetched, extraction

    monkeypatch.setattr(app_module.cache, "get", fake_cache_get)
    monkeypatch.setattr(app_module.cache, "set", fake_cache_set)
    monkeypatch.setattr(app_module.extractor, "extract", fake_extract)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/extract",
            json={"urls": ["https://example.com/recover"]},
        )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["error"] is None
    assert document["title"] == "Recovered"
    assert "fresh extraction" in document["content"]


class _WorkerResponse:
    def __init__(self, body) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.body


class _WorkerClient:
    def __init__(self, body) -> None:
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, _url: str, json: dict) -> _WorkerResponse:
        return _WorkerResponse(self.body)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [[], {"markdown": {"invalid": "type"}}])
async def test_malformed_worker_json_falls_back_instead_of_crashing(monkeypatch, body) -> None:
    monkeypatch.setattr(
        extractor_module.httpx,
        "AsyncClient",
        lambda **_kwargs: _WorkerClient(body),
    )
    settings = Settings()
    extractor = Extractor(settings, HttpFetcher(settings))

    result = await extractor._worker_extract(
        "<html><body>fixture</body></html>",
        "https://example.com/",
        render=False,
    )

    assert result is None
