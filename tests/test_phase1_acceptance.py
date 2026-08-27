from __future__ import annotations

from urllib.parse import urlsplit

import httpx
import pytest

import supracrawl.app as app_module
import supracrawl.fetcher as fetcher_module
import supracrawl.robots as robots_module
from supracrawl.cache import JsonCache
from supracrawl.config import Settings
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchError, FetchResult, HttpFetcher
from supracrawl.robots import RobotsPolicy
from supracrawl.security import UnsafeUrlError, validate_public_url
from supracrawl.urls import normalize_url


async def _allow_test_url(_url: str) -> None:
    return None


def _install_mock_transport(monkeypatch, module, transport: httpx.MockTransport) -> None:
    real_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client_factory)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://metadata.google.internal/",
    ],
)
async def test_ssrf_blocks_local_private_and_metadata_targets(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        await validate_public_url(url)


@pytest.mark.asyncio
async def test_ssrf_converts_invalid_port_into_safe_validation_error() -> None:
    with pytest.raises(UnsafeUrlError, match="invalid port"):
        await validate_public_url("https://example.com:not-a-port/")


def test_url_normalization_preserves_ipv6_brackets_and_removes_tracking() -> None:
    url = "https://[2606:4700:4700::1111]:443/a?utm_source=x&b=2&a=1#fragment"
    assert normalize_url(url) == "https://[2606:4700:4700::1111]/a?a=1&b=2"


@pytest.mark.asyncio
async def test_fetcher_follows_redirect_chain_without_automatic_redirects(monkeypatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/middle"})
        if request.url.path == "/middle":
            return httpx.Response(307, headers={"location": "/final"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><body><main>fixture final page</main></body></html>",
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(fetcher_module, "validate_public_url", _allow_test_url)
    _install_mock_transport(monkeypatch, fetcher_module, transport)
    settings = Settings(obey_robots_txt=False, max_redirects=5)

    result = await HttpFetcher(settings).fetch_html("https://fixture.test/start")

    assert result.final_url == "https://fixture.test/final"
    assert "fixture final page" in result.html
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_fetcher_validates_redirect_target_before_request(monkeypatch) -> None:
    requested_paths: list[str] = []

    async def validator(url: str) -> None:
        if urlsplit(url).hostname == "127.0.0.1":
            raise UnsafeUrlError("fixture private target")

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        raise AssertionError("private redirect target must never be requested")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(fetcher_module, "validate_public_url", validator)
    _install_mock_transport(monkeypatch, fetcher_module, transport)
    settings = Settings(obey_robots_txt=False)

    with pytest.raises(UnsafeUrlError, match="fixture private target"):
        await HttpFetcher(settings).fetch_html("https://fixture.test/start")

    assert requested_paths == ["https://fixture.test/start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404, 429, 500])
async def test_fetcher_surfaces_http_failures(monkeypatch, status: int) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    monkeypatch.setattr(fetcher_module, "validate_public_url", _allow_test_url)
    _install_mock_transport(monkeypatch, fetcher_module, transport)
    settings = Settings(obey_robots_txt=False)

    with pytest.raises(FetchError, match=f"HTTP {status}"):
        await HttpFetcher(settings).fetch_html("https://fixture.test/status")


@pytest.mark.asyncio
async def test_fetcher_rejects_non_html_content(monkeypatch) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"{}",
        )
    )
    monkeypatch.setattr(fetcher_module, "validate_public_url", _allow_test_url)
    _install_mock_transport(monkeypatch, fetcher_module, transport)
    settings = Settings(obey_robots_txt=False)

    with pytest.raises(FetchError, match="Unsupported content type"):
        await HttpFetcher(settings).fetch_html("https://fixture.test/json")


@pytest.mark.asyncio
async def test_fetcher_enforces_declared_html_size_limit(monkeypatch) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "20000"},
            content=b"small body",
        )
    )
    monkeypatch.setattr(fetcher_module, "validate_public_url", _allow_test_url)
    _install_mock_transport(monkeypatch, fetcher_module, transport)
    settings = Settings(obey_robots_txt=False, max_html_bytes=16_384)

    with pytest.raises(FetchError, match="size limit"):
        await HttpFetcher(settings).fetch_html("https://fixture.test/large")


@pytest.mark.asyncio
async def test_fetcher_converts_transport_timeout_to_fetch_error(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fixture timeout", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(fetcher_module, "validate_public_url", _allow_test_url)
    _install_mock_transport(monkeypatch, fetcher_module, transport)
    settings = Settings(obey_robots_txt=False)

    with pytest.raises(FetchError, match="fixture timeout"):
        await HttpFetcher(settings).fetch_html("https://fixture.test/timeout")


@pytest.mark.asyncio
async def test_robots_obeys_disallow_and_allow(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/robots.txt"
        return httpx.Response(
            200,
            text="User-agent: *\nDisallow: /blocked\nAllow: /\n",
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(robots_module, "validate_public_url", _allow_test_url)
    _install_mock_transport(monkeypatch, robots_module, transport)
    policy = RobotsPolicy()

    assert await policy.allowed("https://fixture.test/public", "SupraCrawl") is True
    assert await policy.allowed("https://fixture.test/blocked", "SupraCrawl") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status, expected", [(404, True), (500, False)])
async def test_robots_4xx_allows_and_5xx_fails_closed(monkeypatch, status: int, expected: bool) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    monkeypatch.setattr(robots_module, "validate_public_url", _allow_test_url)
    _install_mock_transport(monkeypatch, robots_module, transport)

    assert await RobotsPolicy().allowed("https://fixture.test/page", "SupraCrawl") is expected


@pytest.mark.asyncio
async def test_corrupt_cache_entry_fails_open() -> None:
    class CorruptRedis:
        async def get(self, _key: str) -> bytes:
            return b"{this-is-not-json"

    cache = JsonCache(None, 60)
    cache.redis = CorruptRedis()  # type: ignore[assignment]

    assert await cache.get("fixture") is None


@pytest.mark.asyncio
async def test_api_enforces_hard_context_budget(monkeypatch) -> None:
    markdown = "# Fixture\n\n" + ("alpha beta gamma delta " * 1200)

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
            title="Fixture",
            markdown=markdown,
            canonical_url=url,
            extractor="fixture",
            quality=1.0,
            rendered=False,
        )
        return fetched, extraction

    monkeypatch.setattr(app_module.extractor, "extract", fake_extract)
    monkeypatch.setattr(app_module.cache, "redis", None)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/extract",
            json={"urls": ["https://example.com/page"], "max_context_tokens": 128},
        )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["error"] is None
    assert document["metadata"]["context_tokens_approx"] <= 128
    assert document["content"]


@pytest.mark.asyncio
async def test_api_preserves_order_with_mixed_success_and_failure(monkeypatch) -> None:
    async def fake_extract(url: str):
        if url.endswith("/bad"):
            raise FetchError("fixture failure")
        fetched = FetchResult(
            fetch_url=url,
            final_url=url,
            status_code=200,
            content_type="text/html",
            html="<html></html>",
            fetched_at="2026-08-27T12:00:00+00:00",
        )
        extraction = Extraction(
            title="Good",
            markdown="# Good\n\nworking content",
            canonical_url=url,
            extractor="fixture",
            quality=1.0,
            rendered=False,
        )
        return fetched, extraction

    monkeypatch.setattr(app_module.extractor, "extract", fake_extract)
    monkeypatch.setattr(app_module.cache, "redis", None)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/extract",
            json={"urls": ["https://example.com/good", "https://example.com/bad"]},
        )

    documents = response.json()["documents"]
    assert documents[0]["url"] == "https://example.com/good"
    assert documents[0]["error"] is None
    assert documents[1]["url"] == "https://example.com/bad"
    assert documents[1]["error"] == "Fetch failed: fixture failure"
