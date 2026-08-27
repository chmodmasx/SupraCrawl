from __future__ import annotations

import inspect

import pytest
from agent.web_search_provider import WebSearchProvider

import integrations.hermes.supracrawl.provider as provider_module
from integrations.hermes.supracrawl.provider import SupraCrawlWebSearchProvider


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "success": True,
            "documents": [
                {
                    "url": "https://example.com/",
                    "title": "Example",
                    "content": "Example content",
                    "raw_content": "",
                    "metadata": {"extractor": "fixture"},
                    "error": None,
                }
            ],
        }


class _FakeClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, _url: str, json: dict) -> _FakeResponse:
        assert json["urls"] == ["https://example.com/"]
        return _FakeResponse()


@pytest.mark.asyncio
async def test_provider_matches_current_hermes_extract_contract(monkeypatch) -> None:
    values = {
        "SUPRACRAWL_URL": "http://supracrawl:8080",
        "SUPRACRAWL_TIMEOUT_S": "20",
    }
    monkeypatch.setattr(
        provider_module,
        "get_provider_env",
        lambda name: values.get(name, ""),
    )
    monkeypatch.setattr(
        provider_module.httpx,
        "AsyncClient",
        lambda **_kwargs: _FakeClient(),
    )

    provider = SupraCrawlWebSearchProvider()

    assert isinstance(provider, WebSearchProvider)
    assert provider.name == "supracrawl"
    assert provider.is_available() is True
    assert provider.supports_search() is False
    assert provider.supports_extract() is True
    assert inspect.iscoroutinefunction(provider.extract)

    result = await provider.extract(["https://example.com/"])

    assert isinstance(result, list)
    assert result[0]["url"] == "https://example.com/"
    assert result[0]["content"] == "Example content"
    assert "success" not in result[0]
    assert "data" not in result[0]


@pytest.mark.asyncio
async def test_provider_returns_per_url_errors_on_backend_failure(monkeypatch) -> None:
    values = {
        "SUPRACRAWL_URL": "",
        "SUPRACRAWL_TIMEOUT_S": "20",
    }
    monkeypatch.setattr(
        provider_module,
        "get_provider_env",
        lambda name: values.get(name, ""),
    )

    provider = SupraCrawlWebSearchProvider()
    result = await provider.extract(["https://example.com/a", "https://example.com/b"])

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item["error"] == "SUPRACRAWL_URL is not configured" for item in result)
