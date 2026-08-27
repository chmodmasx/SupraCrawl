from __future__ import annotations

import inspect

import pytest
from agent.web_search_provider import WebSearchProvider

import integrations.hermes.supracrawl.provider as provider_module
from integrations.hermes.supracrawl.provider import SupraCrawlWebSearchProvider


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _FakeAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, _url: str, json: dict) -> _FakeResponse:
        assert json["urls"] == ["https://example.com/"]
        return _FakeResponse(
            {
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
        )


class _FakeSyncClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, _url: str, json: dict) -> _FakeResponse:
        assert json == {"query": "indexed knowledge", "limit": 5}
        return _FakeResponse(
            {
                "success": True,
                "results": [
                    {
                        "title": "Indexed result",
                        "url": "https://example.com/result",
                        "description": "Relevant indexed passage",
                        "position": 1,
                        "score": 3.2,
                        "metadata": {"document_id": "doc-1"},
                    }
                ],
            }
        )


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
        lambda **_kwargs: _FakeAsyncClient(),
    )

    provider = SupraCrawlWebSearchProvider()

    assert isinstance(provider, WebSearchProvider)
    assert provider.name == "supracrawl"
    assert provider.is_available() is True
    assert provider.supports_search() is True
    assert provider.supports_extract() is True
    assert inspect.iscoroutinefunction(provider.extract)
    assert not inspect.iscoroutinefunction(provider.search)

    result = await provider.extract(["https://example.com/"])

    assert isinstance(result, list)
    assert result[0]["url"] == "https://example.com/"
    assert result[0]["content"] == "Example content"
    assert "success" not in result[0]
    assert "data" not in result[0]


def test_provider_matches_current_hermes_search_contract(monkeypatch) -> None:
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
        "Client",
        lambda **_kwargs: _FakeSyncClient(),
    )

    provider = SupraCrawlWebSearchProvider()
    result = provider.search("indexed knowledge", limit=5)

    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "Indexed result",
                    "url": "https://example.com/result",
                    "description": "Relevant indexed passage",
                    "position": 1,
                }
            ]
        },
    }


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


def test_search_returns_failure_envelope_when_backend_is_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(provider_module, "get_provider_env", lambda _name: "")
    provider = SupraCrawlWebSearchProvider()

    assert provider.search("anything") == {
        "success": False,
        "error": "SUPRACRAWL_URL is not configured",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_timeout", ["0", "-1", "nan", "inf", "-inf", "not-a-number"])
async def test_provider_rejects_non_positive_or_non_finite_timeout(
    monkeypatch, invalid_timeout: str
) -> None:
    values = {
        "SUPRACRAWL_URL": "http://supracrawl:8080",
        "SUPRACRAWL_TIMEOUT_S": invalid_timeout,
    }
    monkeypatch.setattr(
        provider_module,
        "get_provider_env",
        lambda name: values.get(name, ""),
    )

    provider = SupraCrawlWebSearchProvider()
    result = await provider.extract(["https://example.com/"])

    assert len(result) == 1
    assert result[0]["error"] == "SUPRACRAWL_TIMEOUT_S must be a positive finite number"
    assert provider.search("anything")["success"] is False
