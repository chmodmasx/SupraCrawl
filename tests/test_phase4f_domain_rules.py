from __future__ import annotations

from collections.abc import Callable

import pytest

import supracrawl.extractor as extractor_module
from supracrawl.config import Settings
from supracrawl.extractor import Extraction, Extractor
from supracrawl.fetcher import FetchResult, HttpFetcher


def _fetched(final_url: str, html: str | None = None) -> FetchResult:
    return FetchResult(
        fetch_url="https://requested.example/start",
        final_url=final_url,
        status_code=200,
        content_type="text/html",
        html=html
        or (
            "<html><head><title>Fixture</title></head><body><article>"
            "<p>fixture body</p></article></body></html>"
        ),
        fetched_at="2026-09-14T18:00:00+00:00",
    )


def _extraction(markdown: str, *, rendered: bool = False) -> Extraction:
    return Extraction(
        title="Fixture",
        markdown=markdown,
        canonical_url="https://placeholder.example/",
        extractor="readability+playwright" if rendered else "readability",
        quality=0.0,
        rendered=rendered,
    )


def test_domain_rules_default_empty() -> None:
    assert Settings().extraction_domain_rules == []


def test_domain_rule_parses_env_json_and_normalizes_host(monkeypatch) -> None:
    monkeypatch.setenv(
        "SUPRACRAWL_EXTRACTION_DOMAIN_RULES",
        '[{"host":"Example.COM...","remove_selectors":[" .noise "]}]',
    )
    settings = Settings(_env_file=None)

    assert len(settings.extraction_domain_rules) == 1
    rule = settings.extraction_domain_rules[0]
    assert rule.host == "example.com"
    assert rule.remove_selectors == [".noise"]
    assert rule.force_browser is False


@pytest.mark.parametrize(
    "host",
    [
        "https://example.com",
        "example.com:443",
        "example.com/path",
        "user@example.com",
        "*.example.com",
        "127.0.0.1",
        "2001:db8::1",
        "bad..example.com",
        "-bad.example.com",
        "bad-.example.com",
    ],
)
def test_domain_rule_rejects_non_exact_dns_hosts(host: str) -> None:
    with pytest.raises(ValueError):
        Settings(extraction_domain_rules=[{"host": host, "force_browser": True}])


def test_domain_rule_rejects_duplicate_normalized_hosts() -> None:
    with pytest.raises(ValueError, match="duplicate normalized hosts"):
        Settings(
            extraction_domain_rules=[
                {"host": "Example.com.", "force_browser": True},
                {"host": "example.com", "remove_selectors": [".noise"]},
            ]
        )


def test_domain_rule_rejects_unknown_field_and_noop_rule() -> None:
    with pytest.raises(ValueError):
        Settings(
            extraction_domain_rules=[
                {"host": "example.com", "force_browser": True, "content_selector": "main"}
            ]
        )
    with pytest.raises(ValueError, match="must enable"):
        Settings(extraction_domain_rules=[{"host": "example.com"}])


def test_domain_rule_matching_is_exact_and_normalized() -> None:
    settings = Settings(
        extraction_domain_rules=[{"host": "Example.COM.", "remove_selectors": [".noise"]}]
    )
    extractor = Extractor(settings, HttpFetcher(settings))

    assert extractor._domain_rule("https://EXAMPLE.COM./page") is not None
    assert extractor._domain_rule("https://sub.example.com/page") is None
    assert extractor._domain_rule("https://other.example/page") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rule_host", "expect_selectors"),
    [
        ("final.example", True),
        ("requested.example", False),
        ("canonical.example", False),
    ],
)
async def test_matching_uses_final_host_not_requested_or_canonical(
    monkeypatch,
    rule_host: str,
    expect_selectors: bool,
) -> None:
    settings = Settings(
        browser_enabled=False,
        extraction_domain_rules=[{"host": rule_host, "remove_selectors": [".noise"]}],
    )
    extractor = Extractor(settings, HttpFetcher(settings))
    calls: list[list[str] | None] = []

    async def fake_worker(
        _html: str,
        _url: str,
        render: bool,
        remove_selectors: list[str] | None = None,
    ) -> Extraction:
        assert render is False
        calls.append(remove_selectors)
        return _extraction("static extraction")

    monkeypatch.setattr(extractor, "_worker_extract", fake_worker)
    fetched = _fetched(
        "https://final.example/page",
        (
            "<html><head><title>Fixture</title>"
            '<link rel="canonical" href="https://canonical.example/article">'
            "</head><body><article><p>fixture</p></article></body></html>"
        ),
    )
    await extractor.extract_fetched(fetched)

    expected = [[".noise"]] if expect_selectors else [None]
    assert calls == expected


@pytest.mark.asyncio
async def test_force_browser_selects_successful_render_and_carries_selectors(monkeypatch) -> None:
    settings = Settings(
        browser_enabled=True,
        extraction_domain_rules=[
            {
                "host": "final.example",
                "remove_selectors": [".noise"],
                "force_browser": True,
            }
        ],
    )
    extractor = Extractor(settings, HttpFetcher(settings))
    calls: list[tuple[bool, list[str] | None]] = []

    async def fake_worker(
        _html: str,
        _url: str,
        render: bool,
        remove_selectors: list[str] | None = None,
    ) -> Extraction:
        calls.append((render, remove_selectors))
        if render:
            return _extraction("R", rendered=True)
        return _extraction("S" * 1000)

    monkeypatch.setattr(extractor, "_worker_extract", fake_worker)
    monkeypatch.setattr(extractor, "_needs_browser", lambda *_args: False)

    result = await extractor.extract_fetched(_fetched("https://final.example/page"))

    assert calls == [(False, [".noise"]), (True, [".noise"])]
    assert result.markdown == "R"
    assert result.rendered is True


@pytest.mark.asyncio
async def test_force_browser_disabled_keeps_static(monkeypatch) -> None:
    settings = Settings(
        browser_enabled=False,
        extraction_domain_rules=[{"host": "final.example", "force_browser": True}],
    )
    extractor = Extractor(settings, HttpFetcher(settings))
    calls: list[bool] = []

    async def fake_worker(
        _html: str,
        _url: str,
        render: bool,
        remove_selectors: list[str] | None = None,
    ) -> Extraction:
        assert remove_selectors is None
        calls.append(render)
        return _extraction("static")

    monkeypatch.setattr(extractor, "_worker_extract", fake_worker)

    result = await extractor.extract_fetched(_fetched("https://final.example/page"))

    assert calls == [False]
    assert result.markdown == "static"
    assert result.rendered is False


@pytest.mark.asyncio
async def test_force_browser_render_failure_keeps_static(monkeypatch) -> None:
    settings = Settings(
        browser_enabled=True,
        extraction_domain_rules=[{"host": "final.example", "force_browser": True}],
    )
    extractor = Extractor(settings, HttpFetcher(settings))

    async def fake_worker(
        _html: str,
        _url: str,
        render: bool,
        remove_selectors: list[str] | None = None,
    ) -> Extraction | None:
        assert remove_selectors is None
        return None if render else _extraction("static")

    monkeypatch.setattr(extractor, "_worker_extract", fake_worker)
    monkeypatch.setattr(extractor, "_needs_browser", lambda *_args: False)

    result = await extractor.extract_fetched(_fetched("https://final.example/page"))

    assert result.markdown == "static"
    assert result.rendered is False


@pytest.mark.asyncio
async def test_unforced_browser_retains_existing_better_only_selection(monkeypatch) -> None:
    settings = Settings(browser_enabled=True)
    extractor = Extractor(settings, HttpFetcher(settings))
    calls: list[bool] = []

    async def fake_worker(
        _html: str,
        _url: str,
        render: bool,
        remove_selectors: list[str] | None = None,
    ) -> Extraction:
        assert remove_selectors is None
        calls.append(render)
        if render:
            return _extraction("R", rendered=True)
        return _extraction("S" * 1000)

    monkeypatch.setattr(extractor, "_worker_extract", fake_worker)
    monkeypatch.setattr(extractor, "_needs_browser", lambda *_args: True)

    result = await extractor.extract_fetched(_fetched("https://unmatched.example/page"))

    assert calls == [False, True]
    assert result.markdown == "S" * 1000
    assert result.rendered is False


class _PayloadResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"title": "Rendered", "markdown": "rendered markdown"}


class _PayloadClient:
    def __init__(self, capture: Callable[[str, dict], None]) -> None:
        self.capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: dict) -> _PayloadResponse:
        self.capture(url, json)
        return _PayloadResponse()


@pytest.mark.asyncio
async def test_worker_payload_includes_selectors_for_render(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        extractor_module.httpx,
        "AsyncClient",
        lambda **_kwargs: _PayloadClient(lambda url, payload: captured.append((url, payload))),
    )
    settings = Settings(extractor_worker_url="http://worker:3000")
    extractor = Extractor(settings, HttpFetcher(settings))

    result = await extractor._worker_extract(
        "",
        "https://example.com/page",
        render=True,
        remove_selectors=[".noise"],
    )

    assert result is not None
    assert captured == [
        (
            "http://worker:3000/render-extract",
            {"url": "https://example.com/page", "remove_selectors": [".noise"]},
        )
    ]
