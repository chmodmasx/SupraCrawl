from __future__ import annotations

import pytest

import supracrawl.fetcher as fetcher_module
from supracrawl.config import Settings
from supracrawl.crawler import Crawler
from supracrawl.fetcher import FetchError, FetchResult, HttpFetcher
from supracrawl.indexer import RevalidationIndexHit
from supracrawl.security import UnsafeUrlError


@pytest.mark.asyncio
async def test_phase4b_fresh_leaf_rejects_unsafe_admission_before_network_skip() -> None:
    class Fetcher:
        def __init__(self) -> None:
            self.admission_calls: list[str] = []

        async def ensure_fetch_allowed(self, url: str) -> None:
            self.admission_calls.append(url)
            raise UnsafeUrlError("fixture private destination")

        async def fetch_html(self, _url: str, **_kwargs) -> FetchResult:
            raise AssertionError("blocked fresh leaf must not fetch")

    class Extractor:
        async def extract_fetched(self, _fetched: FetchResult):
            raise AssertionError("blocked fresh leaf must not extract")

    class Indexer:
        async def revalidation_document(self, _url: str):
            return RevalidationIndexHit(
                document_id="doc-1",
                content_hash="a" * 64,
                fetched_at="2026-09-10T12:00:00+00:00",
                age_s=60.0,
                etag='"v1"',
            )

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            raise AssertionError("blocked fresh leaf must return before Phase 4A lookup")

        async def index_extraction(self, _fetched: FetchResult, _extraction):
            raise AssertionError("blocked fresh leaf must not index")

    fetcher = Fetcher()
    crawler = Crawler(fetcher, Extractor(), Indexer())  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
        refresh_after_s=3600,
        conditional_revalidate_leaves=True,
    )

    assert fetcher.admission_calls == ["https://example.com/"]
    assert len(outcomes) == 1
    assert outcomes[0].indexed is False
    assert outcomes[0].freshness_skipped is False
    assert outcomes[0].network_fetch_skipped is False
    assert outcomes[0].error == "Unsafe URL: fixture private destination"


@pytest.mark.asyncio
async def test_phase4b_fetch_admission_keeps_robots_policy(monkeypatch) -> None:
    class Robots:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def allowed(self, url: str, user_agent: str) -> bool:
            self.calls.append((url, user_agent))
            return False

    async def allow_public_url(_url: str) -> None:
        return None

    monkeypatch.setattr(fetcher_module, "validate_public_url", allow_public_url)
    robots = Robots()
    settings = Settings(obey_robots_txt=True)
    fetcher = HttpFetcher(settings, robots=robots)  # type: ignore[arg-type]

    with pytest.raises(FetchError, match="Blocked by robots.txt"):
        await fetcher.ensure_fetch_allowed("https://example.com/private")

    assert robots.calls == [("https://example.com/private", settings.user_agent)]
