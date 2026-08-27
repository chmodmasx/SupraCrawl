from __future__ import annotations

import pytest

from supracrawl.crawler import Crawler, discover_links
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.indexer import IndexOutcome


def test_discover_links_normalizes_and_deduplicates() -> None:
    html = """
    <html><body>
      <a href="/a?b=2&a=1#fragment">one</a>
      <a href="https://example.com/a?a=1&b=2">duplicate</a>
      <a href="https://other.example/page">external</a>
      <a href="mailto:test@example.com">mail</a>
    </body></html>
    """

    assert discover_links(html, "https://example.com/start") == [
        "https://example.com/a?a=1&b=2",
        "https://other.example/page",
    ]


@pytest.mark.asyncio
async def test_crawler_respects_same_origin_and_depth() -> None:
    class FakeFetcher:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def fetch_html(self, url: str) -> FetchResult:
            self.calls.append(url)
            if url == "https://example.com/":
                html = (
                    '<a href="/inside">inside</a>'
                    '<a href="https://outside.example/ignored">outside</a>'
                )
            else:
                html = '<a href="/deeper">deeper</a>'
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html=html,
                fetched_at="2026-08-27T12:00:00+00:00",
            )

    class FakeExtractor:
        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            return Extraction(
                title=fetched.final_url,
                markdown="# Page\n\nIndexed body",
                canonical_url=fetched.final_url,
                extractor="fixture",
                quality=1.0,
                rendered=False,
            )

    class FakeIndexer:
        async def index_extraction(
            self,
            fetched: FetchResult,
            _extraction: Extraction,
        ) -> IndexOutcome:
            return IndexOutcome(
                url=fetched.final_url,
                indexed=True,
                document_id=f"doc-{len(fetched.final_url)}",
                content_hash="a" * 64,
                chunks_indexed=1,
            )

    fetcher = FakeFetcher()
    crawler = Crawler(fetcher, FakeExtractor(), FakeIndexer())  # type: ignore[arg-type]

    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=10,
        max_depth=1,
        same_origin=True,
    )

    assert fetcher.calls == ["https://example.com/", "https://example.com/inside"]
    assert [outcome.depth for outcome in outcomes] == [0, 1]
    assert all(outcome.indexed for outcome in outcomes)
