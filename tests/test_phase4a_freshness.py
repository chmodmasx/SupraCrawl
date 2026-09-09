from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import supracrawl.app as app_module
from supracrawl.config import Settings
from supracrawl.crawler import CrawlOutcome, Crawler
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.indexer import FreshIndexHit, IndexOutcome, Indexer
from supracrawl.models import CrawlRequest
from supracrawl.search import SearchBackendError


class _LookupStore:
    def __init__(self, payload: dict | None = None, *, fail: bool = False) -> None:
        self.payload = payload or {"hits": {"hits": []}}
        self.fail = fail
        self.calls: list[tuple[str, str, dict]] = []
        self.ensure_calls = 0

    async def ensure_indices(self) -> None:
        self.ensure_calls += 1

    async def _request_with_index_recovery(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        if self.fail:
            raise SearchBackendError("fixture lookup failure")
        return httpx.Response(200, json=self.payload)


def _document_payload(fetched_at: str) -> dict:
    return {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "document_id": "doc-1",
                        "content_hash": "a" * 64,
                        "fetched_at": fetched_at,
                        "url": "https://example.com/",
                    }
                }
            ]
        }
    }


def test_phase4a_policy_matches_request_contract() -> None:
    policy_path = Path(__file__).resolve().parents[1] / "evaluation/phase4a_freshness_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    contract = policy["request_contract"]

    assert policy["base_certified_sha"] == "09ecd5b67bc54f9758c660b60ca13539697502fd"
    assert contract == {
        "field": "refresh_after_s",
        "default": 0,
        "minimum": 0,
        "maximum": 2_592_000,
        "fresh_condition": "age_s < refresh_after_s",
    }


def test_phase4a_request_contract_defaults_to_disabled_and_is_bounded() -> None:
    request = CrawlRequest(seeds=["https://example.com/"])
    assert request.refresh_after_s == 0

    with pytest.raises(ValidationError):
        CrawlRequest(seeds=["https://example.com/"], refresh_after_s=2_592_001)


@pytest.mark.asyncio
async def test_phase4a_fresh_document_uses_exact_url_and_strict_age_boundary() -> None:
    store = _LookupStore(_document_payload("2026-09-09T12:00:00+00:00"))
    settings = Settings(dense_enabled=False, opensearch_url="http://opensearch:9200")
    indexer = Indexer(settings, object(), store)  # type: ignore[arg-type]

    hit = await indexer.fresh_document(
        "https://example.com/",
        900,
        now=datetime(2026, 9, 9, 12, 10, tzinfo=UTC),
    )

    assert hit == FreshIndexHit(
        document_id="doc-1",
        content_hash="a" * 64,
        fetched_at="2026-09-09T12:00:00+00:00",
        age_s=600.0,
    )
    assert store.ensure_calls == 1
    assert store.calls[0][0] == "POST"
    assert store.calls[0][1] == f"/{settings.opensearch_documents_index}/_search"
    body = store.calls[0][2]["json"]
    assert body["query"] == {"term": {"url": "https://example.com/"}}
    assert body["sort"] == [{"fetched_at": {"order": "desc"}}]

    boundary = await indexer.fresh_document(
        "https://example.com/",
        900,
        now=datetime(2026, 9, 9, 12, 15, tzinfo=UTC),
    )
    assert boundary is None


@pytest.mark.asyncio
async def test_phase4a_invalid_timestamp_or_lookup_failure_falls_through() -> None:
    settings = Settings(dense_enabled=False, opensearch_url="http://opensearch:9200")

    invalid_store = _LookupStore(_document_payload("2026-09-09T12:00:00"))
    invalid_indexer = Indexer(settings, object(), invalid_store)  # type: ignore[arg-type]
    assert (
        await invalid_indexer.fresh_document(
            "https://example.com/",
            900,
            now=datetime(2026, 9, 9, 12, 1, tzinfo=UTC),
        )
        is None
    )

    failing_store = _LookupStore(fail=True)
    failing_indexer = Indexer(settings, object(), failing_store)  # type: ignore[arg-type]
    assert (
        await failing_indexer.fresh_document(
            "https://example.com/",
            900,
            now=datetime(2026, 9, 9, 12, 1, tzinfo=UTC),
        )
        is None
    )


@pytest.mark.asyncio
async def test_phase4a_fresh_skip_preserves_fetch_and_bfs_link_discovery() -> None:
    class FakeFetcher:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def fetch_html(self, url: str) -> FetchResult:
            self.calls.append(url)
            if url == "https://example.com/":
                html = '<a href="/child">child</a>'
            else:
                html = "<p>child</p>"
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html=html,
                fetched_at="2026-09-09T12:30:00+00:00",
            )

    class FakeExtractor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            self.calls.append(fetched.final_url)
            return Extraction(
                title=fetched.final_url,
                markdown="# Child\n\nIndex me",
                canonical_url=fetched.final_url,
                extractor="fixture",
                quality=1.0,
                rendered=False,
            )

    class FakeIndexer:
        def __init__(self) -> None:
            self.lookup_calls: list[tuple[str, int]] = []
            self.index_calls: list[str] = []

        async def fresh_document(self, url: str, refresh_after_s: int):
            self.lookup_calls.append((url, refresh_after_s))
            if url == "https://example.com/":
                return FreshIndexHit(
                    document_id="doc-root",
                    content_hash="b" * 64,
                    fetched_at="2026-09-09T12:00:00+00:00",
                    age_s=1800.0,
                )
            return None

        async def index_extraction(
            self,
            fetched: FetchResult,
            _extraction: Extraction,
        ) -> IndexOutcome:
            self.index_calls.append(fetched.final_url)
            return IndexOutcome(
                url=fetched.final_url,
                indexed=True,
                document_id="doc-child",
                content_hash="c" * 64,
                chunks_indexed=1,
            )

    fetcher = FakeFetcher()
    extractor = FakeExtractor()
    indexer = FakeIndexer()
    crawler = Crawler(fetcher, extractor, indexer)  # type: ignore[arg-type]

    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=5,
        max_depth=1,
        same_origin=True,
        refresh_after_s=3600,
    )

    assert fetcher.calls == ["https://example.com/", "https://example.com/child"]
    assert indexer.lookup_calls == [
        ("https://example.com/", 3600),
        ("https://example.com/child", 3600),
    ]
    assert extractor.calls == ["https://example.com/child"]
    assert indexer.index_calls == ["https://example.com/child"]
    assert outcomes[0].indexed is False
    assert outcomes[0].freshness_skipped is True
    assert outcomes[0].document_id == "doc-root"
    assert outcomes[0].freshness_age_s == 1800.0
    assert outcomes[1].indexed is True
    assert outcomes[1].freshness_skipped is False


@pytest.mark.asyncio
async def test_phase4a_zero_refresh_does_not_invoke_freshness_lookup() -> None:
    class FakeFetcher:
        async def fetch_html(self, url: str) -> FetchResult:
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html="<p>body</p>",
                fetched_at="2026-09-09T12:30:00+00:00",
            )

    class FakeExtractor:
        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            return Extraction(
                title="Page",
                markdown="# Page\n\nIndex me",
                canonical_url=fetched.final_url,
                extractor="fixture",
                quality=1.0,
                rendered=False,
            )

    class FakeIndexer:
        async def fresh_document(self, _url: str, _refresh_after_s: int):
            raise AssertionError("zero refresh must not invoke freshness lookup")

        async def index_extraction(
            self,
            fetched: FetchResult,
            _extraction: Extraction,
        ) -> IndexOutcome:
            return IndexOutcome(
                url=fetched.final_url,
                indexed=True,
                document_id="doc-1",
                content_hash="d" * 64,
                chunks_indexed=1,
            )

    crawler = Crawler(FakeFetcher(), FakeExtractor(), FakeIndexer())  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
    )

    assert len(outcomes) == 1
    assert outcomes[0].indexed is True
    assert outcomes[0].freshness_skipped is False


@pytest.mark.asyncio
async def test_phase4a_api_reports_fresh_skip_count_and_forwards_window(monkeypatch) -> None:
    async def fake_crawl(**kwargs):
        assert kwargs["refresh_after_s"] == 3600
        return [
            CrawlOutcome(
                url="https://example.com/",
                depth=0,
                indexed=False,
                document_id="doc-1",
                content_hash="e" * 64,
                freshness_skipped=True,
                freshness_age_s=120.0,
            )
        ]

    monkeypatch.setattr(app_module.crawler, "crawl", fake_crawl)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/crawl",
            json={
                "seeds": ["https://example.com/"],
                "max_pages": 1,
                "max_depth": 0,
                "same_origin": True,
                "refresh_after_s": 3600,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["pages_visited"] == 1
    assert body["pages_indexed"] == 0
    assert body["pages_skipped_fresh"] == 1
    assert body["pages"][0]["freshness_skipped"] is True
    assert body["pages"][0]["freshness_age_s"] == 120.0
