from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import supracrawl.app as app_module
import supracrawl.fetcher as fetcher_module
from supracrawl.config import Settings
from supracrawl.crawler import Crawler, CrawlOutcome
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult, HttpFetcher
from supracrawl.indexer import Indexer, IndexOutcome, RevalidationIndexHit
from supracrawl.models import CrawlRequest


def _revalidation_hit(*, age_s: float, etag: str | None = '"v1"') -> RevalidationIndexHit:
    return RevalidationIndexHit(
        document_id="doc-1",
        content_hash="a" * 64,
        fetched_at="2026-09-10T10:00:00+00:00",
        age_s=age_s,
        etag=etag,
        last_modified="Wed, 10 Sep 2026 10:00:00 GMT",
    )


def _extraction(url: str) -> Extraction:
    return Extraction(
        title="Page",
        markdown="# Page\n\nIndex me",
        canonical_url=url,
        extractor="fixture",
        quality=1.0,
        rendered=False,
    )


def test_phase4b_policy_matches_opt_in_leaf_contract() -> None:
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation/phase4b_conditional_revalidation_policy.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert policy["base_certified_sha"] == "bf3db918b123c93301ad92ec047717c7af6c7e01"
    assert policy["request_contract"] == {
        "field": "conditional_revalidate_leaves",
        "default": False,
        "requires_refresh_after_s_gt_zero": True,
        "activation_condition": (
            "conditional_revalidate_leaves and refresh_after_s > 0 and depth >= max_depth"
        ),
    }
    assert policy["constraints"]["phase4a_default_behavior_unchanged"] is True
    assert policy["constraints"]["no_non_leaf_304"] is True


def test_phase4b_request_defaults_to_disabled() -> None:
    request = CrawlRequest(seeds=["https://example.com/"])
    assert request.conditional_revalidate_leaves is False


@pytest.mark.asyncio
async def test_phase4b_fetcher_sends_validators_and_accepts_304(monkeypatch) -> None:
    async def allow_public_url(_url: str) -> None:
        return None

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            304,
            headers={
                "ETag": '"v2"',
                "Last-Modified": "Wed, 10 Sep 2026 11:00:00 GMT",
            },
        )

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def client_factory(**kwargs):
        return real_client(
            transport=transport,
            timeout=kwargs["timeout"],
            follow_redirects=kwargs["follow_redirects"],
            headers=kwargs["headers"],
        )

    monkeypatch.setattr(fetcher_module, "validate_public_url", allow_public_url)
    monkeypatch.setattr(fetcher_module.httpx, "AsyncClient", client_factory)

    fetcher = HttpFetcher(Settings(obey_robots_txt=False))
    result = await fetcher.fetch_html(
        "https://example.com/page",
        if_none_match='"v1"',
        if_modified_since="Wed, 10 Sep 2026 10:00:00 GMT",
    )

    assert len(requests) == 1
    assert requests[0].headers["if-none-match"] == '"v1"'
    assert requests[0].headers["if-modified-since"] == "Wed, 10 Sep 2026 10:00:00 GMT"
    assert result.status_code == 304
    assert result.not_modified is True
    assert result.html == ""
    assert result.etag == '"v2"'
    assert result.last_modified == "Wed, 10 Sep 2026 11:00:00 GMT"


@pytest.mark.asyncio
async def test_phase4b_fetcher_drops_validators_after_redirect(monkeypatch) -> None:
    async def allow_public_url(_url: str) -> None:
        return None

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/final"})
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html", "ETag": '"new"'},
            content=b"<p>final</p>",
        )

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def client_factory(**kwargs):
        return real_client(
            transport=transport,
            timeout=kwargs["timeout"],
            follow_redirects=kwargs["follow_redirects"],
            headers=kwargs["headers"],
        )

    monkeypatch.setattr(fetcher_module, "validate_public_url", allow_public_url)
    monkeypatch.setattr(fetcher_module.httpx, "AsyncClient", client_factory)

    fetcher = HttpFetcher(Settings(obey_robots_txt=False))
    result = await fetcher.fetch_html(
        "https://example.com/start",
        if_none_match='"old"',
        if_modified_since="Wed, 10 Sep 2026 10:00:00 GMT",
    )

    assert len(requests) == 2
    assert requests[0].headers["if-none-match"] == '"old"'
    assert "if-none-match" not in requests[1].headers
    assert "if-modified-since" not in requests[1].headers
    assert result.final_url == "https://example.com/final"
    assert result.not_modified is False
    assert result.etag == '"new"'


@pytest.mark.asyncio
async def test_phase4b_revalidation_lookup_returns_age_and_validators() -> None:
    class Store:
        def __init__(self) -> None:
            self.ensure_calls = 0
            self.calls: list[tuple[str, str, dict]] = []

        async def ensure_indices(self) -> None:
            self.ensure_calls += 1

        async def _request_with_index_recovery(self, method: str, path: str, **kwargs):
            self.calls.append((method, path, kwargs))
            return httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "document_id": "doc-1",
                                    "content_hash": "b" * 64,
                                    "fetched_at": "2026-09-10T10:00:00+00:00",
                                    "url": "https://example.com/page",
                                    "etag": '"v1"',
                                    "last_modified": "Wed, 10 Sep 2026 10:00:00 GMT",
                                }
                            }
                        ]
                    }
                },
            )

    store = Store()
    settings = Settings(dense_enabled=False, opensearch_url="http://opensearch:9200")
    indexer = Indexer(settings, object(), store)  # type: ignore[arg-type]

    hit = await indexer.revalidation_document(
        "https://example.com/page",
        now=datetime(2026, 9, 10, 12, 0, tzinfo=UTC),
    )

    assert hit == RevalidationIndexHit(
        document_id="doc-1",
        content_hash="b" * 64,
        fetched_at="2026-09-10T10:00:00+00:00",
        age_s=7200.0,
        etag='"v1"',
        last_modified="Wed, 10 Sep 2026 10:00:00 GMT",
    )
    assert hit.has_validator is True
    assert store.ensure_calls == 1
    body = store.calls[0][2]["json"]
    assert body["query"] == {"term": {"url": "https://example.com/page"}}
    assert "etag" in body["_source"]
    assert "last_modified" in body["_source"]


@pytest.mark.asyncio
async def test_phase4b_successful_index_persists_http_validators_best_effort() -> None:
    class Store:
        def __init__(self) -> None:
            self.mapping_calls: list[tuple[str, str, dict]] = []
            self.update_calls: list[tuple[str, str, dict]] = []

        async def index_document(self, **kwargs):
            return "doc-1", len(kwargs["chunks"])

        async def _request(self, method: str, path: str, **kwargs):
            self.mapping_calls.append((method, path, kwargs))
            return httpx.Response(200, json={"acknowledged": True})

        async def _request_with_index_recovery(self, method: str, path: str, **kwargs):
            self.update_calls.append((method, path, kwargs))
            return httpx.Response(200, json={"result": "updated"})

    store = Store()
    settings = Settings(dense_enabled=False, opensearch_url="http://opensearch:9200")
    indexer = Indexer(settings, object(), store)  # type: ignore[arg-type]
    fetched = FetchResult(
        fetch_url="https://example.com/page",
        final_url="https://example.com/page",
        status_code=200,
        content_type="text/html",
        html="<p>body</p>",
        fetched_at="2026-09-10T12:00:00+00:00",
        etag='"v1"',
        last_modified="Wed, 10 Sep 2026 12:00:00 GMT",
    )

    outcome = await indexer.index_extraction(fetched, _extraction(fetched.final_url))

    assert outcome.indexed is True
    assert store.mapping_calls[0][0:2] == (
        "PUT",
        f"/{settings.opensearch_documents_index}/_mapping",
    )
    mapping = store.mapping_calls[0][2]["json"]["properties"]
    assert mapping["etag"]["type"] == "keyword"
    assert mapping["last_modified"]["type"] == "keyword"
    assert store.update_calls[0][0] == "POST"
    assert store.update_calls[0][2]["json"]["doc"] == {
        "etag": '"v1"',
        "last_modified": "Wed, 10 Sep 2026 12:00:00 GMT",
    }


@pytest.mark.asyncio
async def test_phase4b_default_false_preserves_phase4a_fetch_path() -> None:
    class Fetcher:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_html(self, url: str) -> FetchResult:
            self.calls += 1
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html="<p>body</p>",
                fetched_at="2026-09-10T12:00:00+00:00",
            )

    class Extractor:
        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            return _extraction(fetched.final_url)

    class IndexerFixture:
        async def revalidation_document(self, _url: str):
            raise AssertionError("Phase 4B lookup must remain disabled by default")

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            return None

        async def index_extraction(self, fetched: FetchResult, _extraction: Extraction):
            return IndexOutcome(
                url=fetched.final_url,
                indexed=True,
                document_id="doc-1",
                content_hash="c" * 64,
                chunks_indexed=1,
            )

    fetcher = Fetcher()
    crawler = Crawler(fetcher, Extractor(), IndexerFixture())  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
        refresh_after_s=3600,
    )

    assert fetcher.calls == 1
    assert outcomes[0].indexed is True
    assert outcomes[0].network_fetch_skipped is False


@pytest.mark.asyncio
async def test_phase4b_requires_positive_refresh_window() -> None:
    class Fetcher:
        async def fetch_html(self, url: str) -> FetchResult:
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html="<p>body</p>",
                fetched_at="2026-09-10T12:00:00+00:00",
            )

    class Extractor:
        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            return _extraction(fetched.final_url)

    class IndexerFixture:
        async def revalidation_document(self, _url: str):
            raise AssertionError("zero refresh must disable Phase 4B lookup")

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            raise AssertionError("zero refresh must disable Phase 4A lookup")

        async def index_extraction(self, fetched: FetchResult, _extraction: Extraction):
            return IndexOutcome(url=fetched.final_url, indexed=True, chunks_indexed=1)

    crawler = Crawler(Fetcher(), Extractor(), IndexerFixture())  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
        refresh_after_s=0,
        conditional_revalidate_leaves=True,
    )
    assert outcomes[0].indexed is True


@pytest.mark.asyncio
async def test_phase4b_fresh_leaf_skips_network_extraction_and_indexing() -> None:
    class Fetcher:
        def __init__(self) -> None:
            self.admission_calls: list[str] = []

        async def ensure_fetch_allowed(self, url: str) -> None:
            self.admission_calls.append(url)

        async def fetch_html(self, _url: str, **_kwargs) -> FetchResult:
            raise AssertionError("fresh leaf must not perform a network fetch")

    class Extractor:
        async def extract_fetched(self, _fetched: FetchResult) -> Extraction:
            raise AssertionError("fresh leaf must not extract")

    class IndexerFixture:
        async def revalidation_document(self, _url: str):
            return _revalidation_hit(age_s=120.0)

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            raise AssertionError("pre-fetch fresh leaf should already have returned")

        async def index_extraction(self, _fetched: FetchResult, _extraction: Extraction):
            raise AssertionError("fresh leaf must not index")

    fetcher = Fetcher()
    crawler = Crawler(fetcher, Extractor(), IndexerFixture())  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
        refresh_after_s=3600,
        conditional_revalidate_leaves=True,
    )

    assert fetcher.admission_calls == ["https://example.com/"]
    assert outcomes == [
        CrawlOutcome(
            url="https://example.com/",
            depth=0,
            indexed=False,
            document_id="doc-1",
            content_hash="a" * 64,
            freshness_skipped=True,
            freshness_age_s=120.0,
            network_fetch_skipped=True,
        )
    ]


@pytest.mark.asyncio
async def test_phase4b_stale_leaf_304_touches_document_and_skips_indexing() -> None:
    class Fetcher:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def fetch_html(self, url: str, **kwargs) -> FetchResult:
            self.calls.append(kwargs)
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=304,
                content_type="",
                html="",
                fetched_at="2026-09-10T12:30:00+00:00",
                etag='"v1"',
                last_modified="Wed, 10 Sep 2026 10:00:00 GMT",
                not_modified=True,
            )

    class Extractor:
        async def extract_fetched(self, _fetched: FetchResult) -> Extraction:
            raise AssertionError("304 must not extract")

    class IndexerFixture:
        def __init__(self) -> None:
            self.touches = 0

        async def revalidation_document(self, _url: str):
            return _revalidation_hit(age_s=7200.0)

        async def touch_revalidated_document(self, hit, fetched):
            assert hit.document_id == "doc-1"
            assert fetched.status_code == 304
            self.touches += 1
            return True

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            raise AssertionError("304 must return before Phase 4A lookup")

        async def index_extraction(self, _fetched: FetchResult, _extraction: Extraction):
            raise AssertionError("304 must not index")

    fetcher = Fetcher()
    indexer = IndexerFixture()
    crawler = Crawler(fetcher, Extractor(), indexer)  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
        refresh_after_s=3600,
        conditional_revalidate_leaves=True,
    )

    assert fetcher.calls == [
        {
            "if_none_match": '"v1"',
            "if_modified_since": "Wed, 10 Sep 2026 10:00:00 GMT",
        }
    ]
    assert indexer.touches == 1
    assert outcomes[0].indexed is False
    assert outcomes[0].revalidated_not_modified is True
    assert outcomes[0].freshness_skipped is False
    assert outcomes[0].network_fetch_skipped is False


@pytest.mark.asyncio
async def test_phase4b_304_touch_failure_retries_unconditional_and_indexes() -> None:
    class Fetcher:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def fetch_html(self, url: str, **kwargs) -> FetchResult:
            self.calls.append(kwargs)
            if kwargs:
                return FetchResult(
                    fetch_url=url,
                    final_url=url,
                    status_code=304,
                    content_type="",
                    html="",
                    fetched_at="2026-09-10T12:30:00+00:00",
                    not_modified=True,
                )
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html="<p>body</p>",
                fetched_at="2026-09-10T12:31:00+00:00",
            )

    class Extractor:
        def __init__(self) -> None:
            self.calls = 0

        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            self.calls += 1
            return _extraction(fetched.final_url)

    class IndexerFixture:
        def __init__(self) -> None:
            self.index_calls = 0

        async def revalidation_document(self, _url: str):
            return _revalidation_hit(age_s=7200.0)

        async def touch_revalidated_document(self, _hit, _fetched):
            return False

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            return None

        async def index_extraction(self, fetched: FetchResult, _extraction: Extraction):
            self.index_calls += 1
            return IndexOutcome(
                url=fetched.final_url,
                indexed=True,
                document_id="doc-1",
                content_hash="d" * 64,
                chunks_indexed=1,
            )

    fetcher = Fetcher()
    extractor = Extractor()
    indexer = IndexerFixture()
    crawler = Crawler(fetcher, extractor, indexer)  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
        refresh_after_s=3600,
        conditional_revalidate_leaves=True,
    )

    assert len(fetcher.calls) == 2
    assert fetcher.calls[0]["if_none_match"] == '"v1"'
    assert fetcher.calls[1] == {}
    assert extractor.calls == 1
    assert indexer.index_calls == 1
    assert outcomes[0].indexed is True
    assert outcomes[0].revalidated_not_modified is False


@pytest.mark.asyncio
async def test_phase4b_conditional_200_uses_normal_extraction_and_indexing() -> None:
    class Fetcher:
        async def fetch_html(self, url: str, **kwargs) -> FetchResult:
            assert kwargs["if_none_match"] == '"v1"'
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html="<p>changed</p>",
                fetched_at="2026-09-10T12:30:00+00:00",
                etag='"v2"',
            )

    class Extractor:
        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            return _extraction(fetched.final_url)

    class IndexerFixture:
        async def revalidation_document(self, _url: str):
            return _revalidation_hit(age_s=7200.0)

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            return None

        async def index_extraction(self, fetched: FetchResult, _extraction: Extraction):
            assert fetched.etag == '"v2"'
            return IndexOutcome(url=fetched.final_url, indexed=True, chunks_indexed=1)

    crawler = Crawler(Fetcher(), Extractor(), IndexerFixture())  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=0,
        same_origin=True,
        refresh_after_s=3600,
        conditional_revalidate_leaves=True,
    )
    assert outcomes[0].indexed is True


@pytest.mark.asyncio
async def test_phase4b_non_leaf_preserves_phase4a_bfs_fetch_path() -> None:
    class Fetcher:
        async def fetch_html(self, url: str) -> FetchResult:
            return FetchResult(
                fetch_url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                html='<a href="/child">child</a>',
                fetched_at="2026-09-10T12:00:00+00:00",
            )

    class Extractor:
        async def extract_fetched(self, fetched: FetchResult) -> Extraction:
            return _extraction(fetched.final_url)

    class IndexerFixture:
        async def revalidation_document(self, _url: str):
            raise AssertionError("non-leaf must not use Phase 4B lookup")

        async def fresh_document(self, _url: str, _refresh_after_s: int):
            return None

        async def index_extraction(self, fetched: FetchResult, _extraction: Extraction):
            return IndexOutcome(url=fetched.final_url, indexed=True, chunks_indexed=1)

    crawler = Crawler(Fetcher(), Extractor(), IndexerFixture())  # type: ignore[arg-type]
    outcomes = await crawler.crawl(
        seeds=["https://example.com/"],
        max_pages=1,
        max_depth=1,
        same_origin=True,
        refresh_after_s=3600,
        conditional_revalidate_leaves=True,
    )
    assert outcomes[0].indexed is True


@pytest.mark.asyncio
async def test_phase4b_api_forwards_flag_and_reports_new_counts(monkeypatch) -> None:
    async def fake_crawl(**kwargs):
        assert kwargs["refresh_after_s"] == 3600
        assert kwargs["conditional_revalidate_leaves"] is True
        return [
            CrawlOutcome(
                url="https://example.com/fresh",
                depth=1,
                indexed=False,
                document_id="doc-fresh",
                content_hash="e" * 64,
                freshness_skipped=True,
                freshness_age_s=120.0,
                network_fetch_skipped=True,
            ),
            CrawlOutcome(
                url="https://example.com/revalidated",
                depth=1,
                indexed=False,
                document_id="doc-304",
                content_hash="f" * 64,
                revalidated_not_modified=True,
            ),
        ]

    monkeypatch.setattr(app_module.crawler, "crawl", fake_crawl)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/crawl",
            json={
                "seeds": ["https://example.com/"],
                "max_pages": 2,
                "max_depth": 1,
                "same_origin": True,
                "refresh_after_s": 3600,
                "conditional_revalidate_leaves": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["pages_visited"] == 2
    assert body["pages_indexed"] == 0
    assert body["pages_skipped_fresh"] == 1
    assert body["pages_network_skipped_fresh"] == 1
    assert body["pages_revalidated_not_modified"] == 1
    assert body["pages"][0]["network_fetch_skipped"] is True
    assert body["pages"][1]["revalidated_not_modified"] is True
