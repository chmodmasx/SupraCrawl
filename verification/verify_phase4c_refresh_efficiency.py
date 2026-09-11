from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

import supracrawl.fetcher as fetcher_module
from supracrawl.config import Settings
from supracrawl.crawler import Crawler
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult, HttpFetcher
from supracrawl.indexer import FreshIndexHit, IndexOutcome, RevalidationIndexHit

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation/phase4c_refresh_efficiency_policy.json"
OUTPUT_PATH = ROOT / "phase4c-refresh-efficiency-report.json"
URL = "https://phase4c.example/page"
DOCUMENT_ID = "phase4c-doc"
CONTENT_HASH = "c" * 64


@dataclass(slots=True)
class _Counters:
    target_gets: int = 0
    conditional_gets: int = 0
    unconditional_gets: int = 0
    response_body_bytes: int = 0
    extractions: int = 0
    indexes: int = 0
    touches: int = 0


class _Extractor:
    def __init__(self, counters: _Counters) -> None:
        self.counters = counters

    async def extract_fetched(self, fetched: FetchResult) -> Extraction:
        self.counters.extractions += 1
        return Extraction(
            title="Phase 4C fixture",
            markdown="# Phase 4C\n\nDeterministic refresh efficiency fixture.",
            canonical_url=fetched.final_url,
            extractor="phase4c-fixture",
            quality=1.0,
            rendered=False,
        )


class _Indexer:
    def __init__(
        self,
        counters: _Counters,
        *,
        age_s: float,
        etag: str,
        touch_success: bool,
    ) -> None:
        self.counters = counters
        self.age_s = age_s
        self.etag = etag
        self.touch_success = touch_success

    async def revalidation_document(self, url: str) -> RevalidationIndexHit:
        assert url == URL
        return RevalidationIndexHit(
            document_id=DOCUMENT_ID,
            content_hash=CONTENT_HASH,
            fetched_at="2026-09-10T12:00:00+00:00",
            age_s=self.age_s,
            etag=self.etag,
            last_modified=None,
        )

    async def fresh_document(self, url: str, refresh_after_s: int) -> FreshIndexHit | None:
        assert url == URL
        if self.age_s >= refresh_after_s:
            return None
        return FreshIndexHit(
            document_id=DOCUMENT_ID,
            content_hash=CONTENT_HASH,
            fetched_at="2026-09-10T12:00:00+00:00",
            age_s=self.age_s,
        )

    async def touch_revalidated_document(
        self,
        hit: RevalidationIndexHit,
        fetched: FetchResult,
    ) -> bool:
        assert hit.document_id == DOCUMENT_ID
        assert fetched.not_modified is True
        self.counters.touches += 1
        return self.touch_success

    async def index_extraction(
        self,
        fetched: FetchResult,
        _extraction: Extraction,
    ) -> IndexOutcome:
        self.counters.indexes += 1
        return IndexOutcome(
            url=fetched.final_url,
            indexed=True,
            document_id=DOCUMENT_ID,
            content_hash=CONTENT_HASH,
            chunks_indexed=1,
        )


def _fixture_body(size: int) -> bytes:
    prefix = b"<html><body><p>"
    suffix = b"</p></body></html>"
    filler_size = size - len(prefix) - len(suffix)
    if filler_size < 0:
        raise AssertionError("Phase 4C fixture size is too small")
    body = prefix + (b"x" * filler_size) + suffix
    assert len(body) == size
    return body


async def _measure_scenario(
    *,
    body: bytes,
    etag: str,
    refresh_after_s: int,
    age_s: float,
    conditional_revalidate_leaves: bool,
    conditional_returns_304: bool,
    touch_success: bool,
) -> dict[str, Any]:
    counters = _Counters()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == URL
        counters.target_gets += 1
        conditional = request.headers.get("if-none-match") == etag
        if conditional:
            counters.conditional_gets += 1
        else:
            counters.unconditional_gets += 1

        if conditional and conditional_returns_304:
            return httpx.Response(304, headers={"ETag": etag})

        counters.response_body_bytes += len(body)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html", "ETag": etag},
            content=body,
        )

    async def allow_public_url(_url: str) -> None:
        return None

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    real_validate_public_url = fetcher_module.validate_public_url

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(
            transport=transport,
            timeout=kwargs["timeout"],
            follow_redirects=kwargs["follow_redirects"],
            headers=kwargs["headers"],
        )

    fetcher_module.validate_public_url = allow_public_url
    fetcher_module.httpx.AsyncClient = client_factory  # type: ignore[assignment]
    try:
        fetcher = HttpFetcher(
            Settings(
                obey_robots_txt=False,
                max_html_bytes=max(len(body) + 1024, 16_384),
            )
        )
        indexer = _Indexer(
            counters,
            age_s=age_s,
            etag=etag,
            touch_success=touch_success,
        )
        crawler = Crawler(fetcher, _Extractor(counters), indexer)  # type: ignore[arg-type]
        outcomes = await crawler.crawl(
            seeds=[URL],
            max_pages=1,
            max_depth=0,
            same_origin=True,
            refresh_after_s=refresh_after_s,
            conditional_revalidate_leaves=conditional_revalidate_leaves,
        )
    finally:
        fetcher_module.httpx.AsyncClient = real_client
        fetcher_module.validate_public_url = real_validate_public_url

    assert len(outcomes) == 1
    outcome = outcomes[0]
    return {
        "target_gets": counters.target_gets,
        "conditional_gets": counters.conditional_gets,
        "unconditional_gets": counters.unconditional_gets,
        "response_body_bytes": counters.response_body_bytes,
        "extractions": counters.extractions,
        "indexes": counters.indexes,
        "touches": counters.touches,
        "freshness_skipped": outcome.freshness_skipped,
        "network_fetch_skipped": outcome.network_fetch_skipped,
        "revalidated_not_modified": outcome.revalidated_not_modified,
        "indexed": outcome.indexed,
        "error": outcome.error,
    }


def _assert_scenario_contract(
    name: str,
    measured: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    for key, value in expected.items():
        if not key.startswith("expected_"):
            continue
        measured_key = key.removeprefix("expected_")
        actual = measured.get(measured_key)
        assert actual == value, (
            f"{name}: {measured_key}={actual!r}, expected {value!r}"
        )
    assert measured["error"] is None, f"{name}: unexpected crawl error {measured['error']!r}"


def _reduction_percent(baseline: int, optimized: int) -> float:
    if baseline <= 0:
        raise AssertionError("Phase 4C baseline must be positive")
    return round(((baseline - optimized) / baseline) * 100.0, 3)


async def _run(policy: dict[str, Any]) -> dict[str, Any]:
    fixture = policy["fixture"]
    body = _fixture_body(int(fixture["html_body_bytes"]))
    etag = str(fixture["etag"])
    refresh_after_s = int(fixture["refresh_after_s"])

    inputs = {
        "phase4a_fresh": {
            "age_s": 120.0,
            "conditional_revalidate_leaves": False,
            "conditional_returns_304": False,
            "touch_success": True,
        },
        "phase4b_fresh_leaf": {
            "age_s": 120.0,
            "conditional_revalidate_leaves": True,
            "conditional_returns_304": False,
            "touch_success": True,
        },
        "phase4a_stale_unchanged": {
            "age_s": 7200.0,
            "conditional_revalidate_leaves": False,
            "conditional_returns_304": False,
            "touch_success": True,
        },
        "phase4b_stale_304": {
            "age_s": 7200.0,
            "conditional_revalidate_leaves": True,
            "conditional_returns_304": True,
            "touch_success": True,
        },
        "phase4b_304_touch_failure": {
            "age_s": 7200.0,
            "conditional_revalidate_leaves": True,
            "conditional_returns_304": True,
            "touch_success": False,
        },
    }

    measurements: dict[str, dict[str, Any]] = {}
    for name, scenario_input in inputs.items():
        measured = await _measure_scenario(
            body=body,
            etag=etag,
            refresh_after_s=refresh_after_s,
            **scenario_input,
        )
        _assert_scenario_contract(name, measured, policy["scenarios"][name])
        measurements[name] = measured

    fresh_base = measurements["phase4a_fresh"]
    fresh_opt = measurements["phase4b_fresh_leaf"]
    stale_base = measurements["phase4a_stale_unchanged"]
    stale_opt = measurements["phase4b_stale_304"]

    reductions = {
        "fresh_leaf_target_get_reduction_percent": _reduction_percent(
            fresh_base["target_gets"], fresh_opt["target_gets"]
        ),
        "fresh_leaf_body_byte_reduction_percent": _reduction_percent(
            fresh_base["response_body_bytes"], fresh_opt["response_body_bytes"]
        ),
        "stale_304_body_byte_reduction_percent": _reduction_percent(
            stale_base["response_body_bytes"], stale_opt["response_body_bytes"]
        ),
        "stale_304_extraction_reduction_percent": _reduction_percent(
            stale_base["extractions"], stale_opt["extractions"]
        ),
        "stale_304_index_reduction_percent": _reduction_percent(
            stale_base["indexes"], stale_opt["indexes"]
        ),
    }

    acceptance = policy["acceptance"]
    for key, actual in reductions.items():
        assert actual == float(acceptance[key]), (
            f"Phase 4C reduction {key}={actual}, expected {acceptance[key]}"
        )

    fallback = measurements["phase4b_304_touch_failure"]
    if acceptance["touch_failure_must_restore_unconditional_full_path"]:
        assert fallback["conditional_gets"] == 1
        assert fallback["unconditional_gets"] == 1
        assert fallback["response_body_bytes"] == len(body)
        assert fallback["extractions"] == 1
        assert fallback["indexes"] == 1
        assert fallback["revalidated_not_modified"] is False

    return {
        "schema_version": 1,
        "phase": "4C",
        "gate": policy["gate"],
        "decision": "PASS_REFRESH_EFFICIENCY_GATE",
        "fixture_body_bytes": len(body),
        "measurements": measurements,
        "reductions": reductions,
    }


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["base_certified_sha"] == "542820dfb5fefe05988ff57f60a81c8d0f395698"
    assert policy["constraints"]["production_code_unchanged"] is True
    report = asyncio.run(_run(policy))
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
