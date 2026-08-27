#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.getenv("SUPRACRAWL_VERIFY_URL", "http://127.0.0.1:8080").rstrip("/")
OPENSEARCH_URL = os.getenv("SUPRACRAWL_VERIFY_OPENSEARCH_URL", "http://127.0.0.1:9200").rstrip("/")
DOCUMENTS_INDEX = "supracrawl-documents-v1"
CHUNKS_INDEX = "supracrawl-chunks-v1"


def _json_request(base_url: str, path: str, payload: dict | None = None, method: str | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers=headers,
        method=method or ("GET" if body is None else "POST"),
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed


def _api(path: str, payload: dict | None = None) -> dict:
    status, body = _json_request(BASE_URL, path, payload)
    if status >= 400:
        raise RuntimeError(f"API {path} failed with HTTP {status}: {body}")
    return body


def _opensearch(path: str, payload: dict | None = None, method: str | None = None) -> dict:
    status, body = _json_request(OPENSEARCH_URL, path, payload, method=method)
    if status >= 400:
        raise RuntimeError(f"OpenSearch {path} failed with HTTP {status}: {body}")
    return body


def _compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True)


def _wait_api() -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            status, body = _json_request(BASE_URL, "/v1/health")
            if status == 200 and body.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("SupraCrawl API did not become healthy")


def _wait_opensearch() -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            status, body = _json_request(OPENSEARCH_URL, "/_cluster/health")
            if status == 200 and body.get("status") in {"green", "yellow"}:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("OpenSearch did not become healthy")


def _search(query: str, limit: int = 5) -> list[dict]:
    response = _api("/v1/search", {"query": query, "limit": limit})
    assert response.get("success") is True, response
    results = response.get("results")
    assert isinstance(results, list), response
    return results


def main() -> None:
    _wait_opensearch()
    _wait_api()

    # Seed a small real corpus through the public indexing API.
    indexed = _api(
        "/v1/index",
        {
            "urls": [
                "https://example.com/",
                "https://quotes.toscrape.com/",
            ]
        },
    )
    assert indexed.get("success") is True, indexed
    items = indexed.get("items")
    assert isinstance(items, list) and len(items) == 2, indexed
    assert all(item.get("indexed") is True for item in items), indexed
    assert all((item.get("chunks_indexed") or 0) > 0 for item in items), indexed

    example_item = items[0]
    example_doc_id = example_item.get("document_id")
    assert isinstance(example_doc_id, str) and example_doc_id, example_item

    # BM25 should retrieve the exact Example Domain language above unrelated quotes.
    results = _search("illustrative examples in documents", limit=5)
    assert results, results
    assert urllib.parse.urlsplit(results[0]["url"]).hostname == "example.com", results
    assert results[0]["position"] == 1, results
    assert "illustrative" in results[0]["description"].lower(), results[0]

    documents_count = _opensearch(f"/{DOCUMENTS_INDEX}/_count").get("count")
    chunks_count = _opensearch(f"/{CHUNKS_INDEX}/_count").get("count")
    assert isinstance(documents_count, int) and documents_count >= 2, documents_count
    assert isinstance(chunks_count, int) and chunks_count >= 2, chunks_count

    # Inject an explicitly stale chunk, then reindex the page and verify cleanup.
    stale_chunk = {
        "document_id": example_doc_id,
        "content_hash": "stale-fixture",
        "url": "https://example.com/",
        "title": "Stale fixture",
        "section_path": "obsolete",
        "text": "obsolete sentinel text that must disappear after reindexing",
        "ordinal": 999,
        "approx_tokens": 12,
        "fetched_at": "2026-08-27T12:00:00+00:00",
    }
    _opensearch(
        f"/{CHUNKS_INDEX}/_doc/stale-phase2-fixture?refresh=true",
        stale_chunk,
        method="PUT",
    )
    stale_before = _opensearch(
        f"/{CHUNKS_INDEX}/_count",
        {"query": {"term": {"content_hash": "stale-fixture"}}},
    ).get("count")
    assert stale_before == 1, stale_before

    reindexed = _api("/v1/index", {"urls": ["https://example.com/"]})
    assert reindexed["items"][0]["indexed"] is True, reindexed
    stale_after = _opensearch(
        f"/{CHUNKS_INDEX}/_count",
        {"query": {"term": {"content_hash": "stale-fixture"}}},
    ).get("count")
    assert stale_after == 0, stale_after

    # Bounded crawler must follow same-origin links without escaping the site.
    crawled = _api(
        "/v1/crawl",
        {
            "seeds": ["https://quotes.toscrape.com/"],
            "max_pages": 3,
            "max_depth": 1,
            "same_origin": True,
        },
    )
    assert crawled.get("pages_visited") == 3, crawled
    assert crawled.get("pages_indexed") == 3, crawled
    for page in crawled["pages"]:
        assert urllib.parse.urlsplit(page["url"]).hostname == "quotes.toscrape.com", page
        assert page.get("indexed") is True, page

    # Search must fail honestly while OpenSearch is down and recover after restart.
    _compose("stop", "opensearch")
    status, failure = _json_request(
        BASE_URL,
        "/v1/search",
        {"query": "illustrative examples", "limit": 5},
    )
    assert status == 503, (status, failure)
    assert "OpenSearch" in str(failure), failure

    _compose("start", "opensearch")
    _wait_opensearch()
    recovered = _search("illustrative examples in documents", limit=5)
    assert recovered and urllib.parse.urlsplit(recovered[0]["url"]).hostname == "example.com"

    print("Phase 2 live verification: PASS")


if __name__ == "__main__":
    main()
