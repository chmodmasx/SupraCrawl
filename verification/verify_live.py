#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

BASE_URL = os.getenv("SUPRACRAWL_VERIFY_URL", "http://127.0.0.1:8080").rstrip("/")


def _request_json(path: str, payload: dict | None = None, attempts: int = 3) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            f"{BASE_URL}{path}",
            data=body,
            headers=headers,
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt * 1.5)
    raise RuntimeError(f"request failed after {attempts} attempts: {last_error}")


def _extract(
    urls: list[str],
    *,
    query: str | None = None,
    max_context_tokens: int | None = None,
    force_refresh: bool = False,
    attempts: int = 3,
) -> dict:
    payload: dict = {"urls": urls, "force_refresh": force_refresh}
    if query is not None:
        payload["query"] = query
    if max_context_tokens is not None:
        payload["max_context_tokens"] = max_context_tokens
    return _request_json("/v1/extract", payload, attempts=attempts)


def _one_document(response: dict) -> dict:
    assert response.get("success") is True, response
    documents = response.get("documents")
    assert isinstance(documents, list) and len(documents) == 1, response
    return documents[0]


def _assert_successful_document(document: dict, *, min_chars: int = 40) -> None:
    assert not document.get("error"), document
    assert len(document.get("content") or "") >= min_chars, document
    metadata = document.get("metadata")
    assert isinstance(metadata, dict), document
    assert metadata.get("http_status") == 200, document
    assert metadata.get("content_type") in {"text/html", "application/xhtml+xml"}, document
    assert metadata.get("content_hash"), document


def _compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], check=True)


def _wait_api() -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            health = _request_json("/v1/health", attempts=1)
            if health.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("SupraCrawl API did not become healthy")


def _wait_worker() -> None:
    script = (
        "fetch('http://127.0.0.1:3000/health')"
        ".then(r=>{if(!r.ok)process.exit(1);return r.json()})"
        ".then(j=>{if(j.status!=='ok')process.exit(1)})"
        ".catch(()=>process.exit(1))"
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "exec", "supracrawl-extractor", "node", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("extractor worker did not become healthy")


def _verify_worker_ssrf_guard() -> None:
    script = """
const r = await fetch('http://127.0.0.1:3000/render-extract', {
  method: 'POST',
  headers: {'content-type': 'application/json'},
  body: JSON.stringify({url: 'http://127.0.0.1:3000/health'})
});
const body = await r.json();
if (r.status !== 422 || !String(body.error || '').match(/private|local|routable/i)) {
  console.error(r.status, body);
  process.exit(1);
}
""".strip()
    subprocess.run(
        ["docker", "exec", "supracrawl-extractor", "node", "--input-type=module", "-e", script],
        check=True,
    )


def main() -> None:
    _wait_api()
    health = _request_json("/v1/health")
    assert health["status"] == "ok", health

    # Stable public HTML extraction and provenance.
    first = _one_document(_extract(["https://example.com/"], force_refresh=True))
    _assert_successful_document(first)
    assert "Example Domain" in (first.get("title") or first.get("content") or ""), first

    # Redis cache: identical request without force_refresh must preserve fetch time/hash.
    cached = _one_document(_extract(["https://example.com/"]))
    _assert_successful_document(cached)
    assert cached["metadata"]["fetched_at"] == first["metadata"]["fetched_at"]
    assert cached["metadata"]["content_hash"] == first["metadata"]["content_hash"]

    # force_refresh must bypass the cached document.
    time.sleep(0.05)
    refreshed = _one_document(_extract(["https://example.com/"], force_refresh=True))
    _assert_successful_document(refreshed)
    assert refreshed["metadata"]["fetched_at"] != first["metadata"]["fetched_at"]

    # API-level SSRF rejection and mixed batch semantics.
    unsafe = _one_document(_extract(["http://127.0.0.1:8080/"], force_refresh=True))
    assert str(unsafe.get("error") or "").startswith("Unsafe URL:"), unsafe

    mixed = _extract(
        ["https://example.com/", "http://169.254.169.254/latest/meta-data/"],
        force_refresh=True,
    )["documents"]
    assert len(mixed) == 2, mixed
    _assert_successful_document(mixed[0])
    assert str(mixed[1].get("error") or "").startswith("Unsafe URL:"), mixed[1]

    # Hard LLM context budget.
    budgeted = _one_document(
        _extract(
            ["https://es.wikipedia.org/wiki/Internet"],
            query="historia protocolo red Internet",
            max_context_tokens=128,
            force_refresh=True,
        )
    )
    _assert_successful_document(budgeted, min_chars=20)
    assert budgeted["metadata"]["context_tokens_approx"] <= 128, budgeted

    # Multilingual extraction: Latin and non-Latin scripts.
    spanish = _one_document(
        _extract(
            ["https://es.wikipedia.org/wiki/Internet"],
            query="Internet red mundial",
            force_refresh=True,
        )
    )
    _assert_successful_document(spanish, min_chars=200)
    assert "Internet" in spanish["content"], spanish

    japanese = _one_document(
        _extract(
            ["https://ja.wikipedia.org/wiki/インターネット"],
            force_refresh=True,
        )
    )
    _assert_successful_document(japanese, min_chars=200)
    assert any(ord(char) > 0x3000 for char in japanese["content"]), japanese

    # Real JavaScript-only page must trigger Playwright and produce rendered content.
    rendered = _one_document(
        _extract(["https://quotes.toscrape.com/js/"], force_refresh=True, attempts=4)
    )
    _assert_successful_document(rendered, min_chars=300)
    assert rendered["metadata"]["rendered"] is True, rendered
    assert "playwright" in rendered["metadata"]["extractor"], rendered

    # Worker-level SSRF guard is independent of the Python fetcher.
    _verify_worker_ssrf_guard()

    # Readability worker outage must degrade to Trafilatura, not fail the API.
    _compose("stop", "extractor-worker")
    fallback = _one_document(_extract(["https://example.com/"], force_refresh=True))
    _assert_successful_document(fallback)
    assert fallback["metadata"]["extractor"] == "trafilatura", fallback
    assert fallback["metadata"]["rendered"] is False, fallback
    _compose("start", "extractor-worker")
    _wait_worker()

    # Redis outage must fail open.
    _compose("stop", "redis")
    no_cache = _one_document(_extract(["https://example.com/"], force_refresh=True))
    _assert_successful_document(no_cache)
    _compose("start", "redis")

    # Multiple simultaneous API requests must all complete successfully.
    def concurrent_request(_index: int) -> dict:
        return _one_document(_extract(["https://example.com/"], force_refresh=True))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        concurrent_documents = list(executor.map(concurrent_request, range(4)))
    for document in concurrent_documents:
        _assert_successful_document(document)

    _wait_api()
    print("Phase 1 live verification: PASS")


if __name__ == "__main__":
    main()
