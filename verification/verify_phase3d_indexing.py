from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from supracrawl.config import Settings
from supracrawl.search import document_id

STALE_URL = "https://benchmark.supracrawl.local/phase3d-stale-vector"
EXAMPLE_URL = "https://example.com/"
STALE_QUERY = "antique pineapple semaphore knowledge"


def _hits(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("OpenSearch response is not an object")
    hits = payload.get("hits")
    if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
        raise RuntimeError("OpenSearch response has no hits list")
    return [hit for hit in hits["hits"] if isinstance(hit, dict)]


async def _vector_sources(
    client: httpx.AsyncClient,
    settings: Settings,
    doc_id: str,
) -> list[dict[str, Any]]:
    response = await client.post(
        f"/{settings.opensearch_vector_chunks_index}/_search",
        json={
            "size": 100,
            "_source": ["document_id", "content_hash", "url"],
            "query": {"term": {"document_id": doc_id}},
        },
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"vector inspection returned HTTP {response.status_code}: {response.text[:300]}"
        )
    sources: list[dict[str, Any]] = []
    for hit in _hits(response.json()):
        source = hit.get("_source")
        if isinstance(source, dict):
            sources.append(source)
    return sources


async def _current_hash(
    client: httpx.AsyncClient,
    settings: Settings,
    doc_id: str,
) -> str:
    response = await client.get(f"/{settings.opensearch_documents_index}/_doc/{doc_id}")
    if response.status_code != 200:
        raise RuntimeError(
            f"current document lookup returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("_source"), dict):
        raise RuntimeError("current document lookup returned invalid JSON")
    content_hash = body["_source"].get("content_hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise RuntimeError("current document has no content hash")
    return content_hash


async def _verify_physical_stale_vector(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
) -> None:
    doc_id = document_id(STALE_URL)
    current_hash = await _current_hash(opensearch, settings, doc_id)
    sources = await _vector_sources(opensearch, settings, doc_id)
    if not sources:
        raise RuntimeError("stale-vector fixture has no physical vector document")

    vector_hashes = {
        source.get("content_hash")
        for source in sources
        if isinstance(source.get("content_hash"), str)
    }
    if current_hash in vector_hashes:
        raise RuntimeError(
            "stale-vector fixture unexpectedly contains a vector for current lexical content"
        )
    if not vector_hashes:
        raise RuntimeError("stale-vector fixture has no vector content hash")

    response = await api.post(
        "/v1/search",
        json={"query": STALE_QUERY, "limit": 10, "mode": "hybrid"},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"stale-vector hybrid search returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("stale-vector hybrid search returned invalid JSON")
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "hybrid":
        raise RuntimeError("stale-vector guard passed only by degrading out of hybrid mode")
    if body.get("degraded") is not False:
        raise RuntimeError("stale-vector hybrid search unexpectedly reported degradation")
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("stale-vector hybrid search returned no results list")
    returned_urls = {
        item.get("url")
        for item in results
        if isinstance(item, dict) and isinstance(item.get("url"), str)
    }
    if STALE_URL in returned_urls:
        raise RuntimeError("physical stale vector leaked through current-content hash validation")

    print("phase3d_physical_stale_vector_present=PASS")
    print("phase3d_stale_vector_rejected_while_hybrid_active=PASS")


async def _verify_production_index_endpoint(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
) -> None:
    response = await api.post("/v1/index", json={"urls": [EXAMPLE_URL]})
    if response.status_code != 200:
        raise RuntimeError(
            f"production index endpoint returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict) or body.get("success") is not True:
        raise RuntimeError("production index endpoint did not return success")
    items = body.get("items")
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise RuntimeError("production index endpoint returned an invalid item list")
    item = items[0]
    if item.get("indexed") is not True:
        raise RuntimeError(f"production lexical indexing failed: {item}")
    if item.get("vector_indexed") is not True:
        raise RuntimeError(f"production vector indexing failed: {item}")
    vector_chunks = item.get("vector_chunks_indexed")
    if not isinstance(vector_chunks, int) or vector_chunks <= 0:
        raise RuntimeError("production vector indexing wrote no vector chunks")
    doc_id = item.get("document_id")
    content_hash = item.get("content_hash")
    if not isinstance(doc_id, str) or not isinstance(content_hash, str):
        raise RuntimeError("production index response is missing identity provenance")

    sources = await _vector_sources(opensearch, settings, doc_id)
    if not sources:
        raise RuntimeError("production index response claimed vectors but none exist")
    if not any(source.get("content_hash") == content_hash for source in sources):
        raise RuntimeError("production vector chunks do not match the indexed content hash")

    search = await api.post(
        "/v1/search",
        json={"query": "Example Domain", "limit": 10, "mode": "hybrid"},
    )
    if search.status_code != 200:
        raise RuntimeError(
            f"post-index hybrid search returned HTTP {search.status_code}: {search.text[:300]}"
        )
    search_body = search.json()
    if not isinstance(search_body, dict) or search_body.get("mode_used") != "hybrid":
        raise RuntimeError("post-index search did not use hybrid retrieval")
    results = search_body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("post-index search returned no results list")
    urls = [item.get("url") for item in results if isinstance(item, dict)]
    if not any(
        isinstance(url, str) and url.rstrip("/") == EXAMPLE_URL.rstrip("/")
        for url in urls
    ):
        raise RuntimeError("production-indexed Example Domain was not retrievable")

    print("phase3d_production_index_endpoint_vector_write=PASS")
    print("phase3d_production_indexed_document_retrievable=PASS")


async def _run() -> None:
    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )
    settings = Settings(opensearch_url=opensearch_url)

    async with (
        httpx.AsyncClient(base_url=api_url, timeout=180.0) as api,
        httpx.AsyncClient(base_url=opensearch_url, timeout=30.0) as opensearch,
    ):
        await _verify_physical_stale_vector(api, opensearch, settings)
        await _verify_production_index_endpoint(api, opensearch, settings)

    print("Phase 3D production indexing verification: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
