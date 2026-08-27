from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from supracrawl.config import Settings

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3d_policy.json"


def _policy() -> dict[str, Any]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 3D policy must be a JSON object")
    return payload


async def _set_read_block(
    client: httpx.AsyncClient,
    index_name: str,
    blocked: bool,
) -> None:
    response = await client.put(
        f"/{index_name}/_settings",
        json={"index.blocks.read": blocked},
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"unable to set read block={blocked} on {index_name}: "
            f"HTTP {response.status_code} {response.text[:200]}"
        )


async def _hybrid(client: httpx.AsyncClient, query: str) -> httpx.Response:
    return await client.post(
        "/v1/search",
        json={"query": query, "limit": 5, "mode": "hybrid"},
    )


def _assert_degraded(response: httpx.Response, reason_fragment: str) -> None:
    if response.status_code != 200:
        raise RuntimeError(
            f"expected degraded HTTP 200, got {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("degraded response is not an object")
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "bm25":
        raise RuntimeError("hybrid failure did not degrade to BM25")
    if body.get("degraded") is not True:
        raise RuntimeError("hybrid failure did not report degraded=true")
    reason = body.get("degradation_reason")
    if not isinstance(reason, str) or reason_fragment not in reason:
        raise RuntimeError(
            f"degradation reason {reason!r} does not contain {reason_fragment!r}"
        )


async def _verify_hash_validation_failure(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
    query: str,
) -> None:
    index_name = settings.opensearch_documents_index
    await _set_read_block(opensearch, index_name, True)
    try:
        response = await _hybrid(api, query)
        _assert_degraded(response, "current-content validation")
    finally:
        await _set_read_block(opensearch, index_name, False)
    print("phase3d_hash_validation_failure_degrades=PASS")


async def _verify_lexical_failure(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
    query: str,
) -> None:
    index_name = settings.opensearch_chunks_index
    await _set_read_block(opensearch, index_name, True)
    try:
        response = await _hybrid(api, query)
        if response.status_code != 503:
            raise RuntimeError(
                f"lexical failure returned HTTP {response.status_code}, expected 503"
            )
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("detail"), str):
            raise RuntimeError("lexical 503 response is missing detail")
    finally:
        await _set_read_block(opensearch, index_name, False)
    print("phase3d_lexical_failure_503=PASS")


async def _verify_vector_failure(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
    query: str,
) -> None:
    index_name = settings.opensearch_vector_chunks_index
    response = await opensearch.delete(f"/{index_name}")
    if response.status_code not in {200, 404}:
        raise RuntimeError(
            f"unable to delete vector index: HTTP {response.status_code} {response.text[:200]}"
        )
    response = await opensearch.put(
        f"/{index_name}",
        json={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "dynamic": "strict",
                "properties": {"document_id": {"type": "keyword"}},
            },
        },
    )
    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"unable to create incompatible vector index: "
            f"HTTP {response.status_code} {response.text[:200]}"
        )

    response = await _hybrid(api, query)
    _assert_degraded(response, "vector path unavailable")
    print("phase3d_vector_failure_degrades=PASS")


async def _verify_dense_disabled(api: httpx.AsyncClient, query: str) -> None:
    response = await _hybrid(api, query)
    _assert_degraded(response, "disabled")
    print("phase3d_dense_disabled_degrades=PASS")


async def _run() -> None:
    policy = _policy()
    query = policy["frozen_spot_queries"]["exact_identifier"][0]["query"]
    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )
    dense_disabled_only = os.environ.get("PHASE3D_DENSE_DISABLED_ONLY") == "1"

    settings = Settings(opensearch_url=opensearch_url)
    async with (
        httpx.AsyncClient(base_url=api_url, timeout=60.0) as api,
        httpx.AsyncClient(base_url=opensearch_url, timeout=30.0) as opensearch,
    ):
        if dense_disabled_only:
            await _verify_dense_disabled(api, query)
            return

        await _verify_hash_validation_failure(api, opensearch, settings, query)
        await _verify_lexical_failure(api, opensearch, settings, query)
        await _verify_vector_failure(api, opensearch, settings, query)

    print("Phase 3D live fault matrix: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
