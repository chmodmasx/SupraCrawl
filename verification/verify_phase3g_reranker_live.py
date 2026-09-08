from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any

import httpx
from verify_real_vector_retrieval import _clear_indices, _seed_corpus_with_vectors
from verify_retrieval_baseline import _load_jsonl, _validate_fixture

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder
from supracrawl.reranking import (
    RERANKER_MODEL_REPO,
    RERANKER_PROTECTED_TOP_K,
    RERANKER_REVISION,
    RERANKER_STRATEGY,
)
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3d_policy.json"
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
EXACT_CORPUS_PATH = ROOT / "evaluation" / "phase3c_exact_corpus.jsonl"
EXACT_QUERIES_PATH = ROOT / "evaluation" / "phase3c_exact_queries.jsonl"
RERANKER_KEYS = {
    "reranker_model",
    "reranker_revision",
    "reranker_strategy",
    "reranker_score",
    "first_stage_position",
}


def _load_policy() -> dict[str, Any]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 3D policy must be a JSON object")
    return payload


def _results(body: dict[str, Any]) -> list[dict[str, Any]]:
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("search response has no results list")
    if not all(isinstance(result, dict) for result in results):
        raise RuntimeError("search response contains an invalid result")
    return results


def _urls(body: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for result in _results(body):
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("search result has no URL")
        urls.append(url)
    return urls


def _assert_hybrid_common(body: dict[str, Any]) -> None:
    if body.get("success") is not True:
        raise RuntimeError("hybrid response is not successful")
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "hybrid":
        raise RuntimeError("hybrid response did not remain hybrid")
    if body.get("degraded") is not False or body.get("degradation_reason") is not None:
        raise RuntimeError("hybrid response unexpectedly degraded")
    results = _results(body)
    positions = [result.get("position") for result in results]
    if positions != list(range(1, len(results) + 1)):
        raise RuntimeError("search result positions are not contiguous")


def _assert_disabled(body: dict[str, Any]) -> None:
    _assert_hybrid_common(body)
    if body.get("reranker_enabled") is not False:
        raise RuntimeError("baseline API unexpectedly reports reranker enabled")
    if body.get("reranker_used") is not False:
        raise RuntimeError("baseline API unexpectedly reports reranker used")
    if body.get("reranker_degraded") is not False:
        raise RuntimeError("baseline API unexpectedly reports reranker degradation")
    if body.get("reranker_degradation_reason") is not None:
        raise RuntimeError("baseline API unexpectedly reports reranker failure reason")
    for result in _results(body):
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("baseline result has no metadata")
        if RERANKER_KEYS.intersection(metadata):
            raise RuntimeError("baseline result unexpectedly contains reranker provenance")


def _assert_enabled(baseline: dict[str, Any], enabled: dict[str, Any]) -> None:
    _assert_hybrid_common(enabled)
    if enabled.get("reranker_enabled") is not True:
        raise RuntimeError("canary API did not report reranker enabled")
    if enabled.get("reranker_used") is not True:
        raise RuntimeError("canary API did not report reranker use")
    if enabled.get("reranker_degraded") is not False:
        raise RuntimeError("canary API reported reranker degradation")
    if enabled.get("reranker_degradation_reason") is not None:
        raise RuntimeError("canary API reported an unexpected reranker failure reason")

    baseline_results = _results(baseline)
    enabled_results = _results(enabled)
    if len(baseline_results) != 10 or len(enabled_results) != 10:
        raise RuntimeError("live reranker gate requires exactly ten candidates")

    baseline_urls = _urls(baseline)
    enabled_urls = _urls(enabled)
    if set(enabled_urls) != set(baseline_urls):
        raise RuntimeError("reranker changed frozen top-10 candidate membership")
    protected = RERANKER_PROTECTED_TOP_K
    if set(enabled_urls[:protected]) != set(baseline_urls[:protected]):
        raise RuntimeError("reranker changed protected top-5 membership")

    baseline_by_url = {result["url"]: result for result in baseline_results}
    first_stage_positions: set[int] = set()
    for result in enabled_results:
        url = result["url"]
        baseline_result = baseline_by_url[url]
        baseline_score = baseline_result.get("score")
        enabled_score = result.get("score")
        if baseline_score is None or enabled_score is None:
            raise RuntimeError("hybrid result is missing its first-stage RRF score")
        if not math.isclose(
            float(enabled_score),
            float(baseline_score),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("reranker changed the certified first-stage RRF score")

        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("reranked result has no metadata")
        if metadata.get("reranker_model") != RERANKER_MODEL_REPO:
            raise RuntimeError("live reranker model provenance changed")
        if metadata.get("reranker_revision") != RERANKER_REVISION:
            raise RuntimeError("live reranker revision provenance changed")
        if metadata.get("reranker_strategy") != RERANKER_STRATEGY:
            raise RuntimeError("live reranker strategy provenance changed")
        score = metadata.get("reranker_score")
        if score is None or not math.isfinite(float(score)):
            raise RuntimeError("live reranker score is missing or non-finite")
        first_stage_position = int(metadata.get("first_stage_position", 0))
        expected_position = baseline_urls.index(url) + 1
        if first_stage_position != expected_position:
            raise RuntimeError("live reranker first-stage position provenance changed")
        first_stage_positions.add(first_stage_position)

    if first_stage_positions != set(range(1, 11)):
        raise RuntimeError("live reranker first-stage provenance is incomplete")


async def _post_search(
    client: httpx.AsyncClient,
    *,
    query: str,
    limit: int,
    mode: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "limit": limit}
    if mode is not None:
        payload["mode"] = mode
    response = await client.post("/v1/search", json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"search API returned HTTP {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("search API returned non-object JSON")
    return body


async def _verify_bm25_bypass(client: httpx.AsyncClient, query: str) -> None:
    body = await _post_search(client, query=query, limit=5, mode="bm25")
    if body.get("mode_requested") != "bm25" or body.get("mode_used") != "bm25":
        raise RuntimeError("explicit BM25 request did not remain BM25")
    if body.get("degraded") is not False:
        raise RuntimeError("explicit BM25 request unexpectedly degraded")
    if body.get("reranker_enabled") is not True:
        raise RuntimeError("BM25 bypass lost canary reranker-enabled telemetry")
    if body.get("reranker_used") is not False:
        raise RuntimeError("explicit BM25 request entered the reranker path")
    if body.get("reranker_degraded") is not False:
        raise RuntimeError("BM25 bypass incorrectly reports reranker degradation")
    if body.get("reranker_degradation_reason") is not None:
        raise RuntimeError("BM25 bypass incorrectly reports a reranker failure reason")
    for result in _results(body):
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("BM25 result has no metadata")
        if RERANKER_KEYS.intersection(metadata):
            raise RuntimeError("BM25 bypass result contains reranker provenance")


async def _run() -> None:
    baseline_url = os.environ.get("PHASE3G_BASELINE_API_URL", "http://127.0.0.1:18082")
    enabled_url = os.environ.get("PHASE3G_RERANKER_API_URL", "http://127.0.0.1:18083")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )

    policy = _load_policy()
    corpus = _load_jsonl(CORPUS_PATH) + _load_jsonl(EXACT_CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH) + _load_jsonl(EXACT_QUERIES_PATH)
    id_to_url = _validate_fixture(corpus, queries, minimum_queries=30)

    spot_queries = policy["frozen_spot_queries"]
    entries = [*spot_queries["exact_identifier"], *spot_queries["semantic"]]
    if not entries:
        raise RuntimeError("Phase 3G live gate has no registered spot queries")
    for entry in entries:
        query = entry.get("query")
        target = entry.get("target")
        if not isinstance(query, str) or not query.strip():
            raise RuntimeError("Phase 3G spot query has no text")
        if not isinstance(target, str) or target not in id_to_url:
            raise RuntimeError("Phase 3G spot query references an unknown target")

    settings = Settings(
        opensearch_url=opensearch_url,
        dense_enabled=True,
        search_mode="hybrid",
    )
    store = OpenSearchStore(settings)
    embedder = DenseEmbedder(
        model_name=settings.dense_model_name,
        dimension=settings.dense_dimension,
        query_prefix=settings.dense_query_prefix,
        passage_prefix=settings.dense_passage_prefix,
    )

    try:
        await _clear_indices(store, settings)
        indexed_vector_chunks = await _seed_corpus_with_vectors(store, embedder, corpus)
        if indexed_vector_chunks < len(corpus):
            raise RuntimeError("Phase 3G live seeding wrote fewer vector chunks than documents")
        print(f"phase3g_seeded_vector_chunks={indexed_vector_chunks}")

        timeout = httpx.Timeout(180.0, connect=10.0)
        async with (
            httpx.AsyncClient(base_url=baseline_url, timeout=timeout) as baseline_client,
            httpx.AsyncClient(base_url=enabled_url, timeout=timeout) as enabled_client,
        ):
            for entry in entries:
                query = entry["query"]
                baseline = await _post_search(
                    baseline_client,
                    query=query,
                    limit=10,
                    mode="hybrid",
                )
                enabled = await _post_search(
                    enabled_client,
                    query=query,
                    limit=10,
                    mode="hybrid",
                )
                _assert_disabled(baseline)
                _assert_enabled(baseline, enabled)

                enabled_small = await _post_search(
                    enabled_client,
                    query=query,
                    limit=3,
                    mode="hybrid",
                )
                _assert_hybrid_common(enabled_small)
                if enabled_small.get("reranker_used") is not True:
                    raise RuntimeError("limit<10 request did not use the reranker")
                if _urls(enabled_small) != _urls(enabled)[:3]:
                    raise RuntimeError("limit<10 request did not rerank the frozen top-10 pool")
                print(f"phase3g_live_query={entry['id']} PASS")

            await _verify_bm25_bypass(enabled_client, entries[0]["query"])
            print("phase3g_bm25_bypass=PASS")
    finally:
        await store.close()

    print("Phase 3G live reranker verification: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
