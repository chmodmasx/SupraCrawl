from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from verify_real_vector_retrieval import _clear_indices, _seed_corpus_with_vectors
from verify_retrieval_baseline import _load_jsonl, _percentile, _validate_fixture

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder, EmbeddingBackendError
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.indexer import Indexer
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3d_policy.json"
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
EXACT_CORPUS_PATH = ROOT / "evaluation" / "phase3c_exact_corpus.jsonl"
EXACT_QUERIES_PATH = ROOT / "evaluation" / "phase3c_exact_queries.jsonl"
FETCHED_AT = "2026-08-27T00:00:00+00:00"
STALE_URL = "https://benchmark.supracrawl.local/phase3d-stale-vector"


class _FailingEmbedder:
    async def embed_passages(
        self,
        _passages: list[str],
        *,
        batch_size: int = 16,
    ) -> list[list[float]]:
        del batch_size
        raise EmbeddingBackendError("phase3d forced vector refresh failure")


def _load_policy() -> dict[str, Any]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 3D policy must be a JSON object")
    return payload


def _fixture(
    *,
    url: str,
    title: str,
    text: str,
) -> tuple[FetchResult, Extraction]:
    fetched = FetchResult(
        fetch_url=url,
        final_url=url,
        status_code=200,
        content_type="text/html",
        html="",
        fetched_at=FETCHED_AT,
    )
    extraction = Extraction(
        title=title,
        markdown=text,
        canonical_url=url,
        extractor="phase3d-live-fixture",
        quality=1.0,
        rendered=False,
    )
    return fetched, extraction


def _result_urls(body: dict[str, Any]) -> list[str]:
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("search response has no results list")
    urls: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("search response contains an invalid result")
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("search result has no URL")
        urls.append(url)
    return urls


def _assert_hybrid_response(body: dict[str, Any]) -> None:
    if body.get("success") is not True:
        raise RuntimeError("hybrid API response is not successful")
    if body.get("mode_requested") != "hybrid":
        raise RuntimeError("hybrid API did not report the requested mode")
    if body.get("mode_used") != "hybrid":
        raise RuntimeError("hybrid API did not use hybrid retrieval")
    if body.get("degraded") is not False:
        raise RuntimeError("successful hybrid API response reported degradation")

    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("hybrid API response has no results list")
    positions = [result.get("position") for result in results if isinstance(result, dict)]
    if positions != list(range(1, len(results) + 1)):
        raise RuntimeError("hybrid result positions are not contiguous")
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("hybrid API returned an invalid result")
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("hybrid result has no metadata")
        if metadata.get("retrieval_mode") != "hybrid":
            raise RuntimeError("hybrid result provenance is missing retrieval mode")
        if "lexical_rank" not in metadata or "dense_rank" not in metadata:
            raise RuntimeError("hybrid result provenance is missing component ranks")


async def _post_search(
    client: httpx.AsyncClient,
    *,
    query: str,
    limit: int = 10,
    mode: str | None = None,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {"query": query, "limit": limit}
    if mode is not None:
        payload["mode"] = mode
    started = time.perf_counter()
    response = await client.post("/v1/search", json=payload)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(
            f"search API returned HTTP {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("search API returned non-object JSON")
    return body, latency_ms


async def _verify_default_bm25(client: httpx.AsyncClient, query: str) -> None:
    body, _ = await _post_search(client, query=query, limit=5)
    if body.get("mode_requested") != "bm25" or body.get("mode_used") != "bm25":
        raise RuntimeError("legacy request without mode did not remain BM25")
    if body.get("degraded") is not False:
        raise RuntimeError("default BM25 request unexpectedly reported degradation")


async def _verify_spot_checks(
    client: httpx.AsyncClient,
    policy: dict[str, Any],
    id_to_url: dict[str, str],
) -> list[float]:
    latencies: list[float] = []
    spot_queries = policy["frozen_spot_queries"]
    for group in ("exact_identifier", "semantic"):
        entries = spot_queries[group]
        for entry in entries:
            body, latency_ms = await _post_search(
                client,
                query=entry["query"],
                mode="hybrid",
            )
            _assert_hybrid_response(body)
            latencies.append(latency_ms)
            urls = _result_urls(body)
            target_url = id_to_url[entry["target"]]
            try:
                rank = urls.index(target_url) + 1
            except ValueError as exc:
                raise RuntimeError(
                    f"{entry['id']} target {entry['target']} was not returned"
                ) from exc
            if rank > int(entry["required_rank_max"]):
                raise RuntimeError(
                    f"{entry['id']} target rank {rank} exceeds "
                    f"{entry['required_rank_max']}"
                )
            print(f"phase3d_spot {entry['id']} rank={rank} latency_ms={latency_ms:.3f}")
    return latencies


async def _measure_warm_p95(
    client: httpx.AsyncClient,
    policy: dict[str, Any],
) -> float:
    entries = [
        *policy["frozen_spot_queries"]["exact_identifier"],
        *policy["frozen_spot_queries"]["semantic"],
    ]

    # The first hybrid request may download/load the local model. The policy
    # explicitly excludes first model download and measures warm API latency.
    warmup_body, _ = await _post_search(
        client,
        query=entries[0]["query"],
        mode="hybrid",
    )
    _assert_hybrid_response(warmup_body)

    latencies: list[float] = []
    for _round in range(5):
        for entry in entries:
            body, latency_ms = await _post_search(
                client,
                query=entry["query"],
                mode="hybrid",
            )
            _assert_hybrid_response(body)
            latencies.append(latency_ms)

    p95 = _percentile(latencies, 0.95)
    maximum = float(policy["performance"]["warm_hybrid_api_p95_ms_max"])
    print(f"phase3d_warm_hybrid_api_p95_ms={p95:.3f} limit_ms={maximum:.3f}")
    if p95 > maximum:
        raise RuntimeError(f"warm hybrid API p95 {p95:.3f} ms exceeds {maximum:.3f} ms")
    return p95


async def _verify_stale_vector_guard(
    client: httpx.AsyncClient,
    settings: Settings,
    store: OpenSearchStore,
    embedder: DenseEmbedder,
) -> None:
    v1_text = (
        "# Obsolete version\n\n"
        "antique pineapple semaphore knowledge belongs only to the obsolete vector version"
    )
    v2_text = (
        "# Current version\n\n"
        "current glacier telemetry replaces all prior wording and is authoritative now"
    )
    fetched_v1, extraction_v1 = _fixture(
        url=STALE_URL,
        title="Phase 3D stale vector guard",
        text=v1_text,
    )
    good_indexer = Indexer(settings, None, store, embedder)  # type: ignore[arg-type]
    first = await good_indexer.index_extraction(fetched_v1, extraction_v1)
    if not first.indexed or first.vector_indexed is not True:
        raise RuntimeError(f"initial vector indexing failed: {first}")
    if first.vector_chunks_indexed <= 0:
        raise RuntimeError("initial vector indexing wrote no vector chunks")

    fetched_v2, extraction_v2 = _fixture(
        url=STALE_URL,
        title="Phase 3D stale vector guard",
        text=v2_text,
    )
    failing_indexer = Indexer(
        settings,
        None,  # type: ignore[arg-type]
        store,
        _FailingEmbedder(),  # type: ignore[arg-type]
    )
    second = await failing_indexer.index_extraction(fetched_v2, extraction_v2)
    if not second.indexed:
        raise RuntimeError("lexical refresh was lost after forced vector failure")
    if second.vector_indexed is not False:
        raise RuntimeError("forced vector refresh failure was not reported")
    if not second.vector_error or "forced vector refresh failure" not in second.vector_error:
        raise RuntimeError("forced vector refresh failure reason was not reported")

    body, _ = await _post_search(
        client,
        query="antique pineapple semaphore knowledge",
        mode="hybrid",
    )
    if body.get("mode_requested") != "hybrid":
        raise RuntimeError("stale-vector query lost requested hybrid mode")
    if STALE_URL in _result_urls(body):
        raise RuntimeError("stale vector candidate leaked through current-content hash guard")
    print("phase3d_stale_vector_guard=PASS")
    print("phase3d_lexical_success_after_vector_failure=PASS")


async def _run() -> None:
    policy = _load_policy()
    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )

    defaults = Settings()
    frozen_defaults = policy["production_default"]
    if defaults.search_mode != frozen_defaults["search_mode"]:
        raise RuntimeError("configured search_mode default changed after preregistration")
    if defaults.dense_enabled is not frozen_defaults["dense_enabled"]:
        raise RuntimeError("configured dense_enabled default changed after preregistration")

    settings = Settings(
        opensearch_url=opensearch_url,
        dense_enabled=True,
        search_mode="bm25",
    )
    hybrid_policy = policy["hybrid"]
    if settings.dense_model_name != hybrid_policy["model"]:
        raise RuntimeError("configured dense model differs from frozen Phase 3D policy")
    if settings.dense_dimension != hybrid_policy["dimension"]:
        raise RuntimeError("configured dense dimension differs from frozen Phase 3D policy")
    if settings.hybrid_rrf_k != hybrid_policy["rrf_k"]:
        raise RuntimeError("configured RRF k differs from frozen Phase 3D policy")

    corpus = _load_jsonl(CORPUS_PATH) + _load_jsonl(EXACT_CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH) + _load_jsonl(EXACT_QUERIES_PATH)
    id_to_url = _validate_fixture(corpus, queries, minimum_queries=30)

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
            raise RuntimeError("real vector corpus seeding wrote fewer chunks than documents")
        print(f"phase3d_seeded_vector_chunks={indexed_vector_chunks}")

        timeout = httpx.Timeout(180.0, connect=10.0)
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as client:
            exact_query = policy["frozen_spot_queries"]["exact_identifier"][0]["query"]
            await _verify_default_bm25(client, exact_query)
            print("phase3d_default_bm25_api=PASS")

            # Warm the model before the frozen correctness checks so first-download
            # behavior cannot contaminate either ranking or the registered p95 gate.
            warmup, _ = await _post_search(
                client,
                query=policy["frozen_spot_queries"]["semantic"][0]["query"],
                mode="hybrid",
            )
            _assert_hybrid_response(warmup)
            print("phase3d_hybrid_api_warmup=PASS")

            await _verify_spot_checks(client, policy, id_to_url)
            print("phase3d_frozen_spot_checks=PASS")
            await _measure_warm_p95(client, policy)
            await _verify_stale_vector_guard(client, settings, store, embedder)
    finally:
        await store.close()

    print("Phase 3D hybrid production verification: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
