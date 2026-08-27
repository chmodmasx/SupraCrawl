from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from verify_retrieval_baseline import (
    _delete_index,
    _load_jsonl,
    _percentile,
    _validate_fixture,
)

from supracrawl.chunking import Chunk, approx_tokens
from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder, build_passage_text
from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.fusion import reciprocal_rank_fusion
from supracrawl.search import OpenSearchStore, document_id

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
EXACT_CORPUS_PATH = ROOT / "evaluation" / "phase3c_exact_corpus.jsonl"
EXACT_QUERIES_PATH = ROOT / "evaluation" / "phase3c_exact_queries.jsonl"
POLICY_PATH = ROOT / "evaluation" / "phase3c_policy.json"
FETCHED_AT = "2026-08-27T00:00:00+00:00"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _top_grade_relevant(query: dict[str, Any]) -> str:
    relevance = query["relevance"]
    return min(relevance, key=lambda document_id: (-relevance[document_id], document_id))


def _aggregate(metrics: list[RetrievalMetrics]) -> dict[str, float]:
    aggregate = macro_average(metrics)
    return {
        "mrr_at_10": round(aggregate.mrr_at_10, 6),
        "recall_at_5": round(aggregate.recall_at_5, 6),
        "ndcg_at_10": round(aggregate.ndcg_at_10, 6),
    }


def _prepare_document(
    document: dict[str, Any],
    *,
    passage_prefix: str,
) -> tuple[FetchResult, Extraction, list[Chunk], str, list[str]]:
    chunks: list[Chunk] = []
    markdown_parts: list[str] = []
    passages: list[str] = []
    title = document["title"]
    for ordinal, raw_chunk in enumerate(document["chunks"]):
        raw_path = raw_chunk.get("section_path", [])
        text = raw_chunk.get("text")
        if not isinstance(raw_path, list) or not all(
            isinstance(part, str) for part in raw_path
        ):
            raise RuntimeError(f"invalid section_path in {document['id']}")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"empty chunk in {document['id']}")
        chunk = Chunk(
            ordinal=ordinal,
            section_path=tuple(raw_path),
            text=text,
            approx_tokens=approx_tokens(text),
        )
        chunks.append(chunk)
        markdown_parts.append(text)
        passages.append(
            build_passage_text(
                title=title,
                section_path=raw_path,
                text=text,
                prefix=passage_prefix,
            )
        )

    markdown = "\n\n".join(markdown_parts)
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    fetched = FetchResult(
        fetch_url=document["url"],
        final_url=document["url"],
        status_code=200,
        content_type="text/html",
        html="",
        fetched_at=FETCHED_AT,
    )
    extraction = Extraction(
        title=title,
        markdown=markdown,
        canonical_url=document["url"],
        extractor="phase3c-benchmark-fixture",
        quality=1.0,
        rendered=False,
    )
    return fetched, extraction, chunks, digest, passages


async def _clear_indices(store: OpenSearchStore, settings: Settings) -> None:
    await _delete_index(store, settings.opensearch_documents_index)
    await _delete_index(store, settings.opensearch_chunks_index)
    await _delete_index(store, settings.opensearch_vector_chunks_index)
    store._indices_ready = False
    store._vector_index_ready = False


async def _seed_corpus_with_vectors(
    store: OpenSearchStore,
    embedder: DenseEmbedder,
    corpus: list[dict[str, Any]],
) -> int:
    prepared: list[tuple[FetchResult, Extraction, list[Chunk], str]] = []
    all_passages: list[str] = []
    for document in corpus:
        fetched, extraction, chunks, digest, passages = _prepare_document(
            document,
            passage_prefix=embedder.passage_prefix,
        )
        prepared.append((fetched, extraction, chunks, digest))
        all_passages.extend(passages)

    vectors = await embedder.embed_passages(all_passages, batch_size=16)
    vector_offset = 0
    indexed_vector_chunks = 0
    for fetched, extraction, chunks, digest in prepared:
        chunk_vectors = vectors[vector_offset : vector_offset + len(chunks)]
        vector_offset += len(chunks)
        _, lexical_count = await store.index_document(
            fetched=fetched,
            extraction=extraction,
            chunks=chunks,
            content_hash=digest,
        )
        _, vector_count = await store.index_vector_document(
            fetched=fetched,
            extraction=extraction,
            chunks=chunks,
            content_hash=digest,
            vectors=chunk_vectors,
        )
        if lexical_count != len(chunks) or vector_count != len(chunks):
            raise RuntimeError("lexical/vector chunk count mismatch while seeding benchmark")
        indexed_vector_chunks += vector_count

    if vector_offset != len(vectors):
        raise RuntimeError("vector slicing did not consume all passage embeddings")
    return indexed_vector_chunks


async def _verify_physical_vector_storage(
    store: OpenSearchStore,
    expected_chunks: int,
) -> dict[str, Any]:
    await store.ensure_vector_index(validate=True)
    index_name = store.settings.opensearch_vector_chunks_index
    count_response = await store._request("GET", f"/{index_name}/_count")
    if count_response.status_code >= 400:
        raise RuntimeError(
            f"unable to count vector chunks: HTTP {count_response.status_code}"
        )
    count_payload = count_response.json()
    count = count_payload.get("count") if isinstance(count_payload, dict) else None
    if count != expected_chunks:
        raise RuntimeError(f"vector index count {count!r} != expected {expected_chunks}")

    sample_response = await store._request(
        "POST",
        f"/{index_name}/_search",
        json={
            "size": 1,
            "_source": ["embedding_model", "embedding_dimension", "embedding"],
            "query": {"match_all": {}},
        },
    )
    if sample_response.status_code >= 400:
        raise RuntimeError(
            f"unable to inspect stored vector: HTTP {sample_response.status_code}"
        )
    try:
        source = sample_response.json()["hits"]["hits"][0]["_source"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("vector index sample returned an invalid response") from exc
    embedding = source.get("embedding") if isinstance(source, dict) else None
    if not isinstance(embedding, list) or len(embedding) != store.settings.dense_dimension:
        raise RuntimeError("stored OpenSearch vector has an unexpected dimension")
    if source.get("embedding_model") != store.settings.dense_model_name:
        raise RuntimeError("stored vector model provenance does not match configured model")
    if source.get("embedding_dimension") != store.settings.dense_dimension:
        raise RuntimeError("stored vector dimension provenance does not match configured model")
    return {
        "count": count,
        "model": source["embedding_model"],
        "dimension": source["embedding_dimension"],
    }


async def _timed_bm25(
    store: OpenSearchStore,
    query: str,
    url_to_id: dict[str, str],
) -> tuple[list[str], float]:
    start = time.perf_counter()
    results = await store.search(query, limit=10)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return [url_to_id.get(result["url"], result["url"]) for result in results], latency_ms


async def _timed_dense(
    store: OpenSearchStore,
    embedder: DenseEmbedder,
    query: str,
    url_to_id: dict[str, str],
) -> tuple[list[str], float]:
    start = time.perf_counter()
    vector = await embedder.embed_query(query)
    results = await store.dense_search(vector, limit=10)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return [url_to_id.get(result["url"], result["url"]) for result in results], latency_ms


async def _run_queries(
    store: OpenSearchStore,
    embedder: DenseEmbedder,
    queries: list[dict[str, Any]],
    url_to_id: dict[str, str],
    rrf_k: int,
) -> dict[str, Any]:
    bm25_metrics: list[RetrievalMetrics] = []
    dense_metrics: list[RetrievalMetrics] = []
    hybrid_metrics: list[RetrievalMetrics] = []
    bm25_latencies: list[float] = []
    dense_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    details: list[dict[str, Any]] = []

    for query in queries:
        hybrid_start = time.perf_counter()
        bm25_task = asyncio.create_task(_timed_bm25(store, query["query"], url_to_id))
        dense_task = asyncio.create_task(
            _timed_dense(store, embedder, query["query"], url_to_id)
        )
        (bm25_ranking, bm25_ms), (dense_ranking, dense_ms) = await asyncio.gather(
            bm25_task,
            dense_task,
        )
        hybrid_ranking = reciprocal_rank_fusion(
            [bm25_ranking, dense_ranking],
            k=rrf_k,
            limit=10,
        )
        hybrid_ms = (time.perf_counter() - hybrid_start) * 1000.0

        bm25_metric = evaluate_ranking(bm25_ranking, query["relevance"])
        dense_metric = evaluate_ranking(dense_ranking, query["relevance"])
        hybrid_metric = evaluate_ranking(hybrid_ranking, query["relevance"])
        bm25_metrics.append(bm25_metric)
        dense_metrics.append(dense_metric)
        hybrid_metrics.append(hybrid_metric)
        bm25_latencies.append(bm25_ms)
        dense_latencies.append(dense_ms)
        hybrid_latencies.append(hybrid_ms)
        details.append(
            {
                "id": query["id"],
                "query": query["query"],
                "bm25_top5": bm25_ranking[:5],
                "dense_top5": dense_ranking[:5],
                "hybrid_top5": hybrid_ranking[:5],
            }
        )

    return {
        "aggregate": {
            "bm25": _aggregate(bm25_metrics),
            "dense": _aggregate(dense_metrics),
            "hybrid": _aggregate(hybrid_metrics),
        },
        "latency_ms": {
            "bm25_p95": round(_percentile(bm25_latencies, 0.95), 3),
            "dense_p50": round(_percentile(dense_latencies, 0.50), 3),
            "dense_p95": round(_percentile(dense_latencies, 0.95), 3),
            "dense_mean": round(mean(dense_latencies), 3),
            "hybrid_p50": round(_percentile(hybrid_latencies, 0.50), 3),
            "hybrid_p95": round(_percentile(hybrid_latencies, 0.95), 3),
            "hybrid_mean": round(mean(hybrid_latencies), 3),
        },
        "details": details,
    }


def _metric_value(report: dict[str, Any], ranker: str, metric: str) -> float:
    return float(report["aggregate"][ranker][metric])


def _detail_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {detail["id"]: detail for detail in report["details"]}


def _exact_top1_rate(
    report: dict[str, Any],
    query_by_id: dict[str, dict[str, Any]],
    query_ids: list[str],
    ranker: str,
) -> float:
    details = _detail_map(report)
    successes = 0
    ranking_key = f"{ranker}_top5"
    for query_id in query_ids:
        query = query_by_id.get(query_id)
        detail = details.get(query_id)
        if query is None or detail is None:
            raise RuntimeError(f"policy references unknown exact query {query_id}")
        target = _top_grade_relevant(query)
        ranking = detail[ranking_key]
        if ranking and ranking[0] == target:
            successes += 1
    return successes / len(query_ids)


def _candidate_checks(
    *,
    report: dict[str, Any],
    queries: list[dict[str, Any]],
    policy: dict[str, Any],
    ranker: str,
) -> dict[str, bool]:
    gate = policy["expanded_candidate_gate"]
    exact_query_ids = policy["exact_identifier_queries"]
    query_by_id = {query["id"]: query for query in queries}
    details = _detail_map(report)
    checks = {
        "ndcg_material_improvement": (
            _metric_value(report, ranker, "ndcg_at_10")
            - _metric_value(report, "bm25", "ndcg_at_10")
            >= float(gate["ndcg_at_10_min_delta_vs_bm25"])
        ),
        "mrr_no_regression": (
            _metric_value(report, "bm25", "mrr_at_10")
            - _metric_value(report, ranker, "mrr_at_10")
            <= float(gate["max_mrr_at_10_regression_vs_bm25"])
        ),
        "recall_no_regression": (
            _metric_value(report, "bm25", "recall_at_5")
            - _metric_value(report, ranker, "recall_at_5")
            <= float(gate["max_recall_at_5_regression_vs_bm25"])
        ),
        "exact_identifier_top1": (
            _exact_top1_rate(report, query_by_id, exact_query_ids, ranker)
            >= float(gate["exact_identifier_top1_rate_min"])
        ),
        "p95_latency_guardrail": (
            float(report["latency_ms"][f"{ranker}_p95"])
            <= float(gate["p95_latency_ms_max"])
        ),
    }
    ranking_key = f"{ranker}_top5"
    for query_id in gate["semantic_queries_top5"]:
        query = query_by_id.get(query_id)
        detail = details.get(query_id)
        if query is None or detail is None:
            raise RuntimeError(f"policy references unknown semantic query {query_id}")
        checks[f"semantic_top5:{query_id}"] = (
            _top_grade_relevant(query) in detail[ranking_key]
        )
    return checks


def _select_candidate(
    report: dict[str, Any],
    checks: dict[str, dict[str, bool]],
    policy: dict[str, Any],
) -> str:
    eligible = [
        ranker for ranker in ("dense", "hybrid") if all(checks[ranker].values())
    ]
    if not eligible:
        return "bm25"
    if len(eligible) == 1:
        return eligible[0]

    dense_ndcg = _metric_value(report, "dense", "ndcg_at_10")
    hybrid_ndcg = _metric_value(report, "hybrid", "ndcg_at_10")
    tie_delta = float(policy["selection"]["tie_absolute_delta"])
    if abs(dense_ndcg - hybrid_ndcg) <= tie_delta:
        dense_p95 = float(report["latency_ms"]["dense_p95"])
        hybrid_p95 = float(report["latency_ms"]["hybrid_p95"])
        return "dense" if dense_p95 <= hybrid_p95 else "hybrid"
    return "dense" if dense_ndcg > hybrid_ndcg else "hybrid"


def _assert_original_controls(report: dict[str, Any], policy: dict[str, Any]) -> None:
    failures: list[str] = []
    control = policy["phase3_bm25_control"]
    tolerance = float(control["absolute_tolerance"])
    for metric in ("mrr_at_10", "recall_at_5", "ndcg_at_10"):
        actual = _metric_value(report, "bm25", metric)
        expected = float(control[metric])
        if abs(actual - expected) > tolerance:
            failures.append(
                f"BM25 {metric} {actual:.6f} != {expected:.6f} ± {tolerance}"
            )

    dense_control = policy["phase3b_real_dense_control"]
    for metric, key in (
        ("mrr_at_10", "mrr_at_10_min"),
        ("recall_at_5", "recall_at_5_min"),
        ("ndcg_at_10", "ndcg_at_10_min"),
    ):
        actual = _metric_value(report, "dense", metric)
        minimum = float(dense_control[key])
        if actual < minimum:
            failures.append(f"real dense {metric} {actual:.6f} < {minimum:.6f}")

    query_by_id = {query["id"]: query for query in _load_jsonl(QUERIES_PATH)}
    details = _detail_map(report)
    for query_id in dense_control["semantic_queries_top5"]:
        target = _top_grade_relevant(query_by_id[query_id])
        if target not in details[query_id]["dense_top5"]:
            failures.append(f"real dense semantic query {query_id} missed top 5")

    if failures:
        raise RuntimeError(
            "Phase 3C original-corpus control failed:\n- " + "\n- ".join(failures)
        )


async def _assert_stale_vector_replacement(
    store: OpenSearchStore,
    embedder: DenseEmbedder,
) -> None:
    old_document = {
        "id": "phase3c-stale-vector-check",
        "url": "https://benchmark.supracrawl.local/stale-vector-check",
        "title": "Vector stale replacement check",
        "chunks": [
            {
                "section_path": ["Vector", "Stale replacement"],
                "text": "obsolete semantic payload alpha that must disappear",
            }
        ],
    }
    new_document = {
        **old_document,
        "chunks": [
            {
                "section_path": ["Vector", "Stale replacement"],
                "text": "replacement semantic payload omega that must remain",
            }
        ],
    }

    old_fetched, old_extraction, old_chunks, old_hash, old_passages = _prepare_document(
        old_document,
        passage_prefix=embedder.passage_prefix,
    )
    old_vectors = await embedder.embed_passages(old_passages)
    await store.index_vector_document(
        fetched=old_fetched,
        extraction=old_extraction,
        chunks=old_chunks,
        content_hash=old_hash,
        vectors=old_vectors,
    )

    new_fetched, new_extraction, new_chunks, new_hash, new_passages = _prepare_document(
        new_document,
        passage_prefix=embedder.passage_prefix,
    )
    if new_hash == old_hash:
        raise RuntimeError("stale-vector fixture hashes unexpectedly match")
    new_vectors = await embedder.embed_passages(new_passages)
    await store.index_vector_document(
        fetched=new_fetched,
        extraction=new_extraction,
        chunks=new_chunks,
        content_hash=new_hash,
        vectors=new_vectors,
    )

    index_name = store.settings.opensearch_vector_chunks_index
    doc_id = document_id(old_document["url"])
    response = await store._request(
        "POST",
        f"/{index_name}/_search",
        json={
            "size": 10,
            "_source": ["content_hash", "text"],
            "query": {"term": {"document_id": doc_id}},
        },
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"stale-vector verification search failed: HTTP {response.status_code}"
        )
    try:
        hits = response.json()["hits"]["hits"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("stale-vector verification returned invalid JSON") from exc
    if not isinstance(hits, list) or len(hits) != 1:
        raise RuntimeError(
            f"stale-vector replacement left {len(hits) if isinstance(hits, list) else '?'} "
            "searchable chunks; expected exactly one"
        )
    source = hits[0].get("_source") if isinstance(hits[0], dict) else None
    if not isinstance(source, dict):
        raise RuntimeError("stale-vector replacement returned an invalid hit")
    if source.get("content_hash") != new_hash:
        raise RuntimeError("stale vector content hash remained searchable")
    if source.get("text") != new_document["chunks"][0]["text"]:
        raise RuntimeError("stale vector text remained searchable")


async def _assert_bm25_survives_vector_loss(
    store: OpenSearchStore,
    query: str,
) -> None:
    index_name = store.settings.opensearch_vector_chunks_index
    response = await store._request("DELETE", f"/{index_name}")
    if response.status_code not in {200, 404}:
        raise RuntimeError(
            "unable to remove vector index for degradation test: "
            f"{response.status_code}"
        )
    store._vector_index_ready = False
    results = await store.search(query, limit=5)
    if not results:
        raise RuntimeError("BM25 returned no results after vector-index loss")


async def _run() -> None:
    policy = _load_object(POLICY_PATH)
    base_corpus = _load_jsonl(CORPUS_PATH)
    base_queries = _load_jsonl(QUERIES_PATH)
    exact_corpus = _load_jsonl(EXACT_CORPUS_PATH)
    exact_queries = _load_jsonl(EXACT_QUERIES_PATH)
    combined_corpus = base_corpus + exact_corpus
    combined_queries = base_queries + exact_queries

    base_id_to_url = _validate_fixture(base_corpus, base_queries, minimum_queries=16)
    combined_id_to_url = _validate_fixture(
        combined_corpus,
        combined_queries,
        minimum_queries=30,
    )
    if len(policy["exact_identifier_queries"]) != len(exact_queries):
        raise RuntimeError("Phase 3C exact-query policy and fixture size differ")

    base_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )
    settings = Settings(
        opensearch_url=base_url,
        opensearch_documents_index="supracrawl-phase3c-documents-v1",
        opensearch_chunks_index="supracrawl-phase3c-chunks-v1",
        opensearch_vector_chunks_index="supracrawl-phase3c-vector-chunks-v1",
        dense_model_name=str(policy["model"]),
        dense_dimension=int(policy["dimension"]),
        dense_query_prefix=str(policy["query_prefix"]),
        dense_passage_prefix=str(policy["passage_prefix"]),
        opensearch_timeout_s=10.0,
    )
    if settings.dense_enabled:
        raise RuntimeError("Phase 3C must not enable dense retrieval as the production default")

    store = OpenSearchStore(settings)
    embedder = DenseEmbedder(
        model_name=settings.dense_model_name,
        dimension=settings.dense_dimension,
        query_prefix=settings.dense_query_prefix,
        passage_prefix=settings.dense_passage_prefix,
    )
    rrf_k = int(policy["rrf_k"])

    try:
        await _clear_indices(store, settings)
        base_vector_chunks = await _seed_corpus_with_vectors(
            store,
            embedder,
            base_corpus,
        )
        base_storage = await _verify_physical_vector_storage(
            store,
            base_vector_chunks,
        )
        base_report = await _run_queries(
            store,
            embedder,
            base_queries,
            {url: document_id for document_id, url in base_id_to_url.items()},
            rrf_k,
        )
        _assert_original_controls(base_report, policy)

        await _clear_indices(store, settings)
        combined_vector_chunks = await _seed_corpus_with_vectors(
            store,
            embedder,
            combined_corpus,
        )
        combined_storage = await _verify_physical_vector_storage(
            store,
            combined_vector_chunks,
        )
        expanded_report = await _run_queries(
            store,
            embedder,
            combined_queries,
            {url: document_id for document_id, url in combined_id_to_url.items()},
            rrf_k,
        )

        checks = {
            "dense": _candidate_checks(
                report=expanded_report,
                queries=combined_queries,
                policy=policy,
                ranker="dense",
            ),
            "hybrid": _candidate_checks(
                report=expanded_report,
                queries=combined_queries,
                policy=policy,
                ranker="hybrid",
            ),
        }
        selected = _select_candidate(expanded_report, checks, policy)
        query_by_id = {query["id"]: query for query in combined_queries}
        exact_ids = list(policy["exact_identifier_queries"])
        exact_top1 = {
            ranker: round(
                _exact_top1_rate(expanded_report, query_by_id, exact_ids, ranker),
                6,
            )
            for ranker in ("bm25", "dense", "hybrid")
        }

        # Hardening checks deliberately run after metric collection so they cannot
        # change the frozen benchmark rankings that select the candidate.
        await _assert_stale_vector_replacement(store, embedder)
        await _assert_bm25_survives_vector_loss(store, base_queries[0]["query"])

        report = {
            "benchmark": policy["benchmark"],
            "base_main_sha": policy["base_main_sha"],
            "model": {
                "name": settings.dense_model_name,
                "dimension": settings.dense_dimension,
            },
            "vector_backend": policy["vector_index"],
            "rrf_k": rrf_k,
            "original_control": {
                "documents": len(base_corpus),
                "queries": len(base_queries),
                "storage": base_storage,
                "aggregate": base_report["aggregate"],
                "latency_ms": base_report["latency_ms"],
            },
            "expanded": {
                "documents": len(combined_corpus),
                "queries": len(combined_queries),
                "storage": combined_storage,
                "aggregate": expanded_report["aggregate"],
                "latency_ms": expanded_report["latency_ms"],
                "exact_identifier_top1_rate": exact_top1,
                "candidate_checks": checks,
                "eligible": {
                    ranker: all(ranker_checks.values())
                    for ranker, ranker_checks in checks.items()
                },
                "selected_candidate": selected,
                "queries_detail": expanded_report["details"],
            },
            "production_default": "bm25",
            "production_default_changed": False,
            "hardening": {
                "stale_vector_replacement": "PASS",
            },
            "degradation": {
                "bm25_after_vector_index_loss": "PASS",
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"Phase 3C measured selection: {selected.upper()}")
        print("Phase 3C real vector experiment execution: PASS")
    finally:
        await _clear_indices(store, settings)
        await store.close()


if __name__ == "__main__":
    asyncio.run(_run())
