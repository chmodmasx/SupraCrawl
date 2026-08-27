from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from supracrawl.config import Settings
from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average
from supracrawl.fusion import reciprocal_rank_fusion
from supracrawl.search import OpenSearchStore
from verify_retrieval_baseline import (
    _delete_index,
    _load_jsonl,
    _percentile,
    _seed_corpus,
    _validate_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
BM25_THRESHOLDS_PATH = ROOT / "evaluation" / "bm25_thresholds.json"
POLICY_PATH = ROOT / "evaluation" / "hybrid_candidate_policy.json"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _build_passages(corpus: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    passages: list[str] = []
    document_ids: list[str] = []
    for document in corpus:
        document_id = document["id"]
        title = document["title"]
        for raw_chunk in document["chunks"]:
            section_path = " > ".join(raw_chunk.get("section_path", []))
            text = raw_chunk["text"]
            parts = [title]
            if section_path:
                parts.append(section_path)
            parts.append(text)
            passages.append("passage: " + "\n".join(parts))
            document_ids.append(document_id)
    if not passages:
        raise RuntimeError("hybrid benchmark corpus produced no passages")
    return passages, document_ids


def _register_model(model_name: str, dimension: int) -> None:
    supported = {item["model"] for item in TextEmbedding.list_supported_models()}
    if model_name in supported:
        return
    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=model_name),
        dim=dimension,
        model_file="onnx/model.onnx",
    )


def _embed_passages(
    model: TextEmbedding,
    passages: list[str],
    dimension: int,
) -> np.ndarray:
    vectors = np.asarray(list(model.embed(passages, batch_size=16)), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(passages), dimension):
        raise RuntimeError(
            f"unexpected passage embedding shape {vectors.shape}; "
            f"expected {(len(passages), dimension)}"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("passage embeddings contain non-finite values")
    return vectors


def _embed_query_timed(
    model: TextEmbedding,
    query: str,
    dimension: int,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    vectors = list(model.embed([f"query: {query}"], batch_size=1))
    latency_ms = (time.perf_counter() - start) * 1000.0
    if len(vectors) != 1:
        raise RuntimeError("query embedding did not return exactly one vector")
    vector = np.asarray(vectors[0], dtype=np.float32)
    if vector.shape != (dimension,):
        raise RuntimeError(
            f"unexpected query embedding shape {vector.shape}; expected {(dimension,)}"
        )
    if not np.isfinite(vector).all():
        raise RuntimeError("query embedding contains non-finite values")
    return vector, latency_ms


def _dense_ranking(
    query_vector: np.ndarray,
    passage_vectors: np.ndarray,
    passage_document_ids: list[str],
    limit: int = 10,
) -> list[str]:
    scores = passage_vectors @ query_vector
    order = np.argsort(-scores, kind="stable")
    ranked: list[str] = []
    seen: set[str] = set()
    for index in order:
        document_id = passage_document_ids[int(index)]
        if document_id in seen:
            continue
        seen.add(document_id)
        ranked.append(document_id)
        if len(ranked) >= limit:
            break
    if not ranked:
        raise RuntimeError("dense retrieval produced no document ranking")
    return ranked


async def _timed_bm25(
    store: OpenSearchStore,
    query: str,
    url_to_id: dict[str, str],
) -> tuple[list[str], float]:
    start = time.perf_counter()
    results = await store.search(query, limit=10)
    latency_ms = (time.perf_counter() - start) * 1000.0
    ranking = [url_to_id.get(result["url"], result["url"]) for result in results]
    return ranking, latency_ms


def _aggregate_report(metrics: list[RetrievalMetrics]) -> dict[str, float]:
    aggregate = macro_average(metrics)
    return {
        "mrr_at_10": round(aggregate.mrr_at_10, 6),
        "recall_at_5": round(aggregate.recall_at_5, 6),
        "ndcg_at_10": round(aggregate.ndcg_at_10, 6),
    }


def _top_grade_relevant(query: dict[str, Any]) -> str:
    relevance = query["relevance"]
    return min(relevance, key=lambda document_id: (-relevance[document_id], document_id))


def _semantic_query_ids(policy: dict[str, Any]) -> list[str]:
    query_ids = policy.get("semantic_queries_top5")
    if not isinstance(query_ids, list) or not query_ids:
        raise RuntimeError("promotion policy requires semantic_queries_top5")
    if not all(isinstance(query_id, str) and query_id for query_id in query_ids):
        raise RuntimeError("semantic_queries_top5 must contain non-empty query ids")
    return query_ids


async def _run() -> None:
    corpus = _load_jsonl(CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    bm25_thresholds = _load_object(BM25_THRESHOLDS_PATH)
    policy = _load_object(POLICY_PATH)

    id_to_url = _validate_fixture(corpus, queries, int(bm25_thresholds["minimum_queries"]))
    url_to_id = {url: document_id for document_id, url in id_to_url.items()}

    model_name = str(policy["model"])
    dimension = int(policy["dimension"])
    rrf_k = int(policy["rrf_k"])
    semantic_query_ids = _semantic_query_ids(policy)
    _register_model(model_name, dimension)

    model_start = time.perf_counter()
    model = TextEmbedding(model_name=model_name)
    model_load_ms = (time.perf_counter() - model_start) * 1000.0

    passages, passage_document_ids = _build_passages(corpus)
    passage_start = time.perf_counter()
    passage_vectors = _embed_passages(model, passages, dimension)
    passage_embedding_ms = (time.perf_counter() - passage_start) * 1000.0

    settings = Settings(
        opensearch_url="http://127.0.0.1:9200",
        opensearch_documents_index="supracrawl-eval-documents-v1",
        opensearch_chunks_index="supracrawl-eval-chunks-v1",
        opensearch_timeout_s=10.0,
    )
    store = OpenSearchStore(settings)
    bm25_metrics: list[RetrievalMetrics] = []
    dense_metrics: list[RetrievalMetrics] = []
    hybrid_metrics: list[RetrievalMetrics] = []
    bm25_latencies: list[float] = []
    query_embedding_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    query_report: list[dict[str, Any]] = []

    try:
        await _delete_index(store, settings.opensearch_documents_index)
        await _delete_index(store, settings.opensearch_chunks_index)
        await store.ensure_indices()
        await _seed_corpus(store, corpus)

        for query in queries:
            hybrid_start = time.perf_counter()
            bm25_task = asyncio.create_task(_timed_bm25(store, query["query"], url_to_id))
            dense_task = asyncio.to_thread(
                _embed_query_timed,
                model,
                query["query"],
                dimension,
            )
            (bm25_ranking, bm25_latency_ms), (
                query_vector,
                query_embedding_ms,
            ) = await asyncio.gather(bm25_task, dense_task)
            dense_ranking = _dense_ranking(
                query_vector,
                passage_vectors,
                passage_document_ids,
                limit=10,
            )
            hybrid_ranking = reciprocal_rank_fusion(
                [bm25_ranking, dense_ranking],
                k=rrf_k,
                limit=10,
            )
            hybrid_latency_ms = (time.perf_counter() - hybrid_start) * 1000.0

            bm25_metric = evaluate_ranking(bm25_ranking, query["relevance"])
            dense_metric = evaluate_ranking(dense_ranking, query["relevance"])
            hybrid_metric = evaluate_ranking(hybrid_ranking, query["relevance"])
            bm25_metrics.append(bm25_metric)
            dense_metrics.append(dense_metric)
            hybrid_metrics.append(hybrid_metric)
            bm25_latencies.append(bm25_latency_ms)
            query_embedding_latencies.append(query_embedding_ms)
            hybrid_latencies.append(hybrid_latency_ms)
            query_report.append(
                {
                    "id": query["id"],
                    "language": query.get("language"),
                    "query": query["query"],
                    "bm25_top5": bm25_ranking[:5],
                    "dense_top5": dense_ranking[:5],
                    "hybrid_top5": hybrid_ranking[:5],
                    "hybrid_ndcg_at_10": round(hybrid_metric.ndcg_at_10, 6),
                    "hybrid_latency_ms": round(hybrid_latency_ms, 3),
                }
            )

        bm25_aggregate = macro_average(bm25_metrics)
        dense_aggregate = macro_average(dense_metrics)
        hybrid_aggregate = macro_average(hybrid_metrics)
        p95_hybrid_latency_ms = _percentile(hybrid_latencies, 0.95)

        baseline_failures: list[str] = []
        baseline_checks = (
            (
                bm25_aggregate.mrr_at_10,
                float(bm25_thresholds["mrr_at_10_min"]),
                "MRR@10",
            ),
            (
                bm25_aggregate.recall_at_5,
                float(bm25_thresholds["recall_at_5_min"]),
                "Recall@5",
            ),
            (
                bm25_aggregate.ndcg_at_10,
                float(bm25_thresholds["ndcg_at_10_min"]),
                "nDCG@10",
            ),
        )
        for actual, minimum, label in baseline_checks:
            if actual < minimum:
                baseline_failures.append(f"BM25 {label} {actual:.6f} < {minimum:.6f}")
        if baseline_failures:
            raise RuntimeError(
                "certified BM25 baseline regressed during Phase 3B experiment:\n- "
                + "\n- ".join(baseline_failures)
            )

        checks: dict[str, bool] = {
            "mrr_no_regression": (
                bm25_aggregate.mrr_at_10 - hybrid_aggregate.mrr_at_10
                <= float(policy["max_mrr_at_10_regression"])
            ),
            "recall_no_regression": (
                bm25_aggregate.recall_at_5 - hybrid_aggregate.recall_at_5
                <= float(policy["max_recall_at_5_regression"])
            ),
            "ndcg_material_improvement": (
                hybrid_aggregate.ndcg_at_10 - bm25_aggregate.ndcg_at_10
                >= float(policy["ndcg_at_10_min_delta"])
            ),
            "p95_latency_guardrail": (
                p95_hybrid_latency_ms <= float(policy["p95_hybrid_latency_ms_max"])
            ),
        }
        query_by_id = {query["id"]: query for query in queries}
        detail_by_id = {detail["id"]: detail for detail in query_report}
        for query_id in semantic_query_ids:
            query = query_by_id.get(query_id)
            detail = detail_by_id.get(query_id)
            if query is None or detail is None:
                raise RuntimeError(f"promotion policy references unknown query {query_id}")
            target = _top_grade_relevant(query)
            checks[f"semantic_top5:{query_id}"] = target in detail["hybrid_top5"]

        promotion_passed = all(checks.values())
        report = {
            "benchmark": policy["benchmark"],
            "model": {
                "name": model_name,
                "dimension": dimension,
                "load_ms": round(model_load_ms, 3),
                "passages": len(passages),
                "passage_embedding_ms": round(passage_embedding_ms, 3),
            },
            "documents": len(corpus),
            "queries": len(queries),
            "rrf_k": rrf_k,
            "aggregate": {
                "bm25": _aggregate_report(bm25_metrics),
                "dense": _aggregate_report(dense_metrics),
                "hybrid": _aggregate_report(hybrid_metrics),
                "hybrid_ndcg_delta": round(
                    hybrid_aggregate.ndcg_at_10 - bm25_aggregate.ndcg_at_10,
                    6,
                ),
            },
            "latency_ms": {
                "bm25_p95": round(_percentile(bm25_latencies, 0.95), 3),
                "query_embedding_p95": round(
                    _percentile(query_embedding_latencies, 0.95),
                    3,
                ),
                "hybrid_p50": round(_percentile(hybrid_latencies, 0.50), 3),
                "hybrid_p95": round(p95_hybrid_latency_ms, 3),
                "hybrid_mean": round(mean(hybrid_latencies), 3),
            },
            "promotion": {
                "passed": promotion_passed,
                "checks": checks,
            },
            "queries_detail": query_report,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        verdict = "PASS" if promotion_passed else "REJECT"
        print(f"Phase 3B hybrid promotion candidate: {verdict}")
        print("Phase 3B hybrid experiment execution: PASS")
    finally:
        await _delete_index(store, settings.opensearch_documents_index)
        await _delete_index(store, settings.opensearch_chunks_index)
        await store.close()


if __name__ == "__main__":
    asyncio.run(_run())
