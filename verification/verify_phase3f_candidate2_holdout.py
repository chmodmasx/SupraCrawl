from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from verify_phase3f_baseline import _assert_mode, _post_search
from verify_phase3f_candidate1 import (
    _candidate_text,
    _load_object,
    _metric_dict,
    _peak_rss_mib,
    _prepare_reranker,
    _ranking_from_results,
    _score_timed,
)
from verify_phase3f_holdout_freeze import verify_holdout_freeze
from verify_real_vector_retrieval import _clear_indices, _seed_corpus_with_vectors
from verify_retrieval_baseline import _load_jsonl, _percentile, _validate_fixture

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder
from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average, recall_at_k
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3f_policy.json"
CANDIDATE1_POLICY_PATH = ROOT / "evaluation" / "phase3f_candidate1_policy.json"
CANDIDATE2_POLICY_PATH = ROOT / "evaluation" / "phase3f_candidate2_policy.json"
CORPUS_PATH = ROOT / "evaluation" / "phase3f_holdout_corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "phase3f_holdout_queries.jsonl"


def _candidate2_order(scores: list[float], protected_top_k: int) -> list[int]:
    if len(scores) != 10:
        raise RuntimeError("Candidate 2 requires exactly 10 first-stage candidates")
    if protected_top_k <= 0 or protected_top_k >= len(scores):
        raise RuntimeError("protected_top_k must split the candidate pool")

    score_order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    protected = [index for index in score_order if index < protected_top_k]
    tail = [index for index in score_order if index >= protected_top_k]
    order = protected + tail

    if set(order[:protected_top_k]) != set(range(protected_top_k)):
        raise RuntimeError("Candidate 2 violated protected top-k membership")
    if set(order[protected_top_k:]) != set(range(protected_top_k, len(scores))):
        raise RuntimeError("Candidate 2 violated frozen tail membership")
    return order


def _assert_model_identity(
    candidate1_policy: dict[str, Any],
    candidate2_policy: dict[str, Any],
) -> None:
    candidate1 = candidate1_policy["candidate"]
    candidate2 = candidate2_policy["model"]
    for key in ("model_repo", "revision", "model_file", "model_file_sha256", "license"):
        if str(candidate1[key]) != str(candidate2[key]):
            raise RuntimeError(f"Candidate 2 changed frozen model identity field {key}")
    if str(candidate1_policy["runtime"]["fastembed"]) != str(candidate2["fastembed"]):
        raise RuntimeError("Candidate 2 changed the frozen fastembed version")
    if bool(candidate1_policy["runtime"]["hosted_inference"]) is not False:
        raise RuntimeError("Candidate 1 runtime unexpectedly allowed hosted inference")
    if bool(candidate2["hosted_inference"]) is not False:
        raise RuntimeError("Candidate 2 unexpectedly allows hosted inference")


async def _run() -> None:
    verify_holdout_freeze()

    policy = _load_object(POLICY_PATH)
    candidate1_policy = _load_object(CANDIDATE1_POLICY_PATH)
    candidate2_policy = _load_object(CANDIDATE2_POLICY_PATH)
    _assert_model_identity(candidate1_policy, candidate2_policy)

    formal = candidate2_policy["formal_evaluation"]
    if formal["state"] != "FROZEN_HOLDOUT_PENDING_EXECUTION":
        raise RuntimeError("holdout executor requires the frozen pending-execution state")
    if formal["execution_enabled_in_freeze_commit"] is not False:
        raise RuntimeError(
            "Candidate 2 policy must record that the freeze commit had execution disabled"
        )

    expected_fastembed = str(candidate2_policy["model"]["fastembed"])
    actual_fastembed = importlib.metadata.version("fastembed")
    if actual_fastembed != expected_fastembed:
        raise RuntimeError(
            f"Phase 3F holdout requires fastembed {expected_fastembed}, got {actual_fastembed}"
        )

    corpus = _load_jsonl(CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    id_to_url = _validate_fixture(
        corpus,
        queries,
        minimum_queries=int(_load_object(
            ROOT / "evaluation" / "phase3f_holdout_protocol.json"
        )["minimum_queries"]),
    )
    url_to_id = {url: document_id for document_id, url in id_to_url.items()}

    candidate_pool_size = int(candidate2_policy["candidate_pool_size"])
    protected_top_k = int(candidate2_policy["protected_top_k"])
    if candidate_pool_size != int(policy["reranker_scope"]["candidate_pool_size"]):
        raise RuntimeError("Candidate 2 changed the frozen candidate pool size")

    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )
    output_path = os.environ.get("PHASE3F_CANDIDATE2_HOLDOUT_OUTPUT")

    settings = Settings(opensearch_url=opensearch_url)
    if settings.search_mode != "hybrid" or settings.dense_enabled is not True:
        raise RuntimeError("holdout requires the certified hybrid+dense defaults")

    store = OpenSearchStore(settings)
    embedder = DenseEmbedder(
        model_name=settings.dense_model_name,
        dimension=settings.dense_dimension,
        query_prefix=settings.dense_query_prefix,
        passage_prefix=settings.dense_passage_prefix,
    )

    hybrid_metrics: list[RetrievalMetrics] = []
    candidate2_metrics: list[RetrievalMetrics] = []
    candidate_recall_at_10: list[float] = []
    first_stage_latencies: list[float] = []
    rerank_latencies: list[float] = []
    transform_latencies: list[float] = []
    added_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    rerank_cpu_seconds: list[float] = []
    query_details: list[dict[str, Any]] = []

    try:
        await _clear_indices(store, settings)
        indexed_vector_chunks = await _seed_corpus_with_vectors(store, embedder, corpus)
        if indexed_vector_chunks < len(corpus):
            raise RuntimeError("holdout vector seeding wrote fewer chunks than documents")

        rss_before_model_mib = _peak_rss_mib()
        reranker, model_report = await asyncio.to_thread(
            _prepare_reranker,
            candidate1_policy["candidate"],
        )
        rss_after_model_load_mib = _peak_rss_mib()

        timeout = httpx.Timeout(180.0, connect=10.0)
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as client:
            warm_body, _ = await _post_search(
                client,
                query=queries[0]["query"],
                mode="hybrid",
                limit=candidate_pool_size,
            )
            _assert_mode(warm_body, "hybrid")
            warm_results = warm_body.get("results")
            if not isinstance(warm_results, list) or len(warm_results) != candidate_pool_size:
                raise RuntimeError("holdout warmup did not return the frozen top-10 pool")
            warm_documents = [
                _candidate_text(result)
                for result in warm_results
                if isinstance(result, dict)
            ]
            if len(warm_documents) != candidate_pool_size:
                raise RuntimeError("holdout warmup contains invalid candidate results")
            _, warmup_ms, warmup_cpu_s = await asyncio.to_thread(
                _score_timed,
                reranker,
                queries[0]["query"],
                warm_documents,
            )

            for query in queries:
                body, first_stage_ms = await _post_search(
                    client,
                    query=query["query"],
                    mode="hybrid",
                    limit=candidate_pool_size,
                )
                _assert_mode(body, "hybrid")
                raw_results = body.get("results")
                if not isinstance(raw_results, list) or len(raw_results) != candidate_pool_size:
                    raise RuntimeError(
                        f"query {query['id']} did not return exactly "
                        f"{candidate_pool_size} candidates"
                    )
                if not all(isinstance(result, dict) for result in raw_results):
                    raise RuntimeError(f"query {query['id']} returned an invalid candidate")
                results = list(raw_results)
                first_stage_ranking = _ranking_from_results(results, url_to_id)
                if len(set(first_stage_ranking)) != candidate_pool_size:
                    raise RuntimeError(f"query {query['id']} returned duplicate documents")

                documents = [_candidate_text(result) for result in results]
                scores, rerank_ms, rerank_cpu_s = await asyncio.to_thread(
                    _score_timed,
                    reranker,
                    query["query"],
                    documents,
                )

                transform_started = time.perf_counter()
                order = _candidate2_order(scores, protected_top_k)
                candidate2_ranking = [first_stage_ranking[index] for index in order]
                transform_ms = (time.perf_counter() - transform_started) * 1000.0

                if set(candidate2_ranking[:protected_top_k]) != set(
                    first_stage_ranking[:protected_top_k]
                ):
                    raise RuntimeError(
                        f"Candidate 2 top-5 membership invariant failed for {query['id']}"
                    )
                if set(candidate2_ranking[protected_top_k:]) != set(
                    first_stage_ranking[protected_top_k:]
                ):
                    raise RuntimeError(
                        f"Candidate 2 tail membership invariant failed for {query['id']}"
                    )

                hybrid_metric = evaluate_ranking(first_stage_ranking, query["relevance"])
                candidate2_metric = evaluate_ranking(candidate2_ranking, query["relevance"])
                if candidate2_metric.recall_at_5 != hybrid_metric.recall_at_5:
                    raise RuntimeError(
                        f"Candidate 2 Recall@5 invariant failed for {query['id']}"
                    )

                candidate_recall = recall_at_k(
                    first_stage_ranking,
                    query["relevance"],
                    k=candidate_pool_size,
                )
                added_ms = rerank_ms + transform_ms
                end_to_end_ms = first_stage_ms + added_ms

                hybrid_metrics.append(hybrid_metric)
                candidate2_metrics.append(candidate2_metric)
                candidate_recall_at_10.append(candidate_recall)
                first_stage_latencies.append(first_stage_ms)
                rerank_latencies.append(rerank_ms)
                transform_latencies.append(transform_ms)
                added_latencies.append(added_ms)
                end_to_end_latencies.append(end_to_end_ms)
                rerank_cpu_seconds.append(rerank_cpu_s)
                query_details.append(
                    {
                        "id": query["id"],
                        "language": query.get("language"),
                        "family": query.get("family"),
                        "hybrid_top10": first_stage_ranking,
                        "candidate2_top10": candidate2_ranking,
                        "hybrid_ndcg_at_10": round(hybrid_metric.ndcg_at_10, 6),
                        "candidate2_ndcg_at_10": round(candidate2_metric.ndcg_at_10, 6),
                        "candidate_recall_at_10": round(candidate_recall, 6),
                        "first_stage_latency_ms": round(first_stage_ms, 3),
                        "rerank_latency_ms": round(rerank_ms, 3),
                        "transform_latency_ms": round(transform_ms, 6),
                        "added_latency_ms": round(added_ms, 3),
                        "end_to_end_latency_ms": round(end_to_end_ms, 3),
                    }
                )
    finally:
        await store.close()

    rss_peak_mib = _peak_rss_mib()
    rss_delta_mib = max(0.0, rss_peak_mib - rss_before_model_mib)
    hybrid = _metric_dict(macro_average(hybrid_metrics))
    candidate2 = _metric_dict(macro_average(candidate2_metrics))
    candidate_recall_mean = mean(candidate_recall_at_10)

    promotion = policy["promotion"]
    ndcg_delta = round(candidate2["ndcg_at_10"] - hybrid["ndcg_at_10"], 6)
    mrr_regression = round(hybrid["mrr_at_10"] - candidate2["mrr_at_10"], 6)
    recall_regression = round(hybrid["recall_at_5"] - candidate2["recall_at_5"], 6)
    added_p95_ms = _percentile(added_latencies, 0.95)

    checks = {
        "candidate_recall_at_10": (
            candidate_recall_mean >= float(policy["baseline"]["minimum_candidate_recall_at_10"])
        ),
        "top5_membership_preserved": True,
        "ndcg_material_improvement": (
            ndcg_delta >= float(promotion["ndcg_at_10_min_delta_vs_frozen_hybrid"])
        ),
        "mrr_no_regression": (
            mrr_regression <= float(promotion["max_mrr_at_10_regression_vs_frozen_hybrid"])
        ),
        "recall_no_regression": (
            recall_regression <= float(promotion["max_recall_at_5_regression_vs_frozen_hybrid"])
        ),
        "p95_added_latency": (
            added_p95_ms <= float(promotion["p95_added_latency_ms_max"])
        ),
        "peak_rss_delta": (
            rss_delta_mib <= float(promotion["peak_rss_delta_mib_max"])
        ),
    }
    decision = "PASS_HOLDOUT" if all(checks.values()) else "REJECT_HOLDOUT"

    report = {
        "schema_version": 1,
        "phase": "3F",
        "experiment": candidate2_policy["experiment"],
        "decision": decision,
        "promotion_eligible": decision == "PASS_HOLDOUT",
        "documents": len(corpus),
        "queries": len(queries),
        "candidate_pool_size": candidate_pool_size,
        "protected_top_k": protected_top_k,
        "aggregate": {
            "holdout_hybrid": hybrid,
            "candidate2_top5_preserving": candidate2,
            "candidate_recall_at_10_mean": round(candidate_recall_mean, 6),
            "ndcg_at_10_delta_vs_holdout_hybrid": ndcg_delta,
            "mrr_regression_vs_holdout_hybrid": mrr_regression,
            "recall_at_5_regression_vs_holdout_hybrid": recall_regression,
        },
        "latency_ms": {
            "first_stage_hybrid_p95": round(_percentile(first_stage_latencies, 0.95), 3),
            "reranker_warmup": round(warmup_ms, 3),
            "reranker_p50": round(_percentile(rerank_latencies, 0.50), 3),
            "reranker_p95": round(_percentile(rerank_latencies, 0.95), 3),
            "partition_transform_p95": round(_percentile(transform_latencies, 0.95), 6),
            "added_p50": round(_percentile(added_latencies, 0.50), 3),
            "added_p95": round(added_p95_ms, 3),
            "end_to_end_p95": round(_percentile(end_to_end_latencies, 0.95), 3),
        },
        "cpu_seconds": {
            "reranker_warmup": round(warmup_cpu_s, 6),
            "reranker_total": round(sum(rerank_cpu_seconds), 6),
        },
        "memory_mib": {
            "rss_before_model": round(rss_before_model_mib, 3),
            "rss_after_model_load": round(rss_after_model_load_mib, 3),
            "rss_peak_process": round(rss_peak_mib, 3),
            "peak_rss_delta": round(rss_delta_mib, 3),
            "measurement": "ru_maxrss process peak delta; retained for Phase 3F gate compatibility",
        },
        "model": {
            "model_repo": candidate2_policy["model"]["model_repo"],
            "revision": candidate2_policy["model"]["revision"],
            "model_file": candidate2_policy["model"]["model_file"],
            "model_file_sha256": model_report["model_file_sha256"],
            "fastembed": actual_fastembed,
            "download_ms": model_report["download_ms"],
            "load_ms": model_report["load_ms"],
            "load_cpu_s": model_report["load_cpu_s"],
        },
        "checks": checks,
        "queries_detail": query_details,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
    print(f"Phase 3F Candidate 2 independent holdout: {decision}")


if __name__ == "__main__":
    asyncio.run(_run())
