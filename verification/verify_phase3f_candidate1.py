from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import resource
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder
from huggingface_hub import snapshot_download
from verify_phase3f_baseline import _assert_mode, _post_search
from verify_real_vector_retrieval import _clear_indices, _seed_corpus_with_vectors
from verify_retrieval_baseline import _load_jsonl, _percentile, _validate_fixture

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder
from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3f_policy.json"
BASELINE_PATH = ROOT / "evaluation" / "phase3f_baseline_result.json"
CANDIDATE_POLICY_PATH = ROOT / "evaluation" / "phase3f_candidate1_policy.json"
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
EXACT_CORPUS_PATH = ROOT / "evaluation" / "phase3c_exact_corpus.jsonl"
EXACT_QUERIES_PATH = ROOT / "evaluation" / "phase3c_exact_queries.jsonl"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _write_output(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    # GitHub's Ubuntu runners report ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _metric_dict(metrics: RetrievalMetrics) -> dict[str, float]:
    return {
        "mrr_at_10": round(metrics.mrr_at_10, 6),
        "recall_at_5": round(metrics.recall_at_5, 6),
        "ndcg_at_10": round(metrics.ndcg_at_10, 6),
    }


def _candidate_text(result: dict[str, Any]) -> str:
    title = result.get("title")
    description = result.get("description")
    parts = [
        value.strip()
        for value in (title, description)
        if isinstance(value, str) and value.strip()
    ]
    if not parts:
        raise RuntimeError("hybrid candidate has no title or description for reranking")
    return "\n".join(parts)


def _ranking_from_results(
    results: list[dict[str, Any]],
    url_to_id: dict[str, str],
) -> list[str]:
    ranking: list[str] = []
    for result in results:
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("hybrid candidate has no URL")
        ranking.append(url_to_id.get(url, url))
    return ranking


def _results_from_body(body: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    raw_results = body.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("hybrid response has no results list")
    results = [result for result in raw_results if isinstance(result, dict)]
    if len(results) != len(raw_results):
        raise RuntimeError("hybrid response contains an invalid result")
    if not results or len(results) > expected_count:
        raise RuntimeError("hybrid response returned an invalid candidate count")
    return results


def _prepare_reranker(candidate: dict[str, Any]) -> tuple[TextCrossEncoder, dict[str, Any]]:
    repo_id = str(candidate["model_repo"])
    revision = str(candidate["revision"])
    model_file = str(candidate["model_file"])
    expected_sha256 = str(candidate["model_file_sha256"])
    required_files = [str(item) for item in candidate["required_files"]]

    download_started = time.perf_counter()
    model_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=required_files,
        )
    )
    download_ms = (time.perf_counter() - download_started) * 1000.0

    model_path = model_dir / model_file
    if not model_path.is_file():
        raise RuntimeError(f"reranker ONNX file is missing: {model_path}")
    actual_sha256 = _sha256(model_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "reranker ONNX checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    alias = str(candidate["runtime_model_alias"])
    if not any(item["model"] == alias for item in TextCrossEncoder.list_supported_models()):
        TextCrossEncoder.add_custom_model(
            model=alias,
            model_file=model_file,
            sources=ModelSource(hf=repo_id),
        )

    load_cpu_started = time.process_time()
    load_started = time.perf_counter()
    reranker = TextCrossEncoder(
        model_name=alias,
        specific_model_path=str(model_dir),
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0
    load_cpu_s = time.process_time() - load_cpu_started
    return reranker, {
        "download_ms": round(download_ms, 3),
        "load_ms": round(load_ms, 3),
        "load_cpu_s": round(load_cpu_s, 6),
        "model_file_sha256": actual_sha256,
    }


def _score_timed(
    reranker: TextCrossEncoder,
    query: str,
    documents: list[str],
) -> tuple[list[float], float, float]:
    cpu_started = time.process_time()
    started = time.perf_counter()
    scores = [float(score) for score in reranker.rerank(query, documents)]
    latency_ms = (time.perf_counter() - started) * 1000.0
    cpu_s = time.process_time() - cpu_started
    if len(scores) != len(documents):
        raise RuntimeError("reranker score count differs from candidate count")
    return scores, latency_ms, cpu_s


def _reranked_indices(scores: list[float]) -> list[int]:
    # Stable first-stage position is the deterministic tie-breaker.
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


async def _run() -> None:
    policy = _load_object(POLICY_PATH)
    frozen = _load_object(BASELINE_PATH)
    candidate_policy = _load_object(CANDIDATE_POLICY_PATH)
    candidate = candidate_policy["candidate"]
    output_path = os.environ.get("PHASE3F_CANDIDATE1_OUTPUT")
    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )

    if candidate_policy["base_main_sha"] != policy["base_main_sha"]:
        raise RuntimeError("candidate policy and Phase 3F policy disagree on base main SHA")
    candidate_pool_size = int(policy["reranker_scope"]["candidate_pool_size"])
    if int(candidate_policy["candidate_pool_size"]) != candidate_pool_size:
        raise RuntimeError("candidate policy changed the frozen top-N")
    if candidate_policy["ranking_strategy"] != "score_desc_stable_first_stage_ties":
        raise RuntimeError("candidate ranking strategy differs from preregistration")

    expected_fastembed = str(candidate_policy["runtime"]["fastembed"])
    actual_fastembed = importlib.metadata.version("fastembed")
    if actual_fastembed != expected_fastembed:
        raise RuntimeError(
            f"Phase 3F candidate requires fastembed {expected_fastembed}, got {actual_fastembed}"
        )

    corpus = _load_jsonl(CORPUS_PATH) + _load_jsonl(EXACT_CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH) + _load_jsonl(EXACT_QUERIES_PATH)
    id_to_url = _validate_fixture(corpus, queries, minimum_queries=30)
    url_to_id = {url: document_id for document_id, url in id_to_url.items()}

    settings = Settings(opensearch_url=opensearch_url)
    if settings.search_mode != "hybrid" or settings.dense_enabled is not True:
        raise RuntimeError("candidate experiment requires certified hybrid+dense defaults")
    store = OpenSearchStore(settings)
    embedder = DenseEmbedder(
        model_name=settings.dense_model_name,
        dimension=settings.dense_dimension,
        query_prefix=settings.dense_query_prefix,
        passage_prefix=settings.dense_passage_prefix,
    )

    live_hybrid_metrics: list[RetrievalMetrics] = []
    reranked_metrics: list[RetrievalMetrics] = []
    first_stage_latencies: list[float] = []
    rerank_latencies: list[float] = []
    rerank_cpu_seconds: list[float] = []
    query_details: list[dict[str, Any]] = []

    try:
        await _clear_indices(store, settings)
        indexed_vector_chunks = await _seed_corpus_with_vectors(store, embedder, corpus)
        if indexed_vector_chunks < len(corpus):
            raise RuntimeError("candidate experiment seeded fewer vector chunks than documents")

        rss_before_model_mib = _peak_rss_mib()
        reranker, model_report = await asyncio.to_thread(_prepare_reranker, candidate)
        rss_after_model_mib = _peak_rss_mib()

        timeout = httpx.Timeout(180.0, connect=10.0)
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as client:
            warm_body, _ = await _post_search(
                client,
                query=queries[0]["query"],
                mode="hybrid",
                limit=candidate_pool_size,
            )
            _assert_mode(warm_body, "hybrid")
            warm_results = _results_from_body(warm_body, candidate_pool_size)
            warm_documents = [_candidate_text(result) for result in warm_results]
            _, warmup_ms, warmup_cpu_s = await asyncio.to_thread(
                _score_timed,
                reranker,
                queries[0]["query"],
                warm_documents,
            )

            for query in queries:
                body, first_stage_latency_ms = await _post_search(
                    client,
                    query=query["query"],
                    mode="hybrid",
                    limit=candidate_pool_size,
                )
                _assert_mode(body, "hybrid")
                results = _results_from_body(body, candidate_pool_size)
                first_stage_ranking = _ranking_from_results(results, url_to_id)
                documents = [_candidate_text(result) for result in results]
                scores, rerank_latency_ms, rerank_cpu_s = await asyncio.to_thread(
                    _score_timed,
                    reranker,
                    query["query"],
                    documents,
                )
                order = _reranked_indices(scores)
                reranked_ranking = [first_stage_ranking[index] for index in order]

                live_metric = evaluate_ranking(first_stage_ranking, query["relevance"])
                reranked_metric = evaluate_ranking(reranked_ranking, query["relevance"])
                live_hybrid_metrics.append(live_metric)
                reranked_metrics.append(reranked_metric)
                first_stage_latencies.append(first_stage_latency_ms)
                rerank_latencies.append(rerank_latency_ms)
                rerank_cpu_seconds.append(rerank_cpu_s)
                query_details.append(
                    {
                        "id": query["id"],
                        "language": query.get("language"),
                        "hybrid_top10": first_stage_ranking,
                        "reranked_top10": reranked_ranking,
                        "reranker_scores": [round(scores[index], 6) for index in order],
                        "hybrid_ndcg_at_10": round(live_metric.ndcg_at_10, 6),
                        "reranked_ndcg_at_10": round(reranked_metric.ndcg_at_10, 6),
                        "rerank_latency_ms": round(rerank_latency_ms, 3),
                    }
                )
    finally:
        await store.close()

    rss_peak_mib = _peak_rss_mib()
    rss_delta_mib = max(0.0, rss_peak_mib - rss_before_model_mib)
    live_hybrid = _metric_dict(macro_average(live_hybrid_metrics))
    reranked = _metric_dict(macro_average(reranked_metrics))
    frozen_hybrid = frozen["aggregate"]["hybrid"]
    if live_hybrid != frozen_hybrid:
        raise RuntimeError(
            "certified hybrid first stage changed from the frozen Phase 3F baseline: "
            f"live={live_hybrid}, frozen={frozen_hybrid}"
        )

    promotion = policy["promotion"]
    ndcg_delta = round(reranked["ndcg_at_10"] - frozen_hybrid["ndcg_at_10"], 6)
    mrr_regression = round(frozen_hybrid["mrr_at_10"] - reranked["mrr_at_10"], 6)
    recall_regression = round(frozen_hybrid["recall_at_5"] - reranked["recall_at_5"], 6)
    rerank_p95_ms = _percentile(rerank_latencies, 0.95)

    checks = {
        "ndcg_material_improvement": (
            ndcg_delta >= float(promotion["ndcg_at_10_min_delta_vs_frozen_hybrid"])
        ),
        "mrr_no_regression": (
            mrr_regression <= float(promotion["max_mrr_at_10_regression_vs_frozen_hybrid"])
        ),
        "recall_no_regression": (
            recall_regression
            <= float(promotion["max_recall_at_5_regression_vs_frozen_hybrid"])
        ),
        "p95_added_latency": (
            rerank_p95_ms <= float(promotion["p95_added_latency_ms_max"])
        ),
        "peak_rss_delta": (
            rss_delta_mib <= float(promotion["peak_rss_delta_mib_max"])
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "phase": "3F",
        "experiment": candidate_policy["experiment"],
        "candidate": {
            "id": candidate_policy["candidate_id"],
            "model_repo": candidate["model_repo"],
            "revision": candidate["revision"],
            "model_file": candidate["model_file"],
            "model_file_sha256": model_report["model_file_sha256"],
            "license": candidate["license"],
            "fastembed": actual_fastembed,
        },
        "documents": len(corpus),
        "queries": len(queries),
        "candidate_pool_size": candidate_pool_size,
        "aggregate": {
            "frozen_hybrid": frozen_hybrid,
            "live_hybrid": live_hybrid,
            "reranked": reranked,
            "ndcg_delta_vs_frozen_hybrid": ndcg_delta,
            "mrr_regression_vs_frozen_hybrid": mrr_regression,
            "recall_at_5_regression_vs_frozen_hybrid": recall_regression,
        },
        "latency_ms": {
            "first_stage_hybrid_p95": round(_percentile(first_stage_latencies, 0.95), 3),
            "reranker_warmup": round(warmup_ms, 3),
            "reranker_p50": round(_percentile(rerank_latencies, 0.50), 3),
            "reranker_p95": round(rerank_p95_ms, 3),
            "reranker_mean": round(mean(rerank_latencies), 3),
        },
        "cpu": {
            "model_load_s": model_report["load_cpu_s"],
            "warmup_s": round(warmup_cpu_s, 6),
            "rerank_total_s": round(sum(rerank_cpu_seconds), 6),
            "rerank_mean_s": round(mean(rerank_cpu_seconds), 6),
        },
        "memory_mib": {
            "peak_before_model": round(rss_before_model_mib, 3),
            "peak_after_model_load": round(rss_after_model_mib, 3),
            "peak_after_experiment": round(rss_peak_mib, 3),
            "peak_delta": round(rss_delta_mib, 3),
        },
        "model_runtime": model_report,
        "promotion": {"passed": passed, "checks": checks},
        "queries_detail": query_details,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    print("PHASE3F_CANDIDATE1_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    verdict = "PASS" if passed else "REJECT"
    print(f"Phase 3F candidate 1 promotion: {verdict}")
    if output_path:
        await asyncio.to_thread(_write_output, output_path, rendered + "\n")
    print("Phase 3F candidate 1 experiment execution: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
