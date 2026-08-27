from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from supracrawl.chunking import Chunk, approx_tokens
from supracrawl.config import Settings
from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average
from supracrawl.extractor import Extraction
from supracrawl.fetcher import FetchResult
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
THRESHOLDS_PATH = ROOT / "evaluation" / "bm25_thresholds.json"
FETCHED_AT = "2026-08-27T00:00:00+00:00"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"expected JSON object in {path}:{line_number}")
        records.append(record)
    return records


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _validate_fixture(
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    minimum_queries: int,
) -> dict[str, str]:
    if len(queries) < minimum_queries:
        raise RuntimeError(f"benchmark has {len(queries)} queries; expected at least {minimum_queries}")

    id_to_url: dict[str, str] = {}
    urls: set[str] = set()
    for document in corpus:
        document_id = document.get("id")
        url = document.get("url")
        title = document.get("title")
        chunks = document.get("chunks")
        if not isinstance(document_id, str) or not document_id:
            raise RuntimeError("every corpus document requires a non-empty string id")
        if document_id in id_to_url:
            raise RuntimeError(f"duplicate corpus id: {document_id}")
        if not isinstance(url, str) or not url or url in urls:
            raise RuntimeError(f"invalid or duplicate corpus URL for {document_id}")
        if not isinstance(title, str) or not title:
            raise RuntimeError(f"missing title for {document_id}")
        if not isinstance(chunks, list) or not chunks:
            raise RuntimeError(f"missing chunks for {document_id}")
        id_to_url[document_id] = url
        urls.add(url)

    query_ids: set[str] = set()
    for query in queries:
        query_id = query.get("id")
        text = query.get("query")
        relevance = query.get("relevance")
        if not isinstance(query_id, str) or not query_id or query_id in query_ids:
            raise RuntimeError(f"invalid or duplicate query id: {query_id!r}")
        query_ids.add(query_id)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"query {query_id} has no text")
        if not isinstance(relevance, dict) or not relevance:
            raise RuntimeError(f"query {query_id} has no relevance judgments")
        for relevant_id, grade in relevance.items():
            if relevant_id not in id_to_url:
                raise RuntimeError(f"query {query_id} references unknown document {relevant_id}")
            if not isinstance(grade, int) or grade <= 0:
                raise RuntimeError(f"query {query_id} has invalid relevance grade for {relevant_id}")

    return id_to_url


async def _delete_index(store: OpenSearchStore, index_name: str) -> None:
    response = await store._request("DELETE", f"/{index_name}")
    if response.status_code not in {200, 404}:
        raise RuntimeError(f"unable to delete benchmark index {index_name}: HTTP {response.status_code}")
    store._indices_ready = False


async def _seed_corpus(store: OpenSearchStore, corpus: list[dict[str, Any]]) -> None:
    for document in corpus:
        chunks: list[Chunk] = []
        markdown_parts: list[str] = []
        for ordinal, raw_chunk in enumerate(document["chunks"]):
            if not isinstance(raw_chunk, dict):
                raise RuntimeError(f"invalid chunk in {document['id']}")
            raw_path = raw_chunk.get("section_path", [])
            text = raw_chunk.get("text")
            if not isinstance(raw_path, list) or not all(isinstance(part, str) for part in raw_path):
                raise RuntimeError(f"invalid section_path in {document['id']}")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"empty chunk in {document['id']}")
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    section_path=tuple(raw_path),
                    text=text,
                    approx_tokens=approx_tokens(text),
                )
            )
            markdown_parts.append(text)

        markdown = "\n\n".join(markdown_parts)
        content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        fetched = FetchResult(
            fetch_url=document["url"],
            final_url=document["url"],
            status_code=200,
            content_type="text/html",
            html="",
            fetched_at=FETCHED_AT,
        )
        extraction = Extraction(
            title=document["title"],
            markdown=markdown,
            canonical_url=document["url"],
            extractor="phase3-benchmark-fixture",
            quality=1.0,
            rendered=False,
        )
        _, indexed_chunks = await store.index_document(
            fetched=fetched,
            extraction=extraction,
            chunks=chunks,
            content_hash=content_hash,
        )
        if indexed_chunks != len(chunks):
            raise RuntimeError(f"chunk count mismatch while indexing {document['id']}")


async def _run() -> None:
    corpus = _load_jsonl(CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(thresholds, dict):
        raise RuntimeError("benchmark thresholds must be a JSON object")

    minimum_queries = int(thresholds["minimum_queries"])
    id_to_url = _validate_fixture(corpus, queries, minimum_queries)
    url_to_id = {url: document_id for document_id, url in id_to_url.items()}

    base_url = os.environ.get("SUPRACRAWL_OPENSEARCH_URL", "http://127.0.0.1:9200")
    settings = Settings(
        opensearch_url=base_url,
        opensearch_documents_index="supracrawl-eval-documents-v1",
        opensearch_chunks_index="supracrawl-eval-chunks-v1",
        opensearch_timeout_s=10.0,
    )
    store = OpenSearchStore(settings)
    per_query: list[RetrievalMetrics] = []
    latencies_ms: list[float] = []
    description_tokens_top5: list[int] = []
    query_report: list[dict[str, Any]] = []

    try:
        await _delete_index(store, settings.opensearch_documents_index)
        await _delete_index(store, settings.opensearch_chunks_index)
        await store.ensure_indices()
        await _seed_corpus(store, corpus)

        for query in queries:
            start = time.perf_counter()
            results = await store.search(query["query"], limit=10)
            latency_ms = (time.perf_counter() - start) * 1000.0
            ranked_ids = [url_to_id.get(result["url"], result["url"]) for result in results]
            metric = evaluate_ranking(ranked_ids, query["relevance"])
            result_tokens = sum(approx_tokens(result["description"]) for result in results[:5])

            per_query.append(metric)
            latencies_ms.append(latency_ms)
            description_tokens_top5.append(result_tokens)
            query_report.append(
                {
                    "id": query["id"],
                    "language": query.get("language"),
                    "query": query["query"],
                    "top5": ranked_ids[:5],
                    "mrr_at_10": round(metric.mrr_at_10, 6),
                    "recall_at_5": round(metric.recall_at_5, 6),
                    "ndcg_at_10": round(metric.ndcg_at_10, 6),
                    "latency_ms": round(latency_ms, 3),
                    "description_tokens_top5": result_tokens,
                }
            )

        aggregate = macro_average(per_query)
        p95_latency_ms = _percentile(latencies_ms, 0.95)
        mean_tokens = mean(description_tokens_top5)
        report = {
            "benchmark": thresholds["benchmark"],
            "documents": len(corpus),
            "queries": len(queries),
            "aggregate": {
                "mrr_at_10": round(aggregate.mrr_at_10, 6),
                "recall_at_5": round(aggregate.recall_at_5, 6),
                "ndcg_at_10": round(aggregate.ndcg_at_10, 6),
                "p50_latency_ms": round(_percentile(latencies_ms, 0.50), 3),
                "p95_latency_ms": round(p95_latency_ms, 3),
                "mean_description_tokens_top5": round(mean_tokens, 3),
            },
            "queries_detail": query_report,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

        failures: list[str] = []
        checks = (
            (aggregate.mrr_at_10, float(thresholds["mrr_at_10_min"]), ">=", "MRR@10"),
            (aggregate.recall_at_5, float(thresholds["recall_at_5_min"]), ">=", "Recall@5"),
            (aggregate.ndcg_at_10, float(thresholds["ndcg_at_10_min"]), ">=", "nDCG@10"),
        )
        for actual, expected, operator, label in checks:
            if actual < expected:
                failures.append(f"{label} {actual:.6f} {operator} {expected:.6f} failed")
        if p95_latency_ms > float(thresholds["p95_latency_ms_max"]):
            failures.append(
                f"p95 latency {p95_latency_ms:.3f}ms > "
                f"{float(thresholds['p95_latency_ms_max']):.3f}ms"
            )
        if mean_tokens > float(thresholds["mean_description_tokens_top5_max"]):
            failures.append(
                f"mean top-5 description tokens {mean_tokens:.3f} > "
                f"{float(thresholds['mean_description_tokens_top5_max']):.3f}"
            )

        if failures:
            raise RuntimeError("Phase 3A benchmark gate failed:\n- " + "\n- ".join(failures))

        print("Phase 3A BM25 baseline verification: PASS")
    finally:
        await _delete_index(store, settings.opensearch_documents_index)
        await _delete_index(store, settings.opensearch_chunks_index)
        await store.close()


if __name__ == "__main__":
    asyncio.run(_run())
