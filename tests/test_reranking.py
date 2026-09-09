from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from supracrawl.config import Settings
from supracrawl.reranking import (
    RERANKER_MAX_CONCURRENCY,
    RERANKER_MODEL_REPO,
    RERANKER_PROTECTED_TOP_K,
    RERANKER_REVISION,
    RERANKER_STRATEGY,
    ControlledRerankingSearchService,
    LocalCrossEncoderReranker,
    RerankerBackendError,
    _top5_preserving_indices,
)
from supracrawl.retrieval import SearchExecution


def _result(document_id: str, position: int) -> dict[str, Any]:
    return {
        "title": f"Title {document_id}",
        "url": f"https://example.com/{document_id}",
        "description": f"Description {document_id}",
        "position": position,
        "score": 1.0 / position,
        "metadata": {"document_id": document_id},
    }


def _hybrid_execution(count: int = 12) -> SearchExecution:
    return SearchExecution(
        results=[_result(f"doc-{index}", index) for index in range(1, count + 1)],
        mode_requested="hybrid",
        mode_used="hybrid",
    )


class _BaseService:
    def __init__(self, execution: SearchExecution) -> None:
        self.execution = execution
        self.calls: list[tuple[str, int, str | None]] = []

    async def search(self, query: str, limit: int, *, mode=None) -> SearchExecution:
        self.calls.append((query, limit, mode))
        return SearchExecution(
            results=self.execution.results[:limit],
            mode_requested=self.execution.mode_requested,
            mode_used=self.execution.mode_used,
            degraded=self.execution.degraded,
            degradation_reason=self.execution.degradation_reason,
        )


class _NeverReranker:
    async def rerank(self, _query: str, _results: list[dict[str, Any]]):
        raise AssertionError("reranker must not be called")


class _PassReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.calls.append((query, [item["metadata"]["document_id"] for item in results]))
        return results


class _BrokenReranker:
    async def rerank(self, _query: str, _results: list[dict[str, Any]]):
        raise RerankerBackendError("fixture failure")


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return self.scores[: len(documents)]


class _ConcurrencyCrossEncoder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def rerank(self, _query: str, documents: list[str]) -> list[float]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return [1.0] * len(documents)
        finally:
            with self._lock:
                self.active -= 1


def test_settings_keep_controlled_reranker_disabled_by_default() -> None:
    settings = Settings()
    assert settings.reranker_enabled is False
    assert settings.reranker_warmup_on_startup is False


def test_top5_preserving_order_keeps_partition_membership_and_stable_ties() -> None:
    scores = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.0]
    order = _top5_preserving_indices(scores, RERANKER_PROTECTED_TOP_K)

    assert order == [1, 3, 4, 2, 0, 6, 8, 7, 5, 9]
    assert set(order[:5]) == set(range(5))
    assert set(order[5:]) == set(range(5, 10))

    tied = _top5_preserving_indices([1.0] * 10, 5)
    assert tied == list(range(10))


@pytest.mark.asyncio
async def test_local_reranker_only_reorders_top10_and_preserves_rrf_scores() -> None:
    results = [_result(f"doc-{index}", index) for index in range(1, 13)]
    original_scores = [item["score"] for item in results]
    runtime = LocalCrossEncoderReranker()
    model = _FakeCrossEncoder(
        [0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6, 0.0]
    )
    runtime._model = model

    reranked = await runtime.rerank("  useful query  ", results)
    reranked_ids = [item["metadata"]["document_id"] for item in reranked]

    assert reranked_ids[:5] == ["doc-2", "doc-4", "doc-5", "doc-3", "doc-1"]
    assert set(reranked_ids[:5]) == {f"doc-{index}" for index in range(1, 6)}
    assert set(reranked_ids[5:10]) == {f"doc-{index}" for index in range(6, 11)}
    assert reranked_ids[10:] == ["doc-11", "doc-12"]
    assert [item["position"] for item in reranked] == list(range(1, 13))

    score_by_id = {
        item["metadata"]["document_id"]: item["score"]
        for item in reranked
    }
    for index in range(1, 13):
        assert score_by_id[f"doc-{index}"] == original_scores[index - 1]

    first = reranked[0]["metadata"]
    assert first["reranker_model"] == RERANKER_MODEL_REPO
    assert first["reranker_revision"] == RERANKER_REVISION
    assert first["reranker_strategy"] == RERANKER_STRATEGY
    assert first["reranker_score"] == pytest.approx(0.9)
    assert first["first_stage_position"] == 2
    assert len(model.calls) == 1
    assert model.calls[0][0] == "useful query"
    assert model.calls[0][1][0] == "Title doc-1\nDescription doc-1"


@pytest.mark.asyncio
async def test_warmup_is_singleflight() -> None:
    runtime = LocalCrossEncoderReranker()
    model = _FakeCrossEncoder([1.0])
    calls = 0

    def load_model():
        nonlocal calls
        calls += 1
        time.sleep(0.02)
        return model

    runtime._load_model_sync = load_model  # type: ignore[method-assign]
    await asyncio.gather(*(runtime.warmup() for _ in range(4)))

    assert calls == 1
    assert runtime._model is model


@pytest.mark.asyncio
async def test_failed_load_is_latched_until_process_restart() -> None:
    runtime = LocalCrossEncoderReranker()
    calls = 0

    def fail_load():
        nonlocal calls
        calls += 1
        raise RerankerBackendError("fixture load failure")

    runtime._load_model_sync = fail_load  # type: ignore[method-assign]

    with pytest.raises(RerankerBackendError, match="fixture load failure"):
        await runtime.warmup()
    with pytest.raises(RerankerBackendError, match="load disabled after failure"):
        await runtime.warmup()

    assert calls == 1


@pytest.mark.asyncio
async def test_local_reranker_bounds_inference_concurrency() -> None:
    runtime = LocalCrossEncoderReranker()
    model = _ConcurrencyCrossEncoder()
    runtime._model = model
    results = [_result(f"doc-{index}", index) for index in range(1, 11)]

    first, second = await asyncio.gather(
        runtime.rerank("query one", results),
        runtime.rerank("query two", results),
    )

    assert RERANKER_MAX_CONCURRENCY == 1
    assert model.max_active == 1
    assert len(first) == len(second) == 10


@pytest.mark.asyncio
async def test_disabled_wrapper_is_transparent_and_does_not_widen_first_stage() -> None:
    base = _BaseService(_hybrid_execution())
    service = ControlledRerankingSearchService(
        Settings(reranker_enabled=False),
        base,  # type: ignore[arg-type]
        _NeverReranker(),
    )

    execution = await service.search("query", 3, mode="hybrid")

    assert base.calls == [("query", 3, "hybrid")]
    assert [item["metadata"]["document_id"] for item in execution.results] == [
        "doc-1",
        "doc-2",
        "doc-3",
    ]
    assert execution.reranker_enabled is False
    assert execution.reranker_used is False
    assert execution.reranker_degraded is False


@pytest.mark.asyncio
async def test_enabled_wrapper_widens_to_frozen_pool_and_uses_reranker() -> None:
    base = _BaseService(_hybrid_execution())
    reranker = _PassReranker()
    service = ControlledRerankingSearchService(
        Settings(reranker_enabled=True),
        base,  # type: ignore[arg-type]
        reranker,
    )

    execution = await service.search("query", 3, mode="hybrid")

    assert base.calls == [("query", 10, "hybrid")]
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 10
    assert len(execution.results) == 3
    assert execution.mode_used == "hybrid"
    assert execution.degraded is False
    assert execution.reranker_enabled is True
    assert execution.reranker_used is True
    assert execution.reranker_degraded is False


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_first_stage_hybrid() -> None:
    base = _BaseService(_hybrid_execution())
    service = ControlledRerankingSearchService(
        Settings(reranker_enabled=True),
        base,  # type: ignore[arg-type]
        _BrokenReranker(),
    )

    execution = await service.search("query", 4, mode="hybrid")

    assert [item["metadata"]["document_id"] for item in execution.results] == [
        "doc-1",
        "doc-2",
        "doc-3",
        "doc-4",
    ]
    assert execution.mode_used == "hybrid"
    assert execution.degraded is False
    assert execution.reranker_used is False
    assert execution.reranker_degraded is True
    assert "fixture failure" in (execution.reranker_degradation_reason or "")


@pytest.mark.asyncio
async def test_hybrid_first_stage_degradation_skips_reranker() -> None:
    execution = SearchExecution(
        results=[_result("doc-1", 1)],
        mode_requested="hybrid",
        mode_used="bm25",
        degraded=True,
        degradation_reason="vector unavailable",
    )
    base = _BaseService(execution)
    service = ControlledRerankingSearchService(
        Settings(reranker_enabled=True),
        base,  # type: ignore[arg-type]
        _NeverReranker(),
    )

    result = await service.search("query", 5, mode="hybrid")

    assert result.mode_used == "bm25"
    assert result.degraded is True
    assert result.degradation_reason == "vector unavailable"
    assert result.reranker_used is False
    assert result.reranker_degraded is True
    assert "hybrid retrieval was unavailable" in (
        result.reranker_degradation_reason or ""
    )


@pytest.mark.asyncio
async def test_explicit_bm25_skips_reranker_without_reporting_degradation() -> None:
    execution = SearchExecution(
        results=[_result("doc-1", 1)],
        mode_requested="bm25",
        mode_used="bm25",
    )
    base = _BaseService(execution)
    service = ControlledRerankingSearchService(
        Settings(reranker_enabled=True),
        base,  # type: ignore[arg-type]
        _NeverReranker(),
    )

    result = await service.search("query", 5, mode="bm25")

    assert result.mode_requested == "bm25"
    assert result.mode_used == "bm25"
    assert result.degraded is False
    assert result.reranker_enabled is True
    assert result.reranker_used is False
    assert result.reranker_degraded is False
    assert result.reranker_degradation_reason is None
