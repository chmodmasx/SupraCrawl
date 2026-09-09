from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from supracrawl.config import Settings
from supracrawl.reranking import (
    RERANKER_ADMISSION_TIMEOUT_MS,
    ControlledRerankingSearchService,
    LocalCrossEncoderReranker,
    RerankerCapacityError,
    RerankerTelemetry,
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


class _SleepingCrossEncoder:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def rerank(self, _query: str, documents: list[str]) -> list[float]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay_s)
            return [float(len(documents) - index) for index in range(len(documents))]
        finally:
            with self._lock:
                self.active -= 1


class _BaseService:
    async def search(self, _query: str, limit: int, *, mode=None) -> SearchExecution:
        return SearchExecution(
            results=[_result(f"doc-{index}", index) for index in range(1, limit + 1)],
            mode_requested=mode or "hybrid",
            mode_used=mode or "hybrid",
        )


class _CapacityReranker:
    async def rerank(
        self,
        _query: str,
        _results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raise RerankerCapacityError(
            "reranker capacity saturated after 200 ms admission timeout"
        )

    def last_telemetry(self) -> RerankerTelemetry:
        return RerankerTelemetry(queue_wait_ms=200.0)


def test_phase3h_backpressure_remains_disabled_by_default() -> None:
    settings = Settings()
    assert settings.reranker_backpressure_enabled is False
    assert RERANKER_ADMISSION_TIMEOUT_MS == 200.0


@pytest.mark.asyncio
async def test_successful_rerank_reports_queue_and_inference_timing() -> None:
    runtime = LocalCrossEncoderReranker(backpressure_enabled=True)
    model = _SleepingCrossEncoder(0.01)
    runtime._model = model
    results = [_result(f"doc-{index}", index) for index in range(1, 11)]

    reranked = await runtime.rerank("query", results)
    telemetry = runtime.last_telemetry()

    assert len(reranked) == 10
    assert telemetry.queue_wait_ms is not None
    assert telemetry.queue_wait_ms >= 0.0
    assert telemetry.inference_ms is not None
    assert telemetry.inference_ms >= 0.0
    assert model.max_active == 1


@pytest.mark.asyncio
async def test_backpressure_times_out_without_latching_model_failure() -> None:
    runtime = LocalCrossEncoderReranker(backpressure_enabled=True)
    model = _SleepingCrossEncoder(0.35)
    runtime._model = model
    results = [_result(f"doc-{index}", index) for index in range(1, 11)]

    first = asyncio.create_task(runtime.rerank("query one", results))
    await asyncio.sleep(0.02)

    with pytest.raises(RerankerCapacityError, match="200 ms admission timeout"):
        await runtime.rerank("query two", results)

    saturated = runtime.last_telemetry()
    assert saturated.queue_wait_ms is not None
    assert saturated.queue_wait_ms >= 150.0
    assert saturated.inference_ms is None
    assert runtime._load_error is None

    await first
    recovered = await runtime.rerank("query three", results)

    assert len(recovered) == 10
    assert runtime._load_error is None
    assert model.max_active == 1


@pytest.mark.asyncio
async def test_capacity_fallback_preserves_hybrid_and_reports_wait() -> None:
    service = ControlledRerankingSearchService(
        Settings(reranker_enabled=True, reranker_backpressure_enabled=True),
        _BaseService(),  # type: ignore[arg-type]
        _CapacityReranker(),
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
    assert "capacity saturated" in (execution.reranker_degradation_reason or "")
    assert execution.reranker_queue_wait_ms == pytest.approx(200.0)
    assert execution.reranker_inference_ms is None
