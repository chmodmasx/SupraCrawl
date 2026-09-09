from supracrawl.observability import METRICS_SCHEMA_VERSION, SearchMetrics
from supracrawl.reranking import ControlledSearchExecution


def _execution(
    *,
    degraded: bool = False,
    reranker_enabled: bool = True,
    reranker_used: bool = False,
    reranker_degraded: bool = False,
    reranker_degradation_reason: str | None = None,
    queue_wait_ms: float | None = None,
    inference_ms: float | None = None,
) -> ControlledSearchExecution:
    return ControlledSearchExecution(
        results=[],
        mode_requested="hybrid",
        mode_used="hybrid",
        degraded=degraded,
        degradation_reason="dense unavailable" if degraded else None,
        reranker_enabled=reranker_enabled,
        reranker_used=reranker_used,
        reranker_degraded=reranker_degraded,
        reranker_degradation_reason=reranker_degradation_reason,
        reranker_queue_wait_ms=queue_wait_ms,
        reranker_inference_ms=inference_ms,
    )


def test_metrics_start_at_zero() -> None:
    metrics = SearchMetrics()
    snapshot = metrics.snapshot()

    assert snapshot["schema_version"] == METRICS_SCHEMA_VERSION == 1
    assert snapshot["scope"] == "process"
    for key, value in snapshot.items():
        if key not in {"schema_version", "scope"}:
            assert value == 0 or value == 0.0


def test_metrics_record_success_and_reranker_timings() -> None:
    metrics = SearchMetrics()
    metrics.record_request()
    metrics.record_execution(
        _execution(
            reranker_used=True,
            queue_wait_ms=12.5,
            inference_ms=87.25,
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot["search_requests_total"] == 1
    assert snapshot["search_success_total"] == 1
    assert snapshot["search_backend_errors_total"] == 0
    assert snapshot["reranker_enabled_requests_total"] == 1
    assert snapshot["reranker_used_total"] == 1
    assert snapshot["reranker_degraded_total"] == 0
    assert snapshot["reranker_queue_wait_observations_total"] == 1
    assert snapshot["reranker_queue_wait_ms_sum"] == 12.5
    assert snapshot["reranker_queue_wait_ms_max"] == 12.5
    assert snapshot["reranker_inference_observations_total"] == 1
    assert snapshot["reranker_inference_ms_sum"] == 87.25
    assert snapshot["reranker_inference_ms_max"] == 87.25


def test_metrics_partition_capacity_and_other_degradation() -> None:
    metrics = SearchMetrics()

    metrics.record_request()
    metrics.record_execution(
        _execution(
            reranker_degraded=True,
            reranker_degradation_reason=(
                "reranker unavailable: reranker capacity saturated after "
                "200 ms admission timeout"
            ),
            queue_wait_ms=201.0,
        )
    )
    metrics.record_request()
    metrics.record_execution(
        _execution(
            reranker_degraded=True,
            reranker_degradation_reason="reranker unavailable: model failed",
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot["reranker_degraded_total"] == 2
    assert snapshot["reranker_capacity_fallback_total"] == 1
    assert snapshot["reranker_other_degradation_total"] == 1
    assert snapshot["reranker_queue_wait_observations_total"] == 1
    assert snapshot["reranker_inference_observations_total"] == 0


def test_metrics_record_backend_error_separately() -> None:
    metrics = SearchMetrics()
    metrics.record_request()
    metrics.record_backend_error()

    snapshot = metrics.snapshot()
    assert snapshot["search_requests_total"] == 1
    assert snapshot["search_success_total"] == 0
    assert snapshot["search_backend_errors_total"] == 1
