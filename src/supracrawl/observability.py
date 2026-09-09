from __future__ import annotations

import math
from threading import Lock
from typing import Any

from .reranking import ControlledSearchExecution

METRICS_SCHEMA_VERSION = 1
_CAPACITY_REASON_FRAGMENT = "reranker capacity saturated after "


class SearchMetrics:
    """Process-local operational counters for completed search requests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._search_requests_total = 0
        self._search_success_total = 0
        self._search_backend_errors_total = 0
        self._retrieval_degraded_total = 0
        self._reranker_enabled_requests_total = 0
        self._reranker_used_total = 0
        self._reranker_degraded_total = 0
        self._reranker_capacity_fallback_total = 0
        self._reranker_other_degradation_total = 0
        self._reranker_queue_wait_observations_total = 0
        self._reranker_queue_wait_ms_sum = 0.0
        self._reranker_queue_wait_ms_max = 0.0
        self._reranker_inference_observations_total = 0
        self._reranker_inference_ms_sum = 0.0
        self._reranker_inference_ms_max = 0.0

    def record_request(self) -> None:
        with self._lock:
            self._search_requests_total += 1

    def record_backend_error(self) -> None:
        with self._lock:
            self._search_backend_errors_total += 1

    def record_execution(self, execution: ControlledSearchExecution) -> None:
        with self._lock:
            self._search_success_total += 1
            if execution.degraded:
                self._retrieval_degraded_total += 1
            if execution.reranker_enabled:
                self._reranker_enabled_requests_total += 1
            if execution.reranker_used:
                self._reranker_used_total += 1
            if execution.reranker_degraded:
                self._reranker_degraded_total += 1
                reason = execution.reranker_degradation_reason or ""
                if _CAPACITY_REASON_FRAGMENT in reason:
                    self._reranker_capacity_fallback_total += 1
                else:
                    self._reranker_other_degradation_total += 1

            queue_wait_ms = execution.reranker_queue_wait_ms
            if (
                queue_wait_ms is not None
                and queue_wait_ms >= 0.0
                and math.isfinite(queue_wait_ms)
            ):
                self._reranker_queue_wait_observations_total += 1
                self._reranker_queue_wait_ms_sum += queue_wait_ms
                self._reranker_queue_wait_ms_max = max(
                    self._reranker_queue_wait_ms_max,
                    queue_wait_ms,
                )

            inference_ms = execution.reranker_inference_ms
            if (
                inference_ms is not None
                and inference_ms >= 0.0
                and math.isfinite(inference_ms)
            ):
                self._reranker_inference_observations_total += 1
                self._reranker_inference_ms_sum += inference_ms
                self._reranker_inference_ms_max = max(
                    self._reranker_inference_ms_max,
                    inference_ms,
                )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": METRICS_SCHEMA_VERSION,
                "scope": "process",
                "search_requests_total": self._search_requests_total,
                "search_success_total": self._search_success_total,
                "search_backend_errors_total": self._search_backend_errors_total,
                "retrieval_degraded_total": self._retrieval_degraded_total,
                "reranker_enabled_requests_total": (
                    self._reranker_enabled_requests_total
                ),
                "reranker_used_total": self._reranker_used_total,
                "reranker_degraded_total": self._reranker_degraded_total,
                "reranker_capacity_fallback_total": (
                    self._reranker_capacity_fallback_total
                ),
                "reranker_other_degradation_total": (
                    self._reranker_other_degradation_total
                ),
                "reranker_queue_wait_observations_total": (
                    self._reranker_queue_wait_observations_total
                ),
                "reranker_queue_wait_ms_sum": self._reranker_queue_wait_ms_sum,
                "reranker_queue_wait_ms_max": self._reranker_queue_wait_ms_max,
                "reranker_inference_observations_total": (
                    self._reranker_inference_observations_total
                ),
                "reranker_inference_ms_sum": self._reranker_inference_ms_sum,
                "reranker_inference_ms_max": self._reranker_inference_ms_max,
            }
