from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3i_policy.json"
PHASE3D_POLICY_PATH = ROOT / "evaluation" / "phase3d_policy.json"

_COUNTER_FIELDS = (
    "search_requests_total",
    "search_success_total",
    "search_backend_errors_total",
    "retrieval_degraded_total",
    "reranker_enabled_requests_total",
    "reranker_used_total",
    "reranker_degraded_total",
    "reranker_capacity_fallback_total",
    "reranker_other_degradation_total",
    "reranker_queue_wait_observations_total",
    "reranker_inference_observations_total",
)
_TIMING_FIELDS = (
    "reranker_queue_wait_ms_sum",
    "reranker_queue_wait_ms_max",
    "reranker_inference_ms_sum",
    "reranker_inference_ms_max",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} must be a JSON object")
    return payload


def _query() -> str:
    policy = _load_json(PHASE3D_POLICY_PATH)
    exact = policy["frozen_spot_queries"]["exact_identifier"]
    if not isinstance(exact, list) or not exact:
        raise RuntimeError("Phase 3D exact-identifier spot queries are unavailable")
    query = exact[0].get("query")
    if not isinstance(query, str) or not query:
        raise RuntimeError("registered Phase 3I query is invalid")
    return query


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> int:
    return int(after[key]) - int(before[key])


def _require_delta(
    after: dict[str, Any],
    before: dict[str, Any],
    key: str,
    expected: int,
) -> None:
    actual = _delta(after, before, key)
    if actual != expected:
        raise RuntimeError(
            f"Phase 3I metric {key} delta changed: expected {expected}, got {actual}"
        )


async def _snapshot(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/v1/metrics")
    if response.status_code != 200:
        raise RuntimeError(
            f"Phase 3I metrics endpoint returned HTTP {response.status_code}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("Phase 3I metrics endpoint returned non-object JSON")
    if body.get("schema_version") != 1 or body.get("scope") != "process":
        raise RuntimeError("Phase 3I metrics schema or scope changed")
    for field in _COUNTER_FIELDS:
        value = body.get(field)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Phase 3I counter {field} is invalid")
    for field in _TIMING_FIELDS:
        value = body.get(field)
        if not isinstance(value, int | float) or float(value) < 0.0:
            raise RuntimeError(f"Phase 3I timing {field} is invalid")
    return body


async def _search(client: httpx.AsyncClient, query: str) -> dict[str, Any]:
    response = await client.post(
        "/v1/search",
        json={"query": query, "limit": 10, "mode": "hybrid"},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Phase 3I search returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict) or body.get("success") is not True:
        raise RuntimeError("Phase 3I search response is not successful")
    if body.get("mode_used") != "hybrid":
        raise RuntimeError("Phase 3I search did not remain hybrid")
    return body


async def _run(policy: dict[str, Any], query: str) -> dict[str, Any]:
    requirements = policy["requirements"]
    concurrency = int(requirements["backpressure_burst_concurrency"])
    timeout = httpx.Timeout(60.0, connect=10.0)

    baseline_url = os.environ.get(
        "PHASE3I_BASELINE_API_URL",
        "http://127.0.0.1:18082",
    )
    reranker_url = os.environ.get(
        "PHASE3I_RERANKER_API_URL",
        "http://127.0.0.1:18083",
    )
    backpressure_url = os.environ.get(
        "PHASE3I_BACKPRESSURE_API_URL",
        "http://127.0.0.1:18085",
    )

    async with (
        httpx.AsyncClient(base_url=baseline_url, timeout=timeout) as baseline,
        httpx.AsyncClient(base_url=reranker_url, timeout=timeout) as reranker,
        httpx.AsyncClient(base_url=backpressure_url, timeout=timeout) as backpressure,
    ):
        baseline_before = await _snapshot(baseline)
        if baseline_before.get("reranker_enabled") is not False:
            raise RuntimeError("Phase 3I baseline metrics config changed")
        await _search(baseline, query)
        baseline_after = await _snapshot(baseline)

        _require_delta(
            baseline_after,
            baseline_before,
            "search_requests_total",
            1,
        )
        _require_delta(
            baseline_after,
            baseline_before,
            "search_success_total",
            1,
        )
        _require_delta(
            baseline_after,
            baseline_before,
            "reranker_enabled_requests_total",
            0,
        )

        reranker_before = await _snapshot(reranker)
        if reranker_before.get("reranker_enabled") is not True:
            raise RuntimeError("Phase 3I reranker metrics config changed")
        if reranker_before.get("reranker_backpressure_enabled") is not False:
            raise RuntimeError(
                "Phase 3I historical reranker unexpectedly uses backpressure"
            )
        reranker_body = await _search(reranker, query)
        if reranker_body.get("reranker_used") is not True:
            raise RuntimeError("Phase 3I unsaturated reranker did not run")
        reranker_after = await _snapshot(reranker)

        for field in (
            "search_requests_total",
            "search_success_total",
            "reranker_enabled_requests_total",
            "reranker_used_total",
            "reranker_queue_wait_observations_total",
            "reranker_inference_observations_total",
        ):
            _require_delta(reranker_after, reranker_before, field, 1)
        _require_delta(
            reranker_after,
            reranker_before,
            "reranker_degraded_total",
            0,
        )

        backpressure_before = await _snapshot(backpressure)
        if backpressure_before.get("reranker_enabled") is not True:
            raise RuntimeError("Phase 3I backpressure reranker is not enabled")
        if backpressure_before.get("reranker_backpressure_enabled") is not True:
            raise RuntimeError("Phase 3I backpressure metrics config changed")

        await asyncio.gather(
            *[_search(backpressure, query) for _ in range(concurrency)]
        )
        backpressure_after = await _snapshot(backpressure)

        for field in (
            "search_requests_total",
            "search_success_total",
            "reranker_enabled_requests_total",
        ):
            _require_delta(
                backpressure_after,
                backpressure_before,
                field,
                concurrency,
            )

        used = _delta(
            backpressure_after,
            backpressure_before,
            "reranker_used_total",
        )
        capacity = _delta(
            backpressure_after,
            backpressure_before,
            "reranker_capacity_fallback_total",
        )
        other = _delta(
            backpressure_after,
            backpressure_before,
            "reranker_other_degradation_total",
        )
        degraded = _delta(
            backpressure_after,
            backpressure_before,
            "reranker_degraded_total",
        )
        queue_observations = _delta(
            backpressure_after,
            backpressure_before,
            "reranker_queue_wait_observations_total",
        )
        inference_observations = _delta(
            backpressure_after,
            backpressure_before,
            "reranker_inference_observations_total",
        )

        if used < int(requirements["reranker_used_min"]):
            raise RuntimeError("Phase 3I burst did not record a successful rerank")
        if capacity < int(requirements["capacity_fallback_min"]):
            raise RuntimeError("Phase 3I burst did not record capacity fallback")
        if other != 0:
            raise RuntimeError("Phase 3I burst recorded an unexpected degradation")
        if degraded != capacity:
            raise RuntimeError("Phase 3I degradation partition is inconsistent")
        if used + capacity != concurrency:
            raise RuntimeError("Phase 3I burst accounting does not cover all requests")
        if queue_observations != concurrency:
            raise RuntimeError("Phase 3I queue telemetry did not cover the burst")
        if inference_observations != used:
            raise RuntimeError("Phase 3I inference telemetry count changed")

    return {
        "schema_version": 1,
        "phase": "3I",
        "gate": policy["gate"],
        "decision": "PASS_METRICS_GATE",
        "scope": "process",
        "baseline_delta_requests": 1,
        "reranker_delta_requests": 1,
        "backpressure_burst_requests": concurrency,
        "backpressure_reranker_used": used,
        "backpressure_capacity_fallbacks": capacity,
        "backpressure_other_degradations": other,
        "queue_observations": queue_observations,
        "inference_observations": inference_observations,
    }


def main() -> None:
    policy = _load_json(POLICY_PATH)
    query = _query()
    report = asyncio.run(_run(policy, query))
    output = Path(
        os.environ.get(
            "PHASE3I_METRICS_OUTPUT",
            "phase3i-metrics-report.json",
        )
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Phase 3I operational metrics verification: PASS")


if __name__ == "__main__":
    main()
