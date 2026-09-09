from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx
from verify_retrieval_baseline import _percentile

from supracrawl.config import Settings
from supracrawl.reranking import RERANKER_ADMISSION_TIMEOUT_MS

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3h_policy.json"
PHASE3D_POLICY_PATH = ROOT / "evaluation" / "phase3d_policy.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} must be a JSON object")
    return payload


def _urls(body: dict[str, Any]) -> list[str]:
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Phase 3H response has no results list")
    urls: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("Phase 3H response contains an invalid result")
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("Phase 3H result has no URL")
        urls.append(url)
    return urls


def _assert_hybrid(body: dict[str, Any]) -> None:
    if body.get("success") is not True:
        raise RuntimeError("Phase 3H response is not successful")
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "hybrid":
        raise RuntimeError("Phase 3H request did not remain hybrid")
    if body.get("degraded") is not False:
        raise RuntimeError("Phase 3H first-stage retrieval degraded")
    if body.get("degradation_reason") is not None:
        raise RuntimeError("Phase 3H first-stage reported a degradation reason")
    if body.get("reranker_enabled") is not True:
        raise RuntimeError("Phase 3H canary lost reranker-enabled telemetry")


def _finite_nonnegative(value: Any) -> bool:
    if value is None:
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric >= 0.0


async def _timed_search(
    client: httpx.AsyncClient,
    query: str,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = await client.post(
        "/v1/search",
        json={"query": query, "limit": 10, "mode": "hybrid"},
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(
            f"Phase 3H API returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("Phase 3H API returned non-object JSON")
    return body, latency_ms


async def _run() -> None:
    policy = _load_json(POLICY_PATH)
    phase3d = _load_json(PHASE3D_POLICY_PATH)
    admission = policy["admission_control"]
    live_gate = policy["live_gate"]

    if Settings().reranker_backpressure_enabled is not False:
        raise RuntimeError("Phase 3H backpressure must remain disabled by default")
    if float(admission["admission_timeout_ms"]) != RERANKER_ADMISSION_TIMEOUT_MS:
        raise RuntimeError("Phase 3H policy and runtime admission timeout diverged")

    semantic = phase3d["frozen_spot_queries"]["semantic"]
    if not isinstance(semantic, list) or not semantic:
        raise RuntimeError("Phase 3H could not resolve a frozen semantic spot query")
    query = semantic[0].get("query")
    if not isinstance(query, str) or not query.strip():
        raise RuntimeError("Phase 3H frozen semantic query is invalid")

    baseline_url = os.environ.get(
        "PHASE3H_BASELINE_API_URL",
        "http://127.0.0.1:18082",
    )
    normal_url = os.environ.get(
        "PHASE3H_RERANKER_API_URL",
        "http://127.0.0.1:18083",
    )
    backpressure_url = os.environ.get(
        "PHASE3H_BACKPRESSURE_API_URL",
        "http://127.0.0.1:18085",
    )
    timeout = httpx.Timeout(180.0, connect=10.0)

    async with (
        httpx.AsyncClient(base_url=baseline_url, timeout=timeout) as baseline_client,
        httpx.AsyncClient(base_url=normal_url, timeout=timeout) as normal_client,
        httpx.AsyncClient(base_url=backpressure_url, timeout=timeout) as bp_client,
    ):
        baseline, _ = await _timed_search(baseline_client, query)
        normal, _ = await _timed_search(normal_client, query)
        backpressure, _ = await _timed_search(bp_client, query)

        if baseline.get("reranker_enabled") is not False:
            raise RuntimeError("Phase 3H baseline unexpectedly enables reranking")
        _assert_hybrid(normal)
        _assert_hybrid(backpressure)
        if normal.get("reranker_used") is not True:
            raise RuntimeError("Phase 3H normal canary did not rerank")
        if backpressure.get("reranker_used") is not True:
            raise RuntimeError("Phase 3H backpressure canary failed while unsaturated")
        if backpressure.get("reranker_degraded") is not False:
            raise RuntimeError("Phase 3H backpressure canary degraded while unsaturated")
        if _urls(backpressure) != _urls(normal):
            raise RuntimeError("Phase 3H changed unsaturated reranker ranking")
        if not _finite_nonnegative(backpressure.get("reranker_queue_wait_ms")):
            raise RuntimeError("Phase 3H successful rerank omitted queue timing")
        if not _finite_nonnegative(backpressure.get("reranker_inference_ms")):
            raise RuntimeError("Phase 3H successful rerank omitted inference timing")

        concurrency = int(live_gate["concurrency"])
        responses = await asyncio.gather(
            *(_timed_search(bp_client, query) for _ in range(concurrency))
        )

        baseline_urls = _urls(baseline)
        reranked_urls = _urls(normal)
        latencies: list[float] = []
        queue_waits: list[float] = []
        inference_times: list[float] = []
        used_count = 0
        capacity_fallback_count = 0

        for body, latency_ms in responses:
            _assert_hybrid(body)
            latencies.append(latency_ms)
            queue_wait = body.get("reranker_queue_wait_ms")
            if not _finite_nonnegative(queue_wait):
                raise RuntimeError("Phase 3H burst response omitted queue timing")
            queue_waits.append(float(queue_wait))

            if body.get("reranker_used") is True:
                if body.get("reranker_degraded") is not False:
                    raise RuntimeError("successful Phase 3H rerank reported degradation")
                if _urls(body) != reranked_urls:
                    raise RuntimeError("successful Phase 3H burst ranking changed")
                inference_ms = body.get("reranker_inference_ms")
                if not _finite_nonnegative(inference_ms):
                    raise RuntimeError("successful Phase 3H rerank omitted inference timing")
                inference_times.append(float(inference_ms))
                used_count += 1
                continue

            if body.get("reranker_degraded") is not True:
                raise RuntimeError("Phase 3H burst neither reranked nor degraded")
            reason = body.get("reranker_degradation_reason")
            if not isinstance(reason, str) or "capacity saturated" not in reason:
                raise RuntimeError("Phase 3H burst degradation was not capacity saturation")
            if body.get("reranker_inference_ms") is not None:
                raise RuntimeError("capacity fallback incorrectly reports inference time")
            if _urls(body) != baseline_urls:
                raise RuntimeError("capacity fallback changed first-stage hybrid ranking")
            capacity_fallback_count += 1

        if used_count < 1:
            raise RuntimeError("Phase 3H burst never admitted a reranker request")
        if capacity_fallback_count < 1:
            raise RuntimeError("Phase 3H burst did not exercise capacity fallback")
        if used_count + capacity_fallback_count != concurrency:
            raise RuntimeError("Phase 3H burst accounting is incomplete")

        concurrent_p95_ms = _percentile(latencies, 0.95)
        if concurrent_p95_ms > float(live_gate["concurrent_api_p95_ms_max"]):
            raise RuntimeError(
                "Phase 3H concurrent p95 exceeded preregistered limit: "
                f"{concurrent_p95_ms:.3f} ms"
            )

        recovered, _ = await _timed_search(bp_client, query)
        _assert_hybrid(recovered)
        if recovered.get("reranker_used") is not True:
            raise RuntimeError("Phase 3H capacity saturation latched after the burst")
        if recovered.get("reranker_degraded") is not False:
            raise RuntimeError("Phase 3H post-burst request remained degraded")
        if _urls(recovered) != reranked_urls:
            raise RuntimeError("Phase 3H post-burst reranker ranking changed")

    report = {
        "schema_version": 1,
        "phase": "3H",
        "decision": "PASS_BACKPRESSURE_GATE",
        "admission_timeout_ms": RERANKER_ADMISSION_TIMEOUT_MS,
        "concurrency": concurrency,
        "reranker_used": used_count,
        "capacity_fallbacks": capacity_fallback_count,
        "concurrent_api_p95_ms": round(concurrent_p95_ms, 3),
        "max_queue_wait_ms": round(max(queue_waits), 3),
        "max_inference_ms": (
            round(max(inference_times), 3) if inference_times else None
        ),
        "post_burst_recovered": True,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    output_path = os.environ.get("PHASE3H_BACKPRESSURE_OUTPUT")
    if output_path:
        await asyncio.to_thread(
            Path(output_path).write_text,
            rendered + "\n",
            encoding="utf-8",
        )
    print("Phase 3H reranker backpressure verification: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
