from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from verify_retrieval_baseline import _percentile

from supracrawl.reranking import RERANKER_MAX_CONCURRENCY

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_POLICY_PATH = ROOT / "evaluation" / "phase3g_resource_policy.json"
PHASE3D_POLICY_PATH = ROOT / "evaluation" / "phase3d_policy.json"
PHASE3F_POLICY_PATH = ROOT / "evaluation" / "phase3f_policy.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} must be a JSON object")
    return payload


def _result_urls(body: dict[str, Any]) -> list[str]:
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("search response has no results list")
    urls: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("search response contains an invalid result")
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("search result has no URL")
        urls.append(url)
    return urls


def _assert_response(body: dict[str, Any], *, reranker_enabled: bool) -> None:
    if body.get("success") is not True:
        raise RuntimeError("resource-gate search response is not successful")
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "hybrid":
        raise RuntimeError("resource-gate request did not remain hybrid")
    if body.get("degraded") is not False:
        raise RuntimeError("resource-gate hybrid retrieval degraded")
    if body.get("reranker_enabled") is not reranker_enabled:
        raise RuntimeError("resource-gate reranker_enabled telemetry changed")
    if body.get("reranker_degraded") is not False:
        raise RuntimeError("resource-gate reranker unexpectedly degraded")
    if body.get("reranker_degradation_reason") is not None:
        raise RuntimeError("resource-gate reranker reported a degradation reason")
    if body.get("reranker_used") is not reranker_enabled:
        raise RuntimeError("resource-gate reranker_used telemetry changed")


async def _timed_search(
    client: httpx.AsyncClient,
    *,
    query: str,
    limit: int,
    reranker_enabled: bool,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = await client.post(
        "/v1/search",
        json={"query": query, "limit": limit, "mode": "hybrid"},
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(
            f"resource-gate API returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("resource-gate API returned non-object JSON")
    _assert_response(body, reranker_enabled=reranker_enabled)
    return body, latency_ms


def _process_metrics(container_name: str) -> dict[str, float]:
    snippet = (
        "import json, os, pathlib;"
        "status=pathlib.Path('/proc/1/status').read_text().splitlines();"
        "hwm=int(next(line.split()[1] for line in status if line.startswith('VmHWM:')));"
        "stat=pathlib.Path('/proc/1/stat').read_text().split();"
        "ticks=os.sysconf('SC_CLK_TCK');"
        "cpu=(int(stat[13])+int(stat[14]))/ticks;"
        "print(json.dumps({'vm_hwm_kib':hwm,'cpu_seconds':cpu}))"
    )
    completed = subprocess.run(
        ["docker", "exec", container_name, "python", "-c", snippet],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return {
        "vm_hwm_mib": float(payload["vm_hwm_kib"]) / 1024.0,
        "cpu_seconds": float(payload["cpu_seconds"]),
    }


def _queries(phase3d_policy: dict[str, Any]) -> list[dict[str, str]]:
    frozen = phase3d_policy["frozen_spot_queries"]
    entries = [*frozen["exact_identifier"], *frozen["semantic"]]
    queries: list[dict[str, str]] = []
    for entry in entries:
        query_id = entry.get("id")
        query = entry.get("query")
        if not isinstance(query_id, str) or not isinstance(query, str):
            raise RuntimeError("registered resource-gate query is invalid")
        queries.append({"id": query_id, "query": query})
    return queries


async def _run_batch(
    client: httpx.AsyncClient,
    requests: list[dict[str, str]],
    *,
    limit: int,
    reranker_enabled: bool,
    expected_urls: dict[str, list[str]],
) -> list[float]:
    tasks = [
        _timed_search(
            client,
            query=item["query"],
            limit=limit,
            reranker_enabled=reranker_enabled,
        )
        for item in requests
    ]
    responses = await asyncio.gather(*tasks)
    latencies: list[float] = []
    for item, (body, latency_ms) in zip(requests, responses, strict=True):
        if _result_urls(body) != expected_urls[item["id"]]:
            raise RuntimeError(
                f"concurrent ranking changed for registered query {item['id']}"
            )
        latencies.append(latency_ms)
    return latencies


async def _run() -> None:
    policy = _load_json(RESOURCE_POLICY_PATH)
    phase3d_policy = _load_json(PHASE3D_POLICY_PATH)
    phase3f_policy = _load_json(PHASE3F_POLICY_PATH)
    profile = policy["load_profile"]
    limits = policy["inherited_limits"]
    requirements = policy["requirements"]

    phase3f_promotion = phase3f_policy["promotion"]
    if float(limits["p95_added_latency_ms_max"]) != float(
        phase3f_promotion["p95_added_latency_ms_max"]
    ):
        raise RuntimeError("Phase 3G latency limit diverged from frozen Phase 3F")
    if float(limits["peak_rss_delta_mib_max"]) != float(
        phase3f_promotion["peak_rss_delta_mib_max"]
    ):
        raise RuntimeError("Phase 3G memory limit diverged from frozen Phase 3F")
    if int(requirements["reranker_max_concurrency"]) != RERANKER_MAX_CONCURRENCY:
        raise RuntimeError("resource policy no longer matches reranker concurrency")

    entries = _queries(phase3d_policy)
    if len(entries) != int(profile["spot_query_count"]):
        raise RuntimeError("registered resource-gate query count changed")

    baseline_url = os.environ.get(
        "PHASE3G_BASELINE_API_URL",
        "http://127.0.0.1:18082",
    )
    enabled_url = os.environ.get(
        "PHASE3G_RERANKER_API_URL",
        "http://127.0.0.1:18083",
    )
    baseline_container = os.environ.get(
        "PHASE3G_BASELINE_CONTAINER",
        "phase3g-baseline",
    )
    enabled_container = os.environ.get(
        "PHASE3G_RERANKER_CONTAINER",
        "phase3g-reranker",
    )
    limit = int(profile["request_limit"])
    timeout = httpx.Timeout(
        float(profile["request_timeout_seconds"]),
        connect=10.0,
    )

    sequential_added: list[float] = []
    sequential_baseline: list[float] = []
    sequential_enabled: list[float] = []
    concurrent_baseline: list[float] = []
    concurrent_enabled: list[float] = []
    expected_baseline: dict[str, list[str]] = {}
    expected_enabled: dict[str, list[str]] = {}

    async with (
        httpx.AsyncClient(base_url=baseline_url, timeout=timeout) as baseline_client,
        httpx.AsyncClient(base_url=enabled_url, timeout=timeout) as enabled_client,
    ):
        for _ in range(int(profile["warmup_rounds"])):
            for entry in entries:
                baseline_body, _ = await _timed_search(
                    baseline_client,
                    query=entry["query"],
                    limit=limit,
                    reranker_enabled=False,
                )
                enabled_body, _ = await _timed_search(
                    enabled_client,
                    query=entry["query"],
                    limit=limit,
                    reranker_enabled=True,
                )
                expected_baseline[entry["id"]] = _result_urls(baseline_body)
                expected_enabled[entry["id"]] = _result_urls(enabled_body)

        before_baseline = _process_metrics(baseline_container)
        before_enabled = _process_metrics(enabled_container)

        for round_index in range(int(profile["sequential_rounds"])):
            for entry in entries:
                if round_index % 2 == 0:
                    baseline_body, baseline_ms = await _timed_search(
                        baseline_client,
                        query=entry["query"],
                        limit=limit,
                        reranker_enabled=False,
                    )
                    enabled_body, enabled_ms = await _timed_search(
                        enabled_client,
                        query=entry["query"],
                        limit=limit,
                        reranker_enabled=True,
                    )
                else:
                    enabled_body, enabled_ms = await _timed_search(
                        enabled_client,
                        query=entry["query"],
                        limit=limit,
                        reranker_enabled=True,
                    )
                    baseline_body, baseline_ms = await _timed_search(
                        baseline_client,
                        query=entry["query"],
                        limit=limit,
                        reranker_enabled=False,
                    )
                if _result_urls(baseline_body) != expected_baseline[entry["id"]]:
                    raise RuntimeError(
                        f"baseline ranking changed for registered query {entry['id']}"
                    )
                if _result_urls(enabled_body) != expected_enabled[entry["id"]]:
                    raise RuntimeError(
                        f"reranked ranking changed for registered query {entry['id']}"
                    )
                sequential_baseline.append(baseline_ms)
                sequential_enabled.append(enabled_ms)
                sequential_added.append(max(0.0, enabled_ms - baseline_ms))

        concurrency = int(profile["concurrency"])
        requests = [
            entries[index % len(entries)]
            for index in range(concurrency)
        ]
        for round_index in range(int(profile["concurrent_rounds"])):
            if round_index % 2 == 0:
                concurrent_baseline.extend(
                    await _run_batch(
                        baseline_client,
                        requests,
                        limit=limit,
                        reranker_enabled=False,
                        expected_urls=expected_baseline,
                    )
                )
                concurrent_enabled.extend(
                    await _run_batch(
                        enabled_client,
                        requests,
                        limit=limit,
                        reranker_enabled=True,
                        expected_urls=expected_enabled,
                    )
                )
            else:
                concurrent_enabled.extend(
                    await _run_batch(
                        enabled_client,
                        requests,
                        limit=limit,
                        reranker_enabled=True,
                        expected_urls=expected_enabled,
                    )
                )
                concurrent_baseline.extend(
                    await _run_batch(
                        baseline_client,
                        requests,
                        limit=limit,
                        reranker_enabled=False,
                        expected_urls=expected_baseline,
                    )
                )

    after_baseline = _process_metrics(baseline_container)
    after_enabled = _process_metrics(enabled_container)

    added_p95_ms = _percentile(sequential_added, 0.95)
    rss_delta_mib = max(
        0.0,
        after_enabled["vm_hwm_mib"] - after_baseline["vm_hwm_mib"],
    )
    baseline_cpu_s = max(
        0.0,
        after_baseline["cpu_seconds"] - before_baseline["cpu_seconds"],
    )
    enabled_cpu_s = max(
        0.0,
        after_enabled["cpu_seconds"] - before_enabled["cpu_seconds"],
    )
    cpu_delta_s = max(0.0, enabled_cpu_s - baseline_cpu_s)

    checks = {
        "warm_added_p95": (
            added_p95_ms <= float(limits["p95_added_latency_ms_max"])
        ),
        "peak_rss_delta": (
            rss_delta_mib <= float(limits["peak_rss_delta_mib_max"])
        ),
        "concurrent_request_count": (
            len(concurrent_enabled)
            == int(profile["concurrency"]) * int(profile["concurrent_rounds"])
        ),
        "cpu_time_reported": all(
            math.isfinite(value) and value >= 0.0
            for value in (baseline_cpu_s, enabled_cpu_s, cpu_delta_s)
        ),
    }
    decision = "PASS_RESOURCE_GATE" if all(checks.values()) else "REJECT_RESOURCE_GATE"

    report = {
        "schema_version": 1,
        "phase": "3G",
        "gate": policy["gate"],
        "decision": decision,
        "base_certified_sha": policy["base_certified_sha"],
        "load_profile": profile,
        "limits": limits,
        "latency_ms": {
            "baseline_warm_p95": round(_percentile(sequential_baseline, 0.95), 3),
            "reranker_warm_p95": round(_percentile(sequential_enabled, 0.95), 3),
            "added_warm_p95": round(added_p95_ms, 3),
            "baseline_concurrent_p95": round(
                _percentile(concurrent_baseline, 0.95),
                3,
            ),
            "reranker_concurrent_p95": round(
                _percentile(concurrent_enabled, 0.95),
                3,
            ),
        },
        "memory_mib": {
            "baseline_vm_hwm": round(after_baseline["vm_hwm_mib"], 3),
            "reranker_vm_hwm": round(after_enabled["vm_hwm_mib"], 3),
            "peak_rss_delta": round(rss_delta_mib, 3),
            "measurement": "/proc/1/status VmHWM after identical live load profiles",
        },
        "cpu_seconds": {
            "baseline_workload": round(baseline_cpu_s, 6),
            "reranker_workload": round(enabled_cpu_s, 6),
            "added": round(cpu_delta_s, 6),
            "measurement": "/proc/1/stat user+system time after warmup",
        },
        "concurrency": {
            "requests": len(concurrent_enabled),
            "rounds": int(profile["concurrent_rounds"]),
            "width": int(profile["concurrency"]),
            "all_successful_and_deterministic": True,
        },
        "checks": checks,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    output_path = os.environ.get("PHASE3G_RESOURCE_OUTPUT")
    if output_path:
        await asyncio.to_thread(
            Path(output_path).write_text,
            rendered + "\n",
            encoding="utf-8",
        )
    print(f"Phase 3G live resource gate: {decision}")
    if decision != "PASS_RESOURCE_GATE":
        raise RuntimeError("Phase 3G live resource gate rejected the canary")


if __name__ == "__main__":
    asyncio.run(_run())
