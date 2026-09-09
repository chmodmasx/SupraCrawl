from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

POLICY_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "phase3j_policy.json"
OUTPUT_PATH = Path(os.environ.get("PHASE3J_READINESS_OUTPUT", "phase3j-readiness-report.json"))


def _base_url(name: str) -> str:
    value = os.environ.get(name, "").rstrip("/")
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def _get(client: httpx.Client, base_url: str, path: str) -> httpx.Response:
    return client.get(base_url + path)


def _component_reason(payload: dict[str, Any], component: str) -> str | None:
    components = payload.get("components")
    if not isinstance(components, dict):
        raise AssertionError("readiness payload has no components object")
    value = components.get(component)
    if not isinstance(value, dict):
        raise AssertionError(f"readiness payload has no {component} component")
    reason = value.get("reason")
    return reason if isinstance(reason, str) else None


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    acceptance = policy["acceptance"]
    healthy = {
        "baseline": _base_url("PHASE3J_BASELINE_API_URL"),
        "reranker": _base_url("PHASE3J_RERANKER_API_URL"),
        "backpressure": _base_url("PHASE3J_BACKPRESSURE_API_URL"),
    }
    reranker_fault = _base_url("PHASE3J_RERANKER_FAULT_API_URL")
    opensearch_fault = _base_url("PHASE3J_OPENSEARCH_FAULT_API_URL")

    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "3J",
        "gate": policy["gate"],
        "base_certified_sha": policy["base_certified_sha"],
        "healthy": {},
    }

    with httpx.Client(timeout=30.0) as client:
        for name, base_url in healthy.items():
            health = _get(client, base_url, "/v1/health")
            ready = _get(client, base_url, "/v1/ready")
            assert health.status_code == acceptance["healthy_health_status"]
            assert ready.status_code == acceptance["healthy_ready_status"]
            ready_payload = ready.json()
            assert ready_payload["status"] == acceptance["healthy_ready_body_status"]
            report["healthy"][name] = {
                "health_status": health.status_code,
                "ready_status": ready.status_code,
            }

        health = _get(client, reranker_fault, "/v1/health")
        ready = _get(client, reranker_fault, "/v1/ready")
        ready_payload = ready.json()
        reranker_reason = _component_reason(ready_payload, "reranker")
        assert health.status_code == acceptance[
            "reranker_without_startup_warmup_health_status"
        ]
        assert ready.status_code == acceptance[
            "reranker_without_startup_warmup_ready_status"
        ]
        assert reranker_reason == acceptance["reranker_without_startup_warmup_reason"]
        report["reranker_without_startup_warmup"] = {
            "health_status": health.status_code,
            "ready_status": ready.status_code,
            "reason": reranker_reason,
        }

        health = _get(client, opensearch_fault, "/v1/health")
        ready = _get(client, opensearch_fault, "/v1/ready")
        ready_payload = ready.json()
        opensearch_reason = _component_reason(ready_payload, "lexical")
        assert health.status_code == acceptance["opensearch_fault_health_status"]
        assert ready.status_code == acceptance["opensearch_fault_ready_status"]
        assert opensearch_reason == acceptance["opensearch_fault_reason"]
        report["opensearch_fault"] = {
            "health_status": health.status_code,
            "ready_status": ready.status_code,
            "reason": opensearch_reason,
        }

    report["decision"] = "PASS_READINESS_GATE"
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Phase 3J readiness verification: PASS")


if __name__ == "__main__":
    main()
