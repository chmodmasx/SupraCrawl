from __future__ import annotations

import json
from pathlib import Path

import pytest

from verification.verify_phase4c_refresh_efficiency import _run

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation/phase4c_refresh_efficiency_policy.json"


def test_phase4c_policy_is_measurement_only_and_bound_to_certified_main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    assert policy["base_certified_sha"] == "542820dfb5fefe05988ff57f60a81c8d0f395698"
    assert policy["constraints"] == {
        "production_code_unchanged": True,
        "ranking_behavior_unchanged": True,
        "reranker_behavior_unchanged": True,
        "ssrf_and_robots_behavior_unchanged": True,
        "phase4a_and_phase4b_request_defaults_unchanged": True,
        "no_latency_threshold": True,
        "no_scheduler": True,
        "no_external_network_dependency": True,
    }
    assert policy["fixture"]["html_body_bytes"] == 262_144


@pytest.mark.asyncio
async def test_phase4c_refresh_efficiency_gate() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    report = await _run(policy)

    assert report["decision"] == "PASS_REFRESH_EFFICIENCY_GATE"
    assert report["fixture_body_bytes"] == 262_144
    assert report["reductions"] == {
        "fresh_leaf_target_get_reduction_percent": 100.0,
        "fresh_leaf_body_byte_reduction_percent": 100.0,
        "stale_304_body_byte_reduction_percent": 100.0,
        "stale_304_extraction_reduction_percent": 100.0,
        "stale_304_index_reduction_percent": 100.0,
    }

    measurements = report["measurements"]
    assert measurements["phase4a_fresh"]["target_gets"] == 1
    assert measurements["phase4b_fresh_leaf"]["target_gets"] == 0
    assert measurements["phase4b_stale_304"]["conditional_gets"] == 1
    assert measurements["phase4b_stale_304"]["response_body_bytes"] == 0

    fallback = measurements["phase4b_304_touch_failure"]
    assert fallback["conditional_gets"] == 1
    assert fallback["unconditional_gets"] == 1
    assert fallback["response_body_bytes"] == 262_144
    assert fallback["extractions"] == 1
    assert fallback["indexes"] == 1
