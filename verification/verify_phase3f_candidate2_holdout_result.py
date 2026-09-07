from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "evaluation" / "phase3f_candidate2_holdout_result.json"
POLICY_PATH = ROOT / "evaluation" / "phase3f_policy.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _main() -> None:
    result = _load(RESULT_PATH)
    policy = _load(POLICY_PATH)
    source = result["source"]

    expected_source = {
        "evaluated_head_sha": "ee0ac82a917b5138ed1e7e8850faae3afc78eb15",
        "workflow_run_id": 34146251340,
        "workflow_job_id": 101818812164,
        "artifact_id": 10027849168,
        "artifact_name": "phase3f-candidate2-independent-holdout",
        "artifact_digest": (
            "sha256:4cf9ed5b6604f1e4b75ed5a9c92aaea41717baf83a12e93ccc5afe42152c5806"
        ),
        "artifact_payload_sha256": (
            "42606895387b50cffc07ad54ea58e931930a0135381bb7363d55b53a3fa44f98"
        ),
        "workflow_matrix": "9/9 success",
    }
    if source != expected_source:
        raise RuntimeError("frozen holdout source provenance changed")
    if result["decision"] != "PASS_HOLDOUT" or result["promotion_eligible"] is not True:
        raise RuntimeError("frozen Candidate 2 holdout decision is not PASS_HOLDOUT")

    benchmark = result["benchmark"]
    delta = benchmark["delta"]
    runtime = result["runtime"]
    promotion = policy["promotion"]
    baseline = policy["baseline"]

    checks = {
        "candidate_recall_at_10": (
            float(delta["candidate_recall_at_10_mean"])
            >= float(baseline["minimum_candidate_recall_at_10"])
        ),
        "ndcg_material_improvement": (
            float(delta["ndcg_at_10_delta_vs_holdout_hybrid"])
            >= float(promotion["ndcg_at_10_min_delta_vs_frozen_hybrid"])
        ),
        "mrr_no_regression": (
            float(delta["mrr_regression_vs_holdout_hybrid"])
            <= float(promotion["max_mrr_at_10_regression_vs_frozen_hybrid"])
        ),
        "recall_no_regression": (
            float(delta["recall_at_5_regression_vs_holdout_hybrid"])
            <= float(promotion["max_recall_at_5_regression_vs_frozen_hybrid"])
        ),
        "p95_added_latency": (
            float(runtime["added_p95_ms"])
            <= float(promotion["p95_added_latency_ms_max"])
        ),
        "peak_rss_delta": (
            float(runtime["peak_rss_delta_mib"])
            <= float(promotion["peak_rss_delta_mib_max"])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen holdout no longer satisfies Phase 3F policy: {checks}")
    if result["checks"] != {
        "candidate_recall_at_10": True,
        "top5_membership_preserved": True,
        "ndcg_material_improvement": True,
        "mrr_no_regression": True,
        "recall_no_regression": True,
        "p95_added_latency": True,
        "peak_rss_delta": True,
    }:
        raise RuntimeError("frozen holdout check vector changed")

    per_query = result["diagnostics"]["per_query_ndcg"]
    if int(per_query["wins"]) + int(per_query["ties"]) + int(per_query["losses"]) != 60:
        raise RuntimeError("frozen per-query diagnostic counts do not total 60")
    lexical = result["diagnostics"]["family_ndcg"]["lexical exact identifiers"]
    if float(lexical["delta"]) >= 0:
        raise RuntimeError("expected lexical-identifier caveat is missing from frozen result")

    print("Phase 3F Candidate 2 frozen holdout result: PASS_HOLDOUT")


if __name__ == "__main__":
    _main()
