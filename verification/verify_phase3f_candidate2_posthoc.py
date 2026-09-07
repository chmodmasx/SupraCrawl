from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3f_policy.json"
CANDIDATE2_POLICY_PATH = ROOT / "evaluation" / "phase3f_candidate2_policy.json"
CANDIDATE1_RESULT_PATH = ROOT / "evaluation" / "phase3f_candidate1_result.json"
EVIDENCE_PATH = ROOT / "evaluation" / "phase3f_candidate1_rankings.jsonl"
QUERY_PATHS = (
    ROOT / "evaluation" / "queries.jsonl",
    ROOT / "evaluation" / "phase3c_exact_queries.jsonl",
)


def _load_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return loaded


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(loaded)
    return rows


def _metric_dict(metric: RetrievalMetrics) -> dict[str, float]:
    return {
        "mrr_at_10": round(metric.mrr_at_10, 6),
        "recall_at_5": round(metric.recall_at_5, 6),
        "ndcg_at_10": round(metric.ndcg_at_10, 6),
    }


def _candidate2_ranking(
    hybrid_top10: list[str],
    candidate1_score_order: list[str],
    protected_top_k: int,
) -> list[str]:
    if len(hybrid_top10) != 10 or len(candidate1_score_order) != 10:
        raise RuntimeError("Phase 3F ranking evidence must contain exactly 10 documents")
    if len(set(hybrid_top10)) != 10 or len(set(candidate1_score_order)) != 10:
        raise RuntimeError("Phase 3F ranking evidence contains duplicate document ids")
    if set(hybrid_top10) != set(candidate1_score_order):
        raise RuntimeError("Candidate 1 score order changed the certified hybrid candidate set")
    if protected_top_k <= 0 or protected_top_k >= len(hybrid_top10):
        raise RuntimeError("protected_top_k must split the candidate pool")

    protected = set(hybrid_top10[:protected_top_k])
    top = [document_id for document_id in candidate1_score_order if document_id in protected]
    tail = [document_id for document_id in candidate1_score_order if document_id not in protected]
    ranking = top + tail

    if set(ranking[:protected_top_k]) != set(hybrid_top10[:protected_top_k]):
        raise RuntimeError("Candidate 2 violated frozen top-k membership")
    if set(ranking[protected_top_k:]) != set(hybrid_top10[protected_top_k:]):
        raise RuntimeError("Candidate 2 violated frozen tail membership")
    return ranking


def _main() -> None:
    phase_policy = _load_object(POLICY_PATH)
    candidate2_policy = _load_object(CANDIDATE2_POLICY_PATH)
    candidate1_result = _load_object(CANDIDATE1_RESULT_PATH)
    evidence_rows = _load_jsonl(EVIDENCE_PATH)
    if len(evidence_rows) != 31:
        raise RuntimeError("ranking evidence must contain one header plus 30 query rows")

    header = evidence_rows[0]
    query_rows = evidence_rows[1:]
    source = candidate1_result["source"]
    if header["source"]["artifact_id"] != source["artifact_id"]:
        raise RuntimeError("ranking evidence artifact id does not match Candidate 1 result")
    if header["source"]["artifact_sha256"] != source["artifact_sha256"]:
        raise RuntimeError("ranking evidence artifact digest does not match Candidate 1 result")
    if header["source"]["evaluated_head_sha"] != source["evaluated_head_sha"]:
        raise RuntimeError("ranking evidence evaluated SHA does not match Candidate 1 result")
    if header["candidate_id"] != candidate1_result["candidate"]["id"]:
        raise RuntimeError("ranking evidence candidate id does not match Candidate 1 result")

    if candidate2_policy["base_main_sha"] != phase_policy["base_main_sha"]:
        raise RuntimeError("Candidate 2 and Phase 3F policies disagree on base main SHA")
    if candidate2_policy["candidate_pool_size"] != phase_policy["reranker_scope"][
        "candidate_pool_size"
    ]:
        raise RuntimeError("Candidate 2 changed the frozen top-N candidate pool")
    if candidate2_policy["formal_evaluation"]["state"] != "BLOCKED_PENDING_FROZEN_HOLDOUT":
        raise RuntimeError("formal Candidate 2 execution must remain blocked in this commit")

    relevance_by_id: dict[str, dict[str, int]] = {}
    for path in QUERY_PATHS:
        for query in _load_jsonl(path):
            query_id = str(query["id"])
            if query_id in relevance_by_id:
                raise RuntimeError(f"duplicate benchmark query id: {query_id}")
            relevance_by_id[query_id] = {
                str(document_id): int(grade)
                for document_id, grade in query["relevance"].items()
            }

    if set(relevance_by_id) != {str(row["id"]) for row in query_rows}:
        raise RuntimeError("ranking evidence query ids do not match the frozen 30-query benchmark")

    protected_top_k = int(candidate2_policy["protected_top_k"])
    hybrid_metrics: list[RetrievalMetrics] = []
    candidate1_metrics: list[RetrievalMetrics] = []
    candidate2_metrics: list[RetrievalMetrics] = []
    query_details: list[dict[str, Any]] = []

    for row in query_rows:
        query_id = str(row["id"])
        hybrid = [str(value) for value in row["hybrid_top10"]]
        candidate1 = [str(value) for value in row["candidate1_score_order"]]
        candidate2 = _candidate2_ranking(hybrid, candidate1, protected_top_k)
        relevance = relevance_by_id[query_id]

        hybrid_metric = evaluate_ranking(hybrid, relevance)
        candidate1_metric = evaluate_ranking(candidate1, relevance)
        candidate2_metric = evaluate_ranking(candidate2, relevance)
        if candidate2_metric.recall_at_5 != hybrid_metric.recall_at_5:
            raise RuntimeError(f"Candidate 2 Recall@5 invariant failed for {query_id}")

        hybrid_metrics.append(hybrid_metric)
        candidate1_metrics.append(candidate1_metric)
        candidate2_metrics.append(candidate2_metric)
        query_details.append(
            {
                "id": query_id,
                "hybrid_top10": hybrid,
                "candidate2_top10": candidate2,
                "hybrid_ndcg_at_10": round(hybrid_metric.ndcg_at_10, 6),
                "candidate2_ndcg_at_10": round(candidate2_metric.ndcg_at_10, 6),
            }
        )

    hybrid = _metric_dict(macro_average(hybrid_metrics))
    candidate1 = _metric_dict(macro_average(candidate1_metrics))
    candidate2 = _metric_dict(macro_average(candidate2_metrics))

    if hybrid != candidate1_result["benchmark"]["frozen_hybrid"]:
        raise RuntimeError("ranking evidence does not reproduce the frozen hybrid baseline")
    if candidate1 != candidate1_result["benchmark"]["reranked"]:
        raise RuntimeError("ranking evidence does not reproduce Candidate 1")

    promotion = phase_policy["promotion"]
    runtime = candidate1_result["runtime"]
    ndcg_delta = round(candidate2["ndcg_at_10"] - hybrid["ndcg_at_10"], 6)
    mrr_regression = round(hybrid["mrr_at_10"] - candidate2["mrr_at_10"], 6)
    recall_regression = round(hybrid["recall_at_5"] - candidate2["recall_at_5"], 6)
    checks = {
        "top5_membership_preserved": True,
        "ndcg_material_improvement": (
            ndcg_delta >= float(promotion["ndcg_at_10_min_delta_vs_frozen_hybrid"])
        ),
        "mrr_no_regression": (
            mrr_regression <= float(promotion["max_mrr_at_10_regression_vs_frozen_hybrid"])
        ),
        "recall_no_regression": (
            recall_regression <= float(promotion["max_recall_at_5_regression_vs_frozen_hybrid"])
        ),
        "inherited_candidate1_p95_added_latency": (
            float(runtime["reranker_p95_ms"]) <= float(promotion["p95_added_latency_ms_max"])
        ),
        "inherited_candidate1_peak_rss_delta": (
            float(runtime["peak_rss_delta_mib"]) <= float(promotion["peak_rss_delta_mib_max"])
        ),
    }
    decision = "PASS_POSTHOC" if all(checks.values()) else "REJECT_POSTHOC"

    expected = candidate2_policy["known_benchmark"]
    if decision != expected["expected_decision"]:
        raise RuntimeError(f"Candidate 2 post-hoc decision changed unexpectedly: {decision}")
    expected_metrics = expected["expected_metrics"]
    for key in ("mrr_at_10", "recall_at_5", "ndcg_at_10"):
        if candidate2[key] != float(expected_metrics[key]):
            raise RuntimeError(f"Candidate 2 expected {key} changed: {candidate2[key]}")
    if ndcg_delta != float(expected_metrics["ndcg_at_10_delta_vs_frozen_hybrid"]):
        raise RuntimeError("Candidate 2 expected nDCG delta changed")

    report = {
        "schema_version": 1,
        "phase": "3F",
        "experiment": candidate2_policy["experiment"],
        "decision": decision,
        "promotion_eligible": False,
        "promotion_blocker": "independent frozen holdout has not been authored or executed",
        "aggregate": {
            "frozen_hybrid": hybrid,
            "candidate1_full_rerank": candidate1,
            "candidate2_top5_preserving": candidate2,
            "ndcg_at_10_delta_vs_frozen_hybrid": ndcg_delta,
            "mrr_regression_vs_frozen_hybrid": mrr_regression,
            "recall_at_5_regression_vs_frozen_hybrid": recall_regression,
        },
        "runtime_evidence": {
            "source": (
                "inherited from Candidate 1 because model, model file, runtime, "
                "and top-10 scoring are unchanged"
            ),
            "reranker_p95_ms": runtime["reranker_p95_ms"],
            "peak_rss_delta_mib": runtime["peak_rss_delta_mib"],
            "candidate2_partition_transform_overhead_measured": False,
        },
        "checks": checks,
        "queries_detail": query_details,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    output_path = os.environ.get("PHASE3F_CANDIDATE2_POSTHOC_OUTPUT")
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
    print("Phase 3F Candidate 2 known-benchmark regression: PASS_POSTHOC")


if __name__ == "__main__":
    _main()
