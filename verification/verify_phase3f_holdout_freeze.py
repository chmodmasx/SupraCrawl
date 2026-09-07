from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from verify_retrieval_baseline import _load_jsonl, _validate_fixture

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3f_policy.json"
CANDIDATE2_POLICY_PATH = ROOT / "evaluation" / "phase3f_candidate2_policy.json"
PROTOCOL_PATH = ROOT / "evaluation" / "phase3f_holdout_protocol.json"
MANIFEST_PATH = ROOT / "evaluation" / "phase3f_holdout_manifest.json"
CORPUS_PATH = ROOT / "evaluation" / "phase3f_holdout_corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "phase3f_holdout_queries.jsonl"
EXECUTOR_PATH = ROOT / "verification" / "verify_phase3f_candidate2_holdout.py"
KNOWN_CORPUS_PATHS = (
    ROOT / "evaluation" / "corpus.jsonl",
    ROOT / "evaluation" / "phase3c_exact_corpus.jsonl",
)
KNOWN_QUERY_PATHS = (
    ROOT / "evaluation" / "queries.jsonl",
    ROOT / "evaluation" / "phase3c_exact_queries.jsonl",
)


def _load_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return loaded


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document_content_hash(document: dict[str, Any]) -> str:
    chunks = document.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError(f"document {document.get('id')} has no chunks")
    texts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise RuntimeError(f"document {document.get('id')} has an invalid chunk")
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"document {document.get('id')} has an empty chunk")
        texts.append(text)
    markdown = "\n\n".join(texts)
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def verify_holdout_freeze() -> dict[str, Any]:
    policy = _load_object(POLICY_PATH)
    candidate2 = _load_object(CANDIDATE2_POLICY_PATH)
    protocol = _load_object(PROTOCOL_PATH)
    manifest = _load_object(MANIFEST_PATH)
    corpus = _load_jsonl(CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH)

    if protocol["status"] != "HOLDOUT_FROZEN_PENDING_EXECUTION":
        raise RuntimeError("holdout protocol is not in the frozen pending-execution state")
    if candidate2["formal_evaluation"]["state"] != "FROZEN_HOLDOUT_PENDING_EXECUTION":
        raise RuntimeError("Candidate 2 policy is not frozen for holdout execution")
    if candidate2["formal_evaluation"]["execution_enabled_in_freeze_commit"] is not False:
        raise RuntimeError("holdout execution must remain disabled in the freeze commit")
    if manifest["candidate_scoring_observed"] is not False:
        raise RuntimeError("freeze manifest says candidate scoring was already observed")
    if manifest["execution_enabled"] is not False:
        raise RuntimeError("freeze manifest unexpectedly enables holdout execution")

    base_main_sha = str(policy["base_main_sha"])
    if candidate2["base_main_sha"] != base_main_sha:
        raise RuntimeError("Candidate 2 base main SHA differs from Phase 3F policy")
    if manifest["base_main_sha"] != base_main_sha:
        raise RuntimeError("holdout manifest base main SHA differs from Phase 3F policy")
    if protocol["freeze_artifacts"]["manifest_must_reference_base_main_sha"] != base_main_sha:
        raise RuntimeError("holdout protocol base main SHA differs from Phase 3F policy")

    expected_files = manifest["files"]
    actual_paths = {
        "corpus": CORPUS_PATH,
        "queries": QUERIES_PATH,
        "candidate2_policy": CANDIDATE2_POLICY_PATH,
        "holdout_protocol": PROTOCOL_PATH,
        "holdout_executor": EXECUTOR_PATH,
    }
    if set(expected_files) != set(actual_paths):
        raise RuntimeError("holdout manifest file set changed")
    for key, path in actual_paths.items():
        expected_sha = str(expected_files[key]["sha256"])
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"frozen {key} checksum changed: expected {expected_sha}, got {actual_sha}"
            )
    if _sha256(Path(__file__)) != str(manifest["freeze_verifier_sha256"]):
        raise RuntimeError("holdout freeze verifier changed after the manifest was written")

    if manifest["promotion_thresholds"] != policy["promotion"]:
        raise RuntimeError("Phase 3F promotion thresholds changed after holdout freeze")
    if float(manifest["minimum_candidate_recall_at_10"]) != float(
        policy["baseline"]["minimum_candidate_recall_at_10"]
    ):
        raise RuntimeError("candidate Recall@10 threshold changed after holdout freeze")
    if manifest["candidate_model"] != candidate2["model"]:
        raise RuntimeError("Candidate 2 model identity differs from frozen holdout manifest")
    if int(manifest["candidate_pool_size"]) != int(candidate2["candidate_pool_size"]):
        raise RuntimeError("Candidate 2 candidate pool size differs from manifest")
    if int(manifest["protected_top_k"]) != int(candidate2["protected_top_k"]):
        raise RuntimeError("Candidate 2 protected top-k differs from manifest")
    if manifest["ranking_strategy"] != candidate2["ranking_strategy"]:
        raise RuntimeError("Candidate 2 ranking strategy differs from manifest")

    minimum_queries = int(protocol["minimum_queries"])
    _validate_fixture(corpus, queries, minimum_queries=minimum_queries)
    if len(corpus) != int(manifest["documents"]):
        raise RuntimeError("holdout document count differs from manifest")
    if len(queries) != int(manifest["queries"]):
        raise RuntimeError("holdout query count differs from manifest")

    known_corpus: list[dict[str, Any]] = []
    for path in KNOWN_CORPUS_PATHS:
        known_corpus.extend(_load_jsonl(path))
    known_queries: list[dict[str, Any]] = []
    for path in KNOWN_QUERY_PATHS:
        known_queries.extend(_load_jsonl(path))

    holdout_ids = {str(document["id"]) for document in corpus}
    known_ids = {str(document["id"]) for document in known_corpus}
    overlap_ids = holdout_ids & known_ids
    if overlap_ids:
        raise RuntimeError(f"holdout reused known document ids: {sorted(overlap_ids)}")

    holdout_query_ids = {str(query["id"]) for query in queries}
    known_query_ids = {str(query["id"]) for query in known_queries}
    overlap_query_ids = holdout_query_ids & known_query_ids
    if overlap_query_ids:
        raise RuntimeError(
            f"holdout reused known query ids: {sorted(overlap_query_ids)}"
        )

    holdout_hashes = [_document_content_hash(document) for document in corpus]
    if len(set(holdout_hashes)) != len(holdout_hashes):
        raise RuntimeError("holdout contains duplicate document content hashes")
    known_hashes = {_document_content_hash(document) for document in known_corpus}
    content_overlap = set(holdout_hashes) & known_hashes
    if content_overlap:
        raise RuntimeError("holdout reused document content from the known benchmark")

    for document in corpus:
        language = document.get("language")
        if language not in {"en", "es"}:
            raise RuntimeError(f"document {document['id']} has invalid language {language!r}")
        source_refs = document.get("source_refs")
        if not isinstance(source_refs, list) or not source_refs:
            raise RuntimeError(f"document {document['id']} has no source_refs")
        if not all(
            isinstance(source, str) and source.startswith("src/supracrawl/")
            for source in source_refs
        ):
            raise RuntimeError(f"document {document['id']} has invalid source_refs")

    required_languages = set(protocol["requirements"]["languages"])
    language_counts = Counter(str(query.get("language")) for query in queries)
    if set(language_counts) != required_languages:
        raise RuntimeError(
            f"holdout language coverage changed: {dict(language_counts)}"
        )
    if dict(language_counts) != manifest["language_counts"]:
        raise RuntimeError("holdout language counts differ from manifest")

    required_families = set(protocol["requirements"]["query_families"])
    family_counts = Counter(str(query.get("family")) for query in queries)
    if set(family_counts) != required_families:
        raise RuntimeError(
            f"holdout query-family coverage changed: {dict(family_counts)}"
        )
    if dict(family_counts) != manifest["family_counts"]:
        raise RuntimeError("holdout query-family counts differ from manifest")

    multi_relevance = 0
    document_by_id = {str(document["id"]): document for document in corpus}
    for query in queries:
        relevance = query["relevance"]
        if len(relevance) > 1 and len(set(relevance.values())) > 1:
            multi_relevance += 1

        if query.get("language") == "cross-language":
            query_language = query.get("query_language")
            target_language = query.get("target_language")
            if query_language not in {"en", "es"} or target_language not in {"en", "es"}:
                raise RuntimeError(
                    f"cross-language query {query['id']} lacks language direction"
                )
            if query_language == target_language:
                raise RuntimeError(
                    f"cross-language query {query['id']} has identical source and target"
                )
            top_grade = max(int(grade) for grade in relevance.values())
            top_documents = [
                document_by_id[str(document_id)]
                for document_id, grade in relevance.items()
                if int(grade) == top_grade
            ]
            if not all(
                document.get("language") == target_language
                for document in top_documents
            ):
                raise RuntimeError(
                    f"cross-language query {query['id']} top relevance target is inconsistent"
                )

    minimum_multi = int(protocol["requirements"]["minimum_multi_relevance_queries"])
    if multi_relevance < minimum_multi:
        raise RuntimeError(
            f"holdout has {multi_relevance} graded multi-relevance queries; "
            f"expected at least {minimum_multi}"
        )
    if multi_relevance != int(manifest["graded_multi_relevance_queries"]):
        raise RuntimeError("graded multi-relevance query count differs from manifest")

    report = {
        "documents": len(corpus),
        "queries": len(queries),
        "graded_multi_relevance_queries": multi_relevance,
        "language_counts": dict(language_counts),
        "family_counts": dict(family_counts),
        "content_hash_overlap_with_known_benchmark": 0,
        "document_id_overlap_with_known_benchmark": 0,
        "query_id_overlap_with_known_benchmark": 0,
        "candidate_scoring_observed": False,
        "execution_enabled": False,
    }
    return report


def _main() -> None:
    report = verify_holdout_freeze()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("Phase 3F independent holdout freeze: PASS")


if __name__ == "__main__":
    _main()
