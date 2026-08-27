# Phase 3 acceptance gate

Phase 3 begins with **Phase 3A: Retrieval Evaluation**. No dense embeddings, hybrid fusion, or reranker may be added until the current BM25 implementation has a reproducible measured baseline and every Phase 3A gate below passes.

## Phase 3A functional scope

Phase 3A adds measurement infrastructure, not a new ranking algorithm:

- a versioned deterministic retrieval corpus;
- versioned query relevance judgments (qrels);
- English and Spanish queries plus intentionally cross-language semantic cases that expose lexical-search headroom;
- multi-relevant queries with graded 3/2/1 judgments so nDCG exercises relevance ordering rather than behaving as a binary rank metric;
- standard retrieval metrics: MRR@10, Recall@5, and graded nDCG@10;
- p50/p95 query latency measurement;
- approximate top-5 description token measurement;
- a live benchmark against the real OpenSearch/BM25 path used by `OpenSearchStore.search`;
- explicit benchmark thresholds that fail CI on relevance, latency, or context-size regressions.

The benchmark intentionally isolates retrieval from crawling and extraction. Phase 1 and Phase 2 workflows remain responsible for proving those layers still work end to end.

## Benchmark fixture requirements

The fixture is valid only when:

- corpus document IDs and URLs are unique;
- every document has a title and at least one non-empty chunk;
- query IDs are unique;
- every query has at least one positive relevance judgment;
- every relevance judgment references a document present in the corpus;
- relevance grades are positive integers;
- at least one query has multiple relevant documents with more than one positive grade;
- the benchmark contains at least the minimum query count pinned in `evaluation/bm25_thresholds.json`;
- no live third-party website is required to score retrieval quality.

The initial threshold file may contain bootstrap floors only while the first reference run is being established. **Before Phase 3A certification, those floors must be tightened to the measured BM25 reference baseline with a small documented regression tolerance.** They must not later be loosened merely to make a failing ranking change pass.

## Deterministic acceptance

The Python suite must prove at minimum:

- reciprocal rank uses the first relevant result;
- Recall@k counts unique relevant documents correctly;
- DCG/nDCG use graded relevance and the correct logarithmic discount;
- empty/non-positive qrels cannot create false relevance;
- the Phase 3 cutoffs are MRR@10, Recall@5, and nDCG@10;
- macro aggregation is query-level and rejects an empty benchmark;
- all Phase 1 and Phase 2 unit/regression tests remain green.

Ruff must pass with no ignored new violations.

## Live BM25 baseline acceptance

A dedicated Phase 3 workflow must:

1. Start the repository's pinned OpenSearch service from Docker Compose.
2. Wait for OpenSearch health before benchmarking.
3. Create isolated benchmark document/chunk indexes using the same mappings and indexing code as SupraCrawl.
4. Index the complete deterministic corpus through `OpenSearchStore.index_document`.
5. Execute every benchmark query through `OpenSearchStore.search` with the production BM25 query.
6. Print per-query rankings and metrics so regressions are diagnosable from CI logs.
7. Print aggregate MRR@10, Recall@5, nDCG@10, p50/p95 latency, and mean top-5 description tokens.
8. Fail when any pinned threshold is violated.
9. Remove benchmark indexes even on failure.
10. Exit with `Phase 3A BM25 baseline verification: PASS` only when every check succeeds.
11. Tear down OpenSearch, its network, and benchmark volume cleanly.

## Phase 3A release rule

Phase 3A is not certified until, on the exact commit intended for merge and again after merge on `main`:

- normal CI is fully green;
- Phase 1 Verification remains fully green;
- Phase 2 Verification remains fully green;
- Phase 3 Verification is fully green;
- benchmark thresholds have been tightened from bootstrap values to the measured BM25 reference baseline;
- no known blocker/high-severity correctness or benchmark-integrity defect remains open.

Only after Phase 3A is certified may Phase 3B introduce dense retrieval. The first Phase 3B implementation must compare **BM25 vs dense vs BM25+dense/RRF** on this unchanged benchmark. Hybrid retrieval is retained only if it demonstrates a material relevance improvement without violating the latency/context guardrails. Reranking remains a later, separately measured decision.
