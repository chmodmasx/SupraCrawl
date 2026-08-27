# Phase 3B — Dense/Hybrid retrieval experiment

Phase 3B starts only after the Phase 3A BM25 benchmark is certified. Its first gate is an
**evaluation-only candidate experiment**. It does not change `/v1/search`, Hermes routing,
production OpenSearch mappings, or the default ranking algorithm.

## Certified baseline

The comparison uses the exact versioned Phase 3A corpus, queries, graded qrels, metrics,
and production BM25 path. The certified reference is:

- MRR@10: `0.843750`
- Recall@5: `0.875000`
- nDCG@10: `0.840367`
- observed reference p95 latency: `59.465 ms`
- mean top-5 description tokens: `237.25`

The experiment must first prove that this BM25 baseline still meets its existing regression
floors. A BM25 regression makes the experiment invalid and fails the workflow.

## First dense candidate

The first candidate is `intfloat/multilingual-e5-small`, evaluated locally with FastEmbed
and ONNX Runtime:

- 384-dimensional normalized embeddings
- `query: ` prefix for queries
- `passage: ` prefix for corpus chunks
- passage text combines document title, section path, and chunk text
- no hosted model API and no external retrieval service

The corpus passage embeddings are produced once per benchmark run. Query embedding and the
existing BM25 OpenSearch request execute concurrently. Dense results are collapsed to unique
documents before fusion.

## Fusion

BM25 and dense document rankings are combined with deterministic Reciprocal Rank Fusion:

- `k = 60`
- top 10 documents
- duplicate document ids inside one input ranking contribute at most once
- deterministic tie-breaking by best individual rank, then document id

The fusion helper is independently unit tested and is not wired into production search in
this experiment.

## Pre-registered promotion gate

Promotion criteria are versioned in `evaluation/hybrid_candidate_policy.json` **before the
first candidate result is observed**. The candidate is eligible for a separate production
integration gate only if all of these conditions hold:

1. hybrid nDCG@10 improves over BM25 by at least `+0.02`;
2. hybrid MRR@10 does not regress;
3. hybrid Recall@5 does not regress;
4. the top-grade relevant document for both deliberate cross-language semantic queries
   `q13-cross-language-rrf` and `q14-cross-language-dense` appears in the hybrid top 5;
5. hybrid p95 wall-clock query latency is at most `500 ms` in the GitHub Actions CPU runner.

The policy must not be loosened after seeing a candidate result merely to make that candidate
pass. A later candidate may be evaluated against the same policy, with its model identity
versioned explicitly.

## First measured result

The first valid candidate run used head `e63d3efbc39ecd958812d8f9569bece657c047f1`
and GitHub Actions run `33102415862`. The complete summary is versioned in
`evaluation/hybrid_candidate_result.json`.

| Retrieval | MRR@10 | Recall@5 | nDCG@10 |
| --- | ---: | ---: | ---: |
| BM25 control | 0.843750 | 0.875000 | 0.840367 |
| Dense E5-small | 0.968750 | 1.000000 | 0.966649 |
| BM25 + dense RRF | 0.927083 | 1.000000 | 0.935399 |

The hybrid nDCG delta over BM25 was `+0.095031`. Hybrid p95 wall-clock query latency was
`71.2 ms`; query-embedding p95 was `16.883 ms`. Both deliberate cross-language target
documents appeared in the hybrid top 5. Every pre-registered promotion check passed.

Dense-only retrieval scored higher than equal-weight RRF on all three aggregate relevance
metrics. Therefore the result does **not** establish RRF as the best production architecture.
It establishes that the multilingual semantic signal is strong enough to justify a separate
production-integration comparison of BM25, dense, and hybrid retrieval.

## Measurement limits

This screening benchmark intentionally isolates retrieval quality, but it is not a production
scale test:

- the corpus contains 16 documents and 16 passage vectors in this fixture;
- dense ranking uses normalized in-memory dot products, not an OpenSearch `knn_vector` query;
- model load (`5638.059 ms`) is startup cost and is excluded from per-query latency;
- corpus passage embedding (`320.473 ms` for this fixture) is indexing/offline work and is
  excluded from per-query latency;
- the benchmark is deliberately small and curated, so it cannot by itself certify exact-match,
  identifier-heavy, long-tail, or large-index retrieval behavior.

Production integration must therefore validate the real vector-store path and preserve a
lexical control. The Phase 3B benchmark and policy remain immutable evidence of this screening
result; additional robustness evaluation must be added as a separate gate rather than editing
this result after the fact.

## Workflow semantics

`Phase 3B Evaluation` distinguishes two outcomes:

- **experiment execution PASS**: dependencies/model load, BM25 control, dense retrieval,
  metrics, RRF evaluation, and cleanup all completed correctly;
- **promotion candidate PASS/REJECT**: the measured hybrid result did or did not satisfy the
  pre-registered promotion policy.

A valid experiment with `REJECT` is intentionally a green workflow. Rejection means the
candidate is not promoted; it is not an infrastructure or test failure.

## Gate before production integration

Do not change production retrieval until all of the following are true:

1. the Phase 3B experiment workflow is green on the exact candidate head SHA;
2. CI plus Phase 1, Phase 2, and Phase 3 regression workflows remain green;
3. the complete experiment diff is audited;
4. the measured candidate reports `promotion.passed = true` under the pre-registered policy;
5. no known blocker/high-severity defect remains.

If the candidate is rejected, production remains BM25. If it passes, this authorizes only a
separate production-integration gate. That gate must compare dense and hybrid retrieval on the
real vector-store path because the screening result does not justify choosing one of them by
assumption.
