# Phase 3F — Reranker evaluation gate

Phase 3F begins only after Phase 3E is certified on the exact merged `main` SHA. The certified
starting point is `aa09dc9cfa2bac517219b28fcb0c3e49fa5591dd`.

This phase is an experiment first. It does not authorize a production reranker merely because
a model can be loaded or because a few queries look better.

## Frozen first-stage baseline

Before evaluating any reranker, the Phase 3F workflow must reproduce the current production
retrieval stack and freeze its behavior on the versioned benchmark:

- explicit BM25 remains the lexical control;
- explicit hybrid uses the certified E5 + OpenSearch exact-vector + RRF path;
- omitted search mode must match the certified hybrid default;
- the full Phase 3A corpus/queries and Phase 3C exact-match fixtures are evaluated together;
- ranking quality uses MRR@10, Recall@5, and nDCG@10;
- hybrid candidate-pool Recall@10 is recorded because reranking cannot recover a relevant
  document that never reaches the reranker;
- warm API latency is measured after model warm-up and excludes first model download.

The preregistered policy is `evaluation/phase3f_policy.json`. It must not be weakened after a
candidate result is observed.

## Reranker scope

A Phase 3F candidate may only reorder the top 10 documents produced by the certified hybrid
first stage. During screening it must not alter:

- BM25 scoring or mappings;
- E5 embeddings;
- dense/vector retrieval;
- RRF candidate generation;
- indexing semantics;
- the global production search path.

Hosted inference is out of scope. The candidate must be runnable locally on CPU with a
production-compatible license.

`BAAI/bge-reranker-v2-m3` remains a previously researched candidate, not an irrevocable
selection. Model choice must be justified by multilingual quality, local CPU viability,
license, deterministic behavior, latency, and memory cost.

## Preregistered promotion rule

A reranker is eligible for a later production-integration gate only if all checks pass against
the frozen hybrid baseline:

1. nDCG@10 improves by at least `+0.02`;
2. MRR@10 does not regress;
3. Recall@5 does not regress;
4. first-stage candidate generation remains unchanged;
5. added p95 reranking latency is at most `500 ms` on the GitHub Actions CPU runner;
6. measured peak RSS overhead is at most `2048 MiB`;
7. CPU time, memory, quality, and latency are all reported explicitly.

A correctly executed experiment may produce `REJECT` and still be operationally valid. A
rejected candidate is not integrated. The policy must not be edited merely to turn a reject
into a pass.

## Baseline gate

The baseline itself must satisfy all of these conditions before any reranker candidate is
scored:

- at least 30 frozen queries are evaluated;
- hybrid aggregate MRR@10 is not below BM25;
- hybrid aggregate Recall@5 is not below BM25;
- hybrid aggregate nDCG@10 is not below BM25;
- hybrid candidate-pool Recall@10 is at least `0.95`;
- warm hybrid API p95 is at most `500 ms`;
- the workflow prints `Phase 3F frozen hybrid baseline: PASS`;
- the machine-readable baseline report is retained as a workflow artifact.

After the first valid baseline run, its measured quality/ranking result must be versioned as
immutable evidence before candidate reranking is introduced.

## Certification discipline

Phase 3F is not certified by this baseline alone. Any later production integration must again
satisfy the repository-wide exact-SHA rule: all prior workflows plus the Phase 3F workflow on
the same candidate SHA, review with no blocker/high defect, merge of exactly that candidate,
and complete recertification on the exact resulting `main` SHA.
