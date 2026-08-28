# Phase 3E acceptance — hybrid default promotion

Phase 3E is the separate promotion gate required by Phase 3D before the global default may move from BM25 to the already-certified hybrid retrieval path.

The machine-readable policy is frozen in `evaluation/phase3e_policy.json` from base `main` SHA `f06c19d14b01bc053b912f1ad9244bcac184ea66`. The policy must exist before production-default implementation changes.

## Scope

Phase 3E changes only the default selection of an already-certified retrieval implementation:

- default `search_mode`: `bm25` -> `hybrid`;
- default `dense_enabled`: `false` -> `true`;
- Docker Compose defaults follow the application defaults;
- a legacy `/v1/search` request that omits `mode` uses hybrid when the vector path is healthy;
- a legacy request still receives the same `results` field and the Hermes provider contract remains unchanged;
- explicit BM25 remains available as an operator/request opt-out;
- any vector-side failure degrades the default request to BM25 with explicit backend metadata;
- loss of the lexical backbone remains a request failure.

Phase 3E does not change the E5 model, vector dimension, prefixes, RRF `k`, OpenSearch vector mapping, candidate fusion, extraction, crawling, index identity, or Hermes result mapping.

## Frozen retrieval configuration

The certified Phase 3D retrieval configuration remains unchanged:

- model: `intfloat/multilingual-e5-small`;
- dimension: 384;
- query prefix: `query: `;
- passage prefix: `passage: `;
- RRF `k=60`;
- OpenSearch Lucene `flat` cosine vector index;
- lexical BM25 remains the authoritative fallback.

Any ranking/model/schema change requires a different gate.

## Default behavior requirements

1. `Settings.search_mode` defaults to `hybrid`.
2. `Settings.dense_enabled` defaults to `true`.
3. Docker Compose defaults to `SUPRACRAWL_SEARCH_MODE=hybrid` and `SUPRACRAWL_DENSE_ENABLED=true` when the operator does not override them.
4. A `/v1/search` request containing only `query` and `limit` must report `mode_requested=hybrid`, `mode_used=hybrid`, and `degraded=false` when the vector path is healthy.
5. The response `results` field remains backward compatible.
6. An explicit `mode=bm25` request must report BM25 and must not invoke embedding inference.
7. `SUPRACRAWL_SEARCH_MODE=bm25` remains a supported operator opt-out.
8. `SUPRACRAWL_DENSE_ENABLED=false` remains a supported vector-capability opt-out.

## Default indexing requirements

9. With default settings, successful indexing attempts both lexical and vector writes.
10. Lexical indexing remains authoritative.
11. A vector write failure after lexical success must preserve the lexical indexing success and report the vector failure.
12. Existing lexical-only indexes remain searchable after upgrade; default hybrid requests may degrade to BM25 until content is reindexed with vectors.
13. No stale vector may be served as current content; the Phase 3D `content_hash` guard remains mandatory.

## Fallback requirements

A default request that omits `mode` must degrade to BM25, with `mode_requested=hybrid`, `mode_used=bm25`, `degraded=true`, and a non-empty reason when any of these vector-side conditions occurs:

- dense capability disabled;
- embedding runtime unavailable or inference fails;
- vector index unavailable;
- vector mapping/model/dimension incompatible;
- vector path returns no current candidates, including an existing lexical-only corpus;
- current-content hash validation unavailable.

A lexical/OpenSearch BM25 failure must still return HTTP 503. Hybrid must never mask loss of the lexical backbone.

## Full default-API quality gate

The Phase 3C expanded 30-query corpus and qrels remain frozen. Phase 3E must exercise them through the real `/v1/search` API while **omitting the `mode` field**.

After warm-up and excluding the first model download, the aggregate default-path results must satisfy:

- MRR@10 >= 0.96;
- Recall@5 >= 1.00;
- nDCG@10 >= 0.96;
- exact-identifier top-1 rate >= 1.00;
- `q13-cross-language-rrf` target remains within top 5;
- `q14-cross-language-dense` target remains within top 5;
- warm end-to-end API p95 <= 500 ms.

The query embedding time is included in the latency measurement.

The existing Phase 3C workflow must also remain green, so the promotion cannot hide a retrieval regression by changing this gate's implementation.

## Hermes requirement

The Hermes provider continues to send its existing search payload without a SupraCrawl-specific mode field. No provider contract change is required. Under the promoted backend defaults, that unchanged request now uses hybrid when healthy and transparently receives lexical results if the vector path degrades.

The pinned real Hermes `WebSearchProvider` contract suite must remain green.

## Packaging and runtime requirements

- the standard API image continues to contain the local FastEmbed/ONNX hybrid runtime;
- no hosted embedding API or API key is introduced;
- model loading remains lazy;
- an explicit BM25 request must not require the model to initialize;
- a fresh deployment without model/network availability must remain functionally searchable through BM25 degradation;
- Compose teardown must remove containers, network, and test volumes.

## Certification matrix

Phase 3E cannot close until all of the following pass on one exact candidate SHA and again on the exact merged `main` SHA:

1. Ruff.
2. Complete Python suite with baseline dependencies.
3. Complete Python suite with hybrid dependencies.
4. Full real default-API 30-query quality/latency gate through omitted-mode requests.
5. Real OpenSearch vector indexing and current-content protection.
6. Lexical-only upgrade/degradation scenario.
7. Dense-disabled, embedder-failure, vector-failure, hash-validation-failure, and lexical-failure matrix.
8. Explicit BM25 opt-out without embedding inference.
9. Default indexing attempts vector writes while preserving lexical success on vector failure.
10. Pinned real Hermes provider contract.
11. CI, Phase 1, Phase 2, Phase 3, Phase 3B, Phase 3C, and Phase 3D workflows all remain green.
12. Dedicated Phase 3E workflow is green.
13. Clean Compose teardown.
14. No known blocker/high-severity defect remains in the defined Phase 3E scope.

## Non-goals

Phase 3E does not add a reranker, ANN/HNSW, GPU inference, a new embedding model, RRF tuning, continuous crawling, or cross-index transactional refresh. Those remain separate measured/hardening decisions.

## Promotion meaning

A Phase 3E PASS authorizes `hybrid` as the normal default retrieval mode while preserving explicit/operator BM25 opt-out and automatic BM25 degradation for vector-side failures. It does not authorize any later reranker or ranking change.
