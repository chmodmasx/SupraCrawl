# Phase 3D acceptance — controlled hybrid production capability

Phase 3D certifies the Phase 3C hybrid winner as a production-capable **opt-in** retrieval path. It does not change the global production default. BM25 remains the default until a later, separately reviewed and certified promotion gate.

The machine-readable policy is frozen in `evaluation/phase3d_policy.json`. The policy was committed before Phase 3D implementation work.

## Scope

Phase 3D connects the already-certified local multilingual E5 embedding runtime and exact OpenSearch vector retrieval to the normal indexing and search API paths. It must preserve the lexical path as the authoritative fallback and must make hybrid degradation visible rather than silently pretending dense retrieval succeeded.

This phase does not evaluate ANN/HNSW, GPU inference, large-scale vector capacity, or a default switch from BM25 to hybrid.

## Frozen retrieval behavior

- Configured default search mode: `bm25`.
- Default `dense_enabled`: `false`.
- Supported request modes: `bm25` and `hybrid`.
- If a request omits a mode, the configured default is used.
- An explicit `hybrid` request attempts BM25 plus local dense retrieval and deterministic reciprocal-rank fusion.
- Hybrid uses `intfloat/multilingual-e5-small`, 384 dimensions, E5 query/passage prefixes, and RRF `k=60`.
- Vector storage remains the certified OpenSearch Lucene `flat` cosine path from Phase 3C.

## Correctness requirements

### Backward/default safety

1. A legacy `POST /v1/search` body containing only `query` and `limit` still works and uses BM25 under default settings.
2. `Settings.search_mode` defaults to `bm25` and `dense_enabled` defaults to `false`.
3. Existing Hermes search requests do not need a new field and remain BM25 by default in Phase 3D.
4. The existing `results` response field remains backward compatible.
5. A BM25 request must not require the embedding model to load.

### Hybrid API path

6. An explicit hybrid request with dense capability enabled must run the local query embedder, query the real OpenSearch vector index, fuse lexical and dense document rankings with RRF, and report that hybrid was actually used.
7. Successful hybrid retrieval must report `degraded=false`.
8. Returned hybrid positions must be deterministic and contiguous from 1.
9. Hybrid metadata must expose enough provenance to distinguish lexical/dense ranks and the retrieval mode without changing the Hermes search result contract.

### Current-content protection

10. Dense candidates must carry `document_id` and `content_hash` provenance.
11. Before fusion, dense candidates must be checked against the current `content_hash` stored in the documents index.
12. A dense candidate whose hash does not match the current document is discarded even if the stale vector is still physically present.
13. Failure to validate current hashes is a vector-side failure and must degrade the request to BM25.
14. Successful vector reindexing must still remove stale vector chunks as storage hygiene; read-time validation is an independent correctness guard.

### Production indexing

15. Lexical indexing remains authoritative and occurs for every successful index operation.
16. When dense capability is enabled, the same extracted chunks are embedded locally and written to the vector index with model/dimension provenance.
17. A vector indexing failure after a successful lexical write must not convert that lexical success into an indexing failure.
18. Index responses must explicitly report vector indexing success/failure and vector chunk counts when the vector path is attempted.
19. A later hybrid search must never surface stale vectors from a failed vector refresh because of the current-hash guard.

### Degradation/failure semantics

20. Embedder unavailable or inference failure: return BM25 results with explicit degradation metadata.
21. Vector index unavailable or invalid mapping: return BM25 results with explicit degradation metadata.
22. Dense capability disabled for an explicit hybrid request: return BM25 results with explicit degradation metadata.
23. Current-hash validation unavailable: return BM25 results with explicit degradation metadata.
24. Lexical BM25/OpenSearch failure remains a request failure (`503`); hybrid must not mask loss of the lexical backbone.

## Frozen retrieval spot checks

The exact query strings and targets are frozen in `evaluation/phase3d_policy.json` and reuse already-labelled Phase 3C data.

- The two full Git SHA queries (`q21`, `q22`) must return their exact target at rank 1 under the hybrid API path.
- Cross-language semantic queries `q13` and `q14` must keep their labelled target within the top 5 under the hybrid API path.

These checks are in addition to running the existing Phase 3C benchmark unchanged as a regression workflow.

## Performance gate

After model warm-up and excluding first model download, warm end-to-end **hybrid API** p95 must be at most 500 ms. Query embedding time is included.

The limit intentionally matches the Phase 3C guardrail because GitHub-hosted CPU performance varies. A tighter SLO can be introduced only in a separately measured optimization gate.

## Packaging/runtime requirements

- The standard SupraCrawl API container must contain the local hybrid runtime; a hosted embedding API or API key is not allowed.
- The embedding model remains lazy-loaded so default BM25 startup/search does not initialize inference.
- The real Compose stack must be able to exercise explicit hybrid indexing/search.
- OpenSearch provenance must still fail closed on incompatible vector model/dimension/mapping.

## Hermes requirement

The existing Hermes provider contract remains unchanged in Phase 3D. Its normal search request continues to omit a retrieval mode, which means the certified default stays BM25. The provider must continue to pass its pinned real-Hermes contract suite.

A Hermes-specific hybrid opt-in can be added later, but it is not required to certify the backend production capability and must not be smuggled into this gate after measurement.

## Certification matrix

Phase 3D cannot close unless all of the following are true on one exact candidate SHA and again on the exact merged `main` SHA:

1. Ruff passes.
2. The complete Python suite passes both with the baseline runtime and with hybrid dependencies installed.
3. Real OpenSearch vector indexing and hybrid API retrieval pass.
4. The frozen exact-ID and semantic spot checks pass through the API.
5. Hybrid degradation tests pass for disabled dense, embedder failure, vector failure, and current-hash validation failure.
6. Lexical failure still produces `503`.
7. Real indexing verifies lexical success survives a vector-side failure and that stale vector content cannot leak into hybrid results.
8. The standard Docker API image contains the local hybrid runtime while BM25 remains lazy with respect to model loading.
9. The pinned real Hermes provider contract passes.
10. Every earlier Phase 1, Phase 2, Phase 3, Phase 3B, and Phase 3C workflow remains green.
11. The dedicated Phase 3D workflow is green.
12. Compose teardown is clean.
13. No known blocker or high-severity defect remains in the defined Phase 3D scope.

## Promotion boundary

A Phase 3D PASS means only that `hybrid` is certified as a production-capable opt-in retrieval mode with BM25 fallback. It **does not** authorize changing the global default from `bm25` to `hybrid`.

Changing the default requires a separate gate after Phase 3D is merged and certified on `main`.
