# Phase 3C acceptance — real vector retrieval

Phase 3C answers one question: does the semantic signal that passed the Phase 3B in-memory screening survive when vectors are actually stored and queried in OpenSearch, without sacrificing exact lexical retrieval?

This phase does **not** change the production default. `POST /v1/search` remains BM25 until a later production-promotion change is independently reviewed and certified.

## Frozen inputs

The Phase 3C gate is pre-registered before implementation measurements in `evaluation/phase3c_policy.json`.

The evaluation has two datasets:

1. The certified Phase 3 corpus in `evaluation/corpus.jsonl` and `evaluation/queries.jsonl`. This remains unchanged and is used as a regression/control set.
2. The exact-identifier extension in `evaluation/phase3c_exact_corpus.jsonl` and `evaluation/phase3c_exact_queries.jsonl`. It adds deliberately confusable CVEs, versions, full Git SHAs, API paths, model identifiers, environment-variable names, and IP literals.

The second set exists because a semantic-only corpus can hide an important web-search failure mode: embeddings may consider two identifiers conceptually similar even when only an exact lexical match is correct.

## Vector implementation under test

- Model: `intfloat/multilingual-e5-small`
- Dimension: 384
- Query prefix: `query: `
- Passage prefix: `passage: `
- Storage/query engine: OpenSearch
- Vector field: `knn_vector`
- OpenSearch engine: Lucene
- Method: `flat`
- Space: cosine similarity (`cosinesimil`)
- Fusion candidate: deterministic reciprocal-rank fusion with `k=60`

Lucene `flat` is intentional for this gate. Phase 3C is validating vector correctness and retrieval behavior, not approximate-nearest-neighbor scaling. HNSW/ANN can be evaluated only after the exact vector baseline is certified.

## Required controls

The original BM25 control must reproduce the certified Phase 3 metrics within the tolerance frozen in the policy:

- MRR@10: 0.843750
- Recall@5: 0.875000
- nDCG@10: 0.840367

Real OpenSearch dense retrieval on the original Phase 3 corpus must remain within the pre-registered tolerance of the Phase 3B in-memory dense result. The two cross-language semantic queries q13 and q14 must remain in the dense top 5.

## Expanded candidate gate

Dense and RRF hybrid are evaluated independently against BM25 on the combined original + exact-identifier benchmark. A non-BM25 candidate is eligible only if all of the following are true:

1. nDCG@10 improves by at least 0.02 over BM25.
2. MRR@10 does not regress versus BM25.
3. Recall@5 does not regress versus BM25.
4. Every exact-identifier query retrieves its grade-3 target at rank 1 (100% top-1 rate).
5. q13 and q14 remain within the top 5.
6. Warm p95 end-to-end retrieval latency is at most 500 ms. Query embedding time is included for dense/hybrid.

If both dense and hybrid are eligible, the candidate with higher nDCG@10 wins. If their nDCG values differ by at most 0.005, lower p95 latency is the tie-breaker.

Eligibility is evidence for a later production-promotion decision. It does not modify `/v1/search` in Phase 3C.

## Hard engineering requirements

Phase 3C cannot pass unless all of these are also verified:

- Vectors are physically stored in and queried from OpenSearch; an in-memory cosine scan does not satisfy the gate.
- The vector mapping matches the configured model dimension and cosine space.
- Model/dimension/mapping mismatches fail closed for the vector path instead of silently mixing incompatible embeddings.
- BM25 remains usable when the embedder or vector index is unavailable.
- The default `/v1/search` route remains the certified BM25 implementation.
- Embeddings are local/self-hosted; no hosted embedding API or API key is introduced.
- Vector provenance records the model identifier and dimension.
- Stale vector chunks can be replaced without leaving old content searchable.
- Deterministic unit/regression tests, Ruff, Docker/Compose E2E, the real Hermes contract, and all earlier phase workflows remain green.
- Compose teardown is clean.
- No known blocker or high-severity defect remains in the defined Phase 3C scope.

## Scope limitation

Passing Phase 3C certifies exact vector retrieval correctness and the frozen benchmark. It does not certify ANN/HNSW recall at large scale, GPU throughput, billion-vector capacity, or every possible Internet query distribution. Those require separate measured gates.
