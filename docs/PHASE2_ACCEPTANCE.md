# Phase 2 acceptance gate

Phase 2 adds local search and indexing. It is complete only when every gate below passes on the exact commit that will be merged to `main`, and again after merge on `main` itself.

## Functional scope

The Phase 2 baseline includes:

- separate OpenSearch `documents` and `chunks` indexes;
- stable document identity derived from normalized canonical URLs;
- explicit URL indexing through `POST /v1/index`;
- bounded breadth-first crawling through `POST /v1/crawl`;
- reuse of Phase 1 fetch, robots, redirect, MIME, size, SSRF and extraction protections;
- stale-chunk cleanup when a document is reindexed;
- BM25 retrieval over title, heading path and chunk text;
- result collapse by document so one page cannot occupy multiple search positions;
- extractive descriptions from the best matching chunk;
- `POST /v1/search`;
- Hermes `supports_search() -> True` with the exact Hermes search envelope.

Embeddings, hybrid RRF and reranking are explicitly outside this gate. They belong to a later relevance milestone and must not be introduced before the BM25 baseline is measured.

## Deterministic acceptance

The Python suite must cover, at minimum:

- normalized URL identity produces stable document IDs;
- document and chunk bulk payloads are correct;
- stale chunk cleanup is scoped to the current document and old content hash;
- the search request targets BM25 text fields and collapses by `document_id`;
- API search success and backend-unavailable behavior;
- per-URL index failures remain isolated;
- crawl response accounting;
- crawler link normalization/deduplication;
- same-origin and depth limits;
- all Phase 1 regression tests remain green.

Ruff must pass with no ignored new violations.

## Real Hermes contract

The provider must be tested against the pinned real Hermes Agent commit used by the repository's verification workflow.

The gate requires:

- provider is a `WebSearchProvider`;
- `supports_search()` and `supports_extract()` are both true;
- search is compatible with Hermes' synchronous search dispatch;
- search returns `{"success": true, "data": {"web": [...]}}`;
- extraction still returns the direct list of documents expected by Hermes;
- invalid/unavailable backend configuration returns honest failure shapes.

## Live stack acceptance

The complete Docker Compose stack must start from scratch with:

- SupraCrawl API;
- extraction worker with Playwright/Chromium;
- Redis;
- OpenSearch.

The live Phase 2 verifier must then prove all of the following:

1. Explicit indexing of multiple real public pages succeeds.
2. Both OpenSearch indexes contain data.
3. A distinctive query ranks the expected page first through BM25.
4. Search returns an extractive matching passage rather than a full page.
5. An injected stale chunk is removed when the document is reindexed.
6. A bounded same-origin crawl follows internal links and does not escape the origin.
7. Search returns HTTP 503 while OpenSearch is unavailable instead of silently fabricating results.
8. Search recovers after OpenSearch restarts.
9. The verifier exits with `Phase 2 live verification: PASS`.
10. Compose teardown removes containers, networks and volumes cleanly.

## Release rule

Phase 2 is not certified until:

- normal CI is fully green;
- Phase 1 regression verification remains fully green;
- Phase 2 Verification is fully green;
- all checks are repeated successfully on `main` after merge;
- no known blocker/high-severity correctness issue remains open from review.

Only after this gate is closed may work begin on dense embeddings, RRF, reranking or other Phase 2B/Phase 3 features.
