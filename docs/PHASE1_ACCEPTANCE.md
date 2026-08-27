# Phase 1 acceptance criteria

Phase 1 is the extraction backend only. It is not considered complete merely because unit tests or container builds pass. The milestone closes only when every acceptance gate below is green and there are no known blocker/high-severity defects in the implemented Phase 1 scope.

## Release gate

All of the following are mandatory before Phase 2 starts:

1. Normal CI is green on the candidate commit.
2. Deterministic Phase 1 acceptance tests are green.
3. The complete Docker Compose stack builds from scratch and passes live E2E verification.
4. The SupraCrawl provider is tested against a pinned real Hermes Agent `WebSearchProvider` implementation.
5. Security regression coverage passes for API SSRF and browser-worker SSRF entry points.
6. Redis and extractor-worker outages degrade as designed instead of taking down extraction.
7. Hard context budgets are never exceeded by the returned selected content.
8. No known blocker/high-severity defect remains open in Phase 1 scope.

## Deterministic matrix

The Python acceptance suite covers:

- localhost, RFC1918, link-local metadata, IPv6 loopback and ULA blocking;
- malformed ports becoming safe validation errors instead of uncaught exceptions;
- IPv6 URL normalization;
- manual redirect chains;
- redirect-target validation before any request is sent;
- 403, 404, 429 and 500 handling;
- unsupported MIME rejection;
- configured HTML size limits;
- transport timeout conversion to `FetchError`;
- `robots.txt` allow/disallow behavior;
- 4xx robots fail-open and 5xx robots fail-closed policy;
- corrupt Redis entries failing open;
- hard context-token budget enforcement;
- mixed batch success/failure ordering.

## Live E2E matrix

The complete Compose stack is exercised against live public pages and must prove:

- API health;
- real static HTML extraction and provenance;
- Redis cache hit behavior;
- `force_refresh` bypass;
- API-level SSRF rejection;
- mixed valid/unsafe batches;
- 128-token hard context budget on real content;
- Spanish extraction;
- non-Latin/Japanese extraction;
- Playwright fallback on a JavaScript-only page;
- browser-worker SSRF rejection;
- Trafilatura fallback while the Readability/Playwright worker is stopped;
- successful extraction while Redis is stopped;
- several simultaneous extraction requests;
- recovery after dependency restarts.

## Hermes compatibility

The verification workflow checks out Hermes Agent at:

`42ac29eacc4d743ed2df7db0f886b99111d9e68b`

The test imports Hermes' real `agent.web_search_provider.WebSearchProvider` and verifies that the SupraCrawl provider:

- is an actual subclass instance;
- advertises extraction but not search;
- implements asynchronous `extract()`;
- resolves configuration via Hermes' config-aware `get_provider_env()`;
- returns the extraction contract expected by Hermes: a list of document dictionaries, not a search-style success/data envelope;
- returns per-URL errors if the SupraCrawl backend cannot be used.

The Hermes pin makes the certification reproducible. Compatibility with a newer Hermes revision should be re-certified by deliberately updating the pin and running the full verification workflow.

## Scope boundary

Passing this matrix certifies Phase 1 extraction behavior covered above. It does not certify future search/indexing functionality, hostile Internet behavior that cannot be simulated exhaustively, or infrastructure controls outside the application process. DNS-rebinding TOCTOU risk still requires production egress/network policy in addition to application validation.
