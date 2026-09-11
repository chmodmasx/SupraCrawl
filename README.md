# SupraCrawl

SupraCrawl is a self-hosted web retrieval and extraction backend designed for LLM agents, with Hermes Agent as the first integration target.

Its guiding principle is simple:

> Fetch once, clean once, index once, and send the model only what answers the query.

## Goals

SupraCrawl provides a local, controllable alternative to hosted web-extraction services while optimizing for **quality per token**, not raw HTML throughput.

The project separates three concerns:

1. **Discovery** — find URLs to process.
2. **Extraction** — convert a URL into clean, structured content.
3. **Retrieval** — search already processed content and return only the most relevant passages.

SupraCrawl is not attempting to build a whole-web search engine in one step. Search operates over the corpus that SupraCrawl has explicitly indexed or crawled.

## Current architecture

```text
                    Hermes Agent
                        |
             web_search / web_extract
                        |
               SupraCrawl provider
                        |
                  FastAPI Gateway
          +-------------+-------------+
          |             |             |
      /v1/search    /v1/index     /v1/extract
          |             |             |
      Retrieval       Indexer      Extraction
      /      \           |             |
   BM25      dense     HTTP fetch    HTTP-first
     \        /          |             |
        RRF          chunking       clean enough?
         |               |          /          \
   current-hash      lexical +     yes          no
     guard           vector write   |            |
         |               |       static       Playwright
   document collapse ----+------- extraction   fallback
         |
   compact passages
```

The promoted retrieval default is `hybrid`: BM25 remains the authoritative lexical backbone, multilingual E5 provides local dense retrieval, and deterministic reciprocal-rank fusion combines both rankings. Any vector-side failure degrades explicitly to BM25. Operators can still force BM25.

A separately packaged Phase 3G reranker can optionally reorder the certified hybrid top-10 while preserving first-stage top-5 membership. It remains disabled by default and is not part of the standard production image. Phase 3H adds independently opt-in admission backpressure, Phase 3I adds process-local operational metrics, Phase 3J separates liveness from serving readiness, Phase 4A adds opt-in freshness admission for crawl reindex work, Phase 4B adds independently opt-in conditional HTTP revalidation for crawl leaves, Phase 4C measures the deterministic network/indexing savings of those certified refresh paths, and Phase 4D adds independently opt-in process-local fixed-delay scheduling that delegates every cycle to the same certified crawler.

## Design rules

- HTTP fetch first; browser rendering only as fallback.
- No generative LLM in extraction or retrieval.
- Preserve headings, lists, tables, code blocks and useful links.
- Treat `rel=canonical` as a signal, not an absolute truth.
- Block SSRF destinations and revalidate every redirect.
- Respect `robots.txt` in the fetch path.
- Cache successful extract responses, while indexing remains an explicit operation.
- Store provenance and extraction metadata.
- Do not return full pages to the agent by default.
- Search chunks, but collapse results by document so one page cannot occupy every result slot.
- Keep lexical indexing authoritative even when vector indexing fails.
- Validate dense candidates against the current document `content_hash` before fusion.
- Introduce rerankers only after a benchmark proves an additional gain.
- Keep reranker rollout isolated from the certified default retrieval path and fail back to first-stage hybrid.
- Bound reranker queueing explicitly when backpressure is enabled rather than allowing unbounded request buildup.
- Keep operational metrics read-only and independent from ranking decisions.
- Keep liveness cheap and independent from dependency health; use readiness for serving-path admission decisions.
- Keep crawl freshness opt-in and fail open to normal reindexing when freshness cannot be established safely.
- Allow target-page network skipping only for crawl leaves, after the same SSRF/robots admission as normal fetching, so BFS discovery semantics remain unchanged.
- Keep scheduled refresh independently opt-in and process-local; scheduling must reuse the certified crawler rather than introduce a second fetch/index path.

## API

### Extract

```text
POST /v1/extract
```

Fetches and cleans up to 10 URLs, then returns only the selected passages that fit the context budget.

### Index

```text
POST /v1/index
```

Fetches, extracts and chunks up to 50 explicit URLs. Under the promoted defaults, successful lexical indexing is followed by local E5 embedding and OpenSearch vector indexing. A vector-side failure is reported without erasing a successful lexical write.

### Crawl

```text
POST /v1/crawl
```

Runs a bounded breadth-first crawl using the same SSRF, redirect, MIME, size and robots protections as extraction. Defaults to same-origin discovery and is hard-limited by request depth/page budgets.

`refresh_after_s` is an additive opt-in freshness window. Its default is `0`, which preserves the legacy crawl path and performs normal extraction/indexing on every successful fetch. Under the certified Phase 4A behavior, values greater than zero still fetch each visited page so BFS link discovery is preserved, then query existing document metadata by exact persisted final URL. A page is considered fresh only when its indexed age is strictly less than the requested window.

Fresh pages skip extraction, chunking, embeddings and lexical/vector reindexing. They are reported as visited but not indexed with `freshness_skipped: true`, `document_id`, `content_hash`, and `freshness_age_s`; `pages_skipped_fresh` reports the aggregate count. Invalid or naive timestamps, an age exactly equal to the window, malformed lookup responses, or OpenSearch lookup failures fall through to normal reindexing.

Phase 4B adds `conditional_revalidate_leaves`, independently opt-in and `false` by default. It is active only when `conditional_revalidate_leaves=true`, `refresh_after_s>0`, and the current page is a crawl leaf (`depth >= max_depth`). Non-leaf pages always retain the Phase 4A full-fetch path so child discovery is unchanged.

For an eligible fresh leaf, SupraCrawl first performs the same public-URL and `robots.txt` admission used by normal fetching; only after that admission succeeds may it skip the target-page GET. Such pages set `network_fetch_skipped: true` and contribute to `pages_network_skipped_fresh`. A stale leaf with persisted `ETag` and/or `Last-Modified` sends `If-None-Match` / `If-Modified-Since`. Conditional validators are dropped before following redirects.

An HTTP `304 Not Modified` skips extraction and lexical/vector reindexing and refreshes the stored document timestamp. It is reported with `revalidated_not_modified: true` and contributes to `pages_revalidated_not_modified`. If the metadata touch fails, SupraCrawl immediately performs an unconditional full GET and resumes the normal extraction/indexing path. Conditional `200` responses also continue through the normal path. Missing validators, lookup failures, or best-effort validator-persistence failures never make crawling fail closed.

Phase 4D can schedule those same crawl semantics without adding a new endpoint or a scheduler-specific fetch path. Scheduling is disabled by default. When enabled, one `asyncio` task per API process runs an immediate crawl cycle and then waits the configured interval **after the cycle finishes or fails** before starting the next cycle, so cycles do not overlap. A cycle exception is logged and contained; later cycles continue. Application shutdown cancels and awaits the task before closing shared resources.

Scheduler configuration:

```text
SUPRACRAWL_CRAWL_SCHEDULER_ENABLED=false
SUPRACRAWL_CRAWL_SCHEDULER_SEEDS=[]
SUPRACRAWL_CRAWL_SCHEDULER_INTERVAL_S=21600
SUPRACRAWL_CRAWL_SCHEDULER_MAX_PAGES=25
SUPRACRAWL_CRAWL_SCHEDULER_MAX_DEPTH=1
SUPRACRAWL_CRAWL_SCHEDULER_SAME_ORIGIN=true
SUPRACRAWL_CRAWL_SCHEDULER_REFRESH_AFTER_S=21600
SUPRACRAWL_CRAWL_SCHEDULER_CONDITIONAL_REVALIDATE_LEAVES=true
```

`SUPRACRAWL_CRAWL_SCHEDULER_SEEDS` is a JSON list of absolute HTTP(S) URLs. Enabling the scheduler with an empty seed list is rejected at settings validation. The minimum scheduler interval is 60 seconds. Because Phase 4D is intentionally process-local, enabling it in multiple API replicas causes each replica to schedule its own cycles; Phase 4D provides no distributed leader election, persistent scheduler state, or cross-process deduplication.

### Search

```text
POST /v1/search
```

A request that omits `mode` uses the configured default. The promoted default is `hybrid`, combining BM25 and local multilingual E5 retrieval with RRF. `mode: "bm25"` remains available for an explicit lexical-only request.

Hybrid responses report `mode_requested`, `mode_used`, `degraded`, and `degradation_reason`. If the vector path is disabled or unavailable, the request falls back to BM25; failure of the lexical backbone remains a request failure.

When the optional Phase 3G reranker is enabled, responses additionally report `reranker_enabled`, `reranker_used`, `reranker_degraded`, `reranker_degradation_reason`, `reranker_queue_wait_ms`, and `reranker_inference_ms`. Explicit BM25 requests bypass reranking. A reranker load or inference failure preserves the certified first-stage hybrid ranking and is reported separately from retrieval degradation. With Phase 3H backpressure enabled, capacity saturation also fails open to the first-stage hybrid ranking instead of allowing unbounded queue growth.

### Health

```text
GET /v1/health
```

Returns cheap process liveness only: service identity and version. It intentionally does not test OpenSearch, dense embeddings, vector mappings, or reranker readiness, so dependency failures do not make the liveness probe fail.

### Readiness

```text
GET /v1/ready
```

Returns `200` with `status: "ready"` only when the process can serve the configured retrieval path, otherwise `503` with structured component status and reason fields.

Readiness validates the lexical OpenSearch indexes for every configured mode. For `hybrid`, it also requires dense retrieval to be enabled, validates the local dense runtime once per process, and validates the vector index mapping. BM25 mode does not require the dense runtime. When reranking is enabled, readiness requires startup warmup to be configured; readiness itself never loads the reranker model. Because application startup awaits that warmup, a reachable process with the requirement satisfied has already crossed the model-load boundary before serving traffic.

Search failure and degradation semantics remain unchanged: readiness is an operational admission signal, not a replacement for the existing fail-open retrieval behavior.

### Metrics

```text
GET /v1/metrics
```

Returns schema-versioned, process-local operational counters for completed search work. The endpoint reports request/success/backend-error counts, retrieval degradation, reranker enabled/used/degraded counts, capacity versus other reranker degradation, and queue/inference observation counts with sum/max timing aggregates. It also reports whether reranking and backpressure are enabled in that process.

Metrics are read-only and reset only when the serving process restarts. There is no remote reset operation and no external metrics dependency in the Phase 3I capability.

## Retrieval defaults

```text
SUPRACRAWL_SEARCH_MODE=hybrid
SUPRACRAWL_DENSE_ENABLED=true
SUPRACRAWL_RERANKER_ENABLED=false
SUPRACRAWL_RERANKER_BACKPRESSURE_ENABLED=false
```

Operator opt-out:

```text
SUPRACRAWL_SEARCH_MODE=bm25
SUPRACRAWL_DENSE_ENABLED=false
```

The certified hybrid configuration is:

- `intfloat/multilingual-e5-small`
- 384 dimensions
- E5 `query: ` / `passage: ` prefixes
- RRF `k=60`
- OpenSearch Lucene `flat` cosine vector index

No hosted embedding API or API key is required.

## Controlled reranker canary

Phase 3G uses a dedicated `Dockerfile.reranker` image so the standard production image retains the previously certified hybrid dependency path. The canary runtime pins the exact Candidate 2 environment and preloads the frozen model for offline startup.

The frozen reranker identity is:

- model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- revision: `1427fd6`
- runtime: FastEmbed `0.8.0`
- ONNX: `onnx/model_quint8_avx2.onnx`
- ONNX SHA-256: `6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b`
- candidate pool: first-stage top-10
- protected set: first-stage top-5 membership
- strategy: `score_desc_within_frozen_top5_and_tail`

The certified Phase 3G capability includes live-gated model warmup, offline model verification, exact provenance, top-10/top-5 invariants, stable ranking, BM25 bypass, real failure fallback, and a latched model-load failure boundary.

Its preregistered live resource gate originally passed with:

- warm added p95: `275.988 ms` against a `500 ms` maximum;
- peak RSS delta: `407.742 MiB` against a `2048 MiB` maximum;
- 24 concurrent requests in 3 rounds of width 8 with no failures/degradations and deterministic ranking;
- incremental CPU reported as `41.07 s` for the measured workload.

The same Phase 3G load test observed reranker concurrent p95 of `2083.837 ms` versus `392.841 ms` for the baseline. No concurrent-latency threshold was preregistered, so this is not treated as a failed historical gate, but it is a rollout constraint: the reranker remains **default OFF / canary only** rather than a global default.

Phase 3H adds an independently opt-in admission timeout of `200 ms` while keeping inference concurrency fixed at 1. Its certified 8-request burst produced `625.329 ms` concurrent API p95 against a preregistered `1000 ms` maximum, with 6 capacity fallbacks, 2 successful reranks, and successful post-burst recovery. Saturation is not latched as a model failure.

Phase 3I adds process-local operational accounting. Its first certified code candidate recorded a controlled 8-request backpressure burst as 7 capacity fallbacks and 1 successful rerank with zero other degradation, 8 queue observations and 1 inference observation. The same candidate reran the inherited Phase 3G resource gate successfully with `293.507 ms` warm added p95 and `404.379 MiB` peak RSS delta, and reran the Phase 3H gate at `610.865 ms` concurrent API p95. Phase 3I was then certified again after merge on `main` at `200bac77659b2cfae828585643fb7bfe778d0f4d`.

The Phase 3J code candidate preserves those historical gates before running readiness checks. On `93c63330a7ef59a1934f8dc5dd6872203d1089d4`, the inherited Phase 3G resource gate passed with `275.659 ms` warm added p95 and `406.32 MiB` peak RSS delta, while Phase 3H passed at `796.321 ms` concurrent API p95. Only after those gates and Phase 3I metrics passed did the readiness gate execute.

## Local stack

```bash
docker compose up -d --build
```

The Compose stack includes:

- FastAPI API with the local FastEmbed/ONNX hybrid runtime
- Readability / Playwright extraction worker
- Redis extraction cache
- OpenSearch 3.8.0

The standard Compose API does not bundle or enable the optional reranker runtime. The dedicated canary image is required for the Phase 3G/3H reranker path.

For orchestration, use `/v1/health` as liveness and `/v1/ready` as the traffic-admission readiness probe. A process may remain live while readiness returns `503` if a required serving dependency is unavailable.

The Compose file exposes the Phase 4D scheduler settings but keeps scheduling disabled by default. To enable it, provide a non-empty JSON seed list and explicitly set `SUPRACRAWL_CRAWL_SCHEDULER_ENABLED=true`.

OpenSearch security is disabled in the provided single-node Compose configuration. That configuration is for local/self-hosted development on a trusted host; do not expose port 9200 to an untrusted network without enabling proper OpenSearch security and network controls.

## Hermes

Set:

```bash
export SUPRACRAWL_URL=http://127.0.0.1:8080
```

Then Hermes can use SupraCrawl for both capabilities:

```yaml
web:
  search_backend: supracrawl
  extract_backend: supracrawl
```

The Hermes provider does not need a SupraCrawl-specific retrieval-mode field. It uses the backend default, so a healthy default deployment uses hybrid retrieval and transparently receives BM25 results when the vector side degrades.

Search covers only content already present in the SupraCrawl index. A separate discovery provider can still be used while building the corpus.

## Roadmap

### Phase 1 — Extraction — certified

HTTP-first extraction, Readability/Trafilatura, Playwright fallback, structural chunking, context budgeting, Redis caching, SSRF/robots controls, Hermes `web_extract`, deterministic/live/real-Hermes gates.

### Phase 2 — Search / index — certified

OpenSearch document/chunk indexes, explicit indexing, bounded crawler, stable document identity, stale-chunk cleanup, BM25, result collapse, `/v1/search`, Hermes `web_search`, deterministic/live/real-Hermes gates.

### Phase 3A — Retrieval evaluation — certified

Versioned corpus and qrels, MRR@10, Recall@5, graded nDCG@10, latency/context measurements, frozen BM25 baseline.

### Phase 3B — Dense/hybrid experiment — certified

Local multilingual E5 baseline and deterministic BM25+dense/RRF comparison on the frozen benchmark.

### Phase 3C — Real vector candidate selection — certified

Real OpenSearch vector storage/querying, expanded exact-identifier benchmark, physical stale-vector replacement checks, and hybrid selected as the only eligible candidate.

### Phase 3D — Controlled hybrid production capability — certified

Production local embedding/index/search path, current-content hash validation, vector-side degradation to BM25, lexical failure semantics, standard-container packaging, and real API fault matrix. Hybrid remained opt-in during this phase.

### Phase 3E — Hybrid default promotion — certified

The already-certified hybrid path was promoted to the global default after omitted-mode API requests preserved the frozen quality, exact-identifier, latency, upgrade, fallback and Hermes gates without a ranking/model/schema change.

### Phase 3F — Reranker evaluation and Candidate 2 selection — certified

A preregistered cross-encoder experiment selected the exact top-5-preserving Candidate 2 after the independent frozen holdout passed the quality, Recall@5, latency and memory promotion checks. The holdout also recorded a lexical exact-identifier family regression, so the result does not justify unconditional reranker rollout.

### Phase 3G — Controlled reranker production capability — certified

The exact frozen reranker is packaged in a dedicated canary image while the standard image/default path remains unchanged. Startup warmup, offline model availability, single-flight loading, bounded inference concurrency, exact ranking/provenance invariants, BM25 bypass, failure fallback, load-failure latching, live E2E and resource gates are certified. Global reranker promotion remains out of scope; default OFF / canary is the intended rollout state.

### Phase 3H — Reranker admission backpressure — certified

An independently opt-in `200 ms` admission timeout prevents an 8-request burst from building an unbounded reranker queue. Capacity saturation fails open to the certified hybrid ranking, remains distinct from model failure, and recovers after the burst. The preregistered live p95 gate passed at `625.329 ms` against `1000 ms` while historical Phase 3G behavior remained reproducible with backpressure disabled.

### Phase 3I — Operational search metrics — certified

The read-only `/v1/metrics` endpoint exposes process-local search/reranker/backpressure counters plus queue/inference timing aggregates without new runtime dependencies or ranking changes. Its exact implementation and documentation candidate passed the complete workflow matrix, and the merged `main` SHA `200bac77659b2cfae828585643fb7bfe778d0f4d` was independently certified with 9/9 push workflows and zero failures.

### Phase 3J — Serving readiness contract — certified

`/v1/health` remains a cheap liveness contract while `/v1/ready` validates whether the configured serving path can accept traffic. Healthy baseline, reranker and backpressure canaries return health/readiness `200`; a reranker-enabled process without startup warmup remains live but returns readiness `503` with `reranker_startup_warmup_required`; and an OpenSearch-fault process remains live but returns readiness `503` with `opensearch_unavailable_or_indices_invalid`.

The exact Phase 3J code candidate `93c63330a7ef59a1934f8dc5dd6872203d1089d4` passed `PASS_READINESS_GATE` and the complete 9/9 workflow matrix. The documentation-complete candidate and the merged `main` SHA `09ecd5b67bc54f9758c660b60ca13539697502fd` were then independently certified with 9/9 workflows and zero failures. Readiness does not load the reranker, does not change search degradation semantics, and introduces no new performance threshold.

### Phase 4A — Freshness-aware crawl admission — certified

`/v1/crawl` accepts an opt-in `refresh_after_s` window while preserving `0` as the legacy default. Freshness is checked only after the protected network fetch, so link discovery is unchanged; a fresh hit skips extraction, chunking, embeddings and lexical/vector writes. Freshness lookup failures and unsafe timestamp states fail open to the existing reindex path.

The preregistered code candidate `d3aa2679b94d3b26993302bd94680381c96178c7` passed the complete 9/9 workflow matrix, followed by a documentation-complete candidate and the merged `main` SHA `bf3db918b123c93301ad92ec047717c7af6c7e01`, which was independently certified with 9/9 push workflows and zero failures. Conditional GET and target-page network avoidance were intentionally deferred to the next measured phase.

### Phase 4B — Leaf conditional HTTP revalidation — certified

Phase 4B adds independently opt-in `conditional_revalidate_leaves` behavior on top of the certified Phase 4A freshness baseline. Only crawl leaves are eligible, preserving full-fetch behavior on every page that can discover children. Fresh leaves may skip the target-page GET only after the same SSRF/robots admission as normal fetching succeeds; stale leaves may send persisted `ETag`/`Last-Modified` validators. Redirects discard validators, `304` refreshes the stored document timestamp without reindexing, and a failed metadata touch forces an immediate unconditional GET.

The corrected code candidate `ee3ba836fd5e22b04b97eb6428bbb3c62f3d397b` and documentation-complete candidate `cd8a530daafe7c5a9142ece23b25176d61f88707` each passed the complete 9/9 workflow matrix. The merged `main` SHA `542820dfb5fefe05988ff57f60a81c8d0f395698` was then independently certified with exactly 9/9 push workflows and zero failures.

### Phase 4C — Refresh efficiency measurement — certified

Phase 4C changes no production code. A preregistered deterministic gate compares the certified Phase 4A and Phase 4B refresh paths using a fixed 262,144-byte HTML fixture and counts target GETs, response-body bytes, extraction calls, index writes and metadata touches.

The certified measurement candidate `e01392c612dd66f445f84795272a8516333cc394` passed the complete 9/9 workflow matrix. Against the preregistered fixture, an eligible fresh leaf reduced target GETs and response-body bytes by `100%`; a stale leaf returning `304 Not Modified` reduced response-body bytes, extraction work and index writes by `100%`. The touch-failure control performed one conditional GET followed by one unconditional full GET and restored extraction/indexing, proving the optimization fails back to the full path rather than accepting a false freshness result. The exact evidence is frozen in `evaluation/phase4c_refresh_efficiency_report.json`.

The documentation/evidence-complete candidate `4cd2320cfaf93818f0559e0cf54a58aa339ea2a4` passed the complete 9/9 workflow matrix, and the merged `main` SHA `bf52c6e73e4e4d125e36729736815214df05371f` was independently certified with exactly 9/9 push workflows and zero failures.

### Phase 4D — Process-local refresh scheduler — current gate

Phase 4D adds independently opt-in continuous refresh orchestration without changing the crawler algorithm. A single FastAPI-lifespan-owned task per process delegates every cycle to the certified `Crawler`, starts its first cycle without an initial delay, forbids overlap by waiting for each crawl to finish, waits the full configured interval after completion or failure, contains ordinary cycle exceptions, and cancels cleanly during shutdown.

The preregistered policy-only candidate `b036db5dd1cea2752a631d488643d98c865cfd2e` passed the complete 9/9 workflow matrix before implementation was accepted. The functional candidate `c20d97bfe1e021239fd8dfb410f180586ef9482c` then passed 9/9 with the scheduler default disabled and the manual `/v1/crawl` contract unchanged. The operational configuration candidate `532e7b6a66774cf112411d032a9d3173a7700a62` exposed the same opt-in settings through `.env.example` and Compose and independently passed 9/9.

Phase 4D remains open until this documentation-complete head passes the complete 9-workflow matrix, the PR is merged with its exact head SHA, and the resulting merged `main` SHA independently passes exactly 9/9 push workflows with zero failures.

### Later measured work

- distributed scheduling, leader election or cross-process deduplication only if multi-replica deployment measurements justify it;
- per-domain extraction rules;
- metrics export/aggregation or persistence only when deployment topology requires it;
- persistent originals/provenance storage where justified;
- scale-specific ANN/GPU work only when corpus/load measurements require it;
- any new ranking behavior only with a newly preregistered evaluation and fresh independent validation data.

## Verification policy

A phase is not considered complete because it compiles or because a unit suite is green. Each milestone must pass its deterministic suite, container checks, live E2E acceptance matrix and pinned real-Hermes contract test on the exact candidate SHA and again on the exact merged `main` SHA before it is declared certified.
