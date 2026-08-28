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

### Search

```text
POST /v1/search
```

A request that omits `mode` uses the configured default. The promoted default is `hybrid`, combining BM25 and local multilingual E5 retrieval with RRF. `mode: "bm25"` remains available for an explicit lexical-only request.

Hybrid responses report `mode_requested`, `mode_used`, `degraded`, and `degradation_reason`. If the vector path is disabled or unavailable, the request falls back to BM25; failure of the lexical backbone remains a request failure.

## Retrieval defaults

```text
SUPRACRAWL_SEARCH_MODE=hybrid
SUPRACRAWL_DENSE_ENABLED=true
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

## Local stack

```bash
docker compose up -d --build
```

The Compose stack includes:

- FastAPI API with the local FastEmbed/ONNX hybrid runtime
- Readability / Playwright extraction worker
- Redis extraction cache
- OpenSearch 3.8.0

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

### Phase 3E — Hybrid default promotion — current gate

Promote the already-certified hybrid path to the global default only if omitted-mode API requests preserve the frozen quality, exact-identifier, latency, upgrade, fallback and Hermes gates. No ranking/model/schema changes are allowed in this phase.

### Later measured work

- optional reranker over a small top-N, only if a separate benchmark proves additional value;
- continuous crawling and refresh policies;
- per-domain extraction rules;
- observability and backpressure;
- persistent originals/provenance storage where justified;
- scale-specific ANN/GPU work only when corpus/load measurements require it.

## Verification policy

A phase is not considered complete because it compiles or because a unit suite is green. Each milestone must pass its deterministic suite, container checks, live E2E acceptance matrix and pinned real-Hermes contract test on the exact candidate SHA and again on the exact merged `main` SHA before it is declared certified.
