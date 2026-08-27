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

SupraCrawl is not attempting to build a whole-web search engine in one step. The current search capability operates over the corpus that SupraCrawl has explicitly indexed or crawled.

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
      OpenSearch      Indexer      Extraction
       chunks +          |             |
      documents      HTTP fetch     HTTP-first
          |             |             |
       BM25         Readability      clean enough?
          |          / Trafilatura   /          \
      collapse by        |          yes          no
       document       chunking       |            |
          |             |        static       Playwright
      top passage -------+-------- extraction   fallback
```

## Design rules

- HTTP fetch first; browser rendering only as fallback.
- No generative LLM in the normal extraction or BM25 retrieval path.
- Preserve headings, lists, tables, code blocks and useful links.
- Treat `rel=canonical` as a signal, not an absolute truth.
- Block SSRF destinations and revalidate every redirect.
- Respect `robots.txt` in the fetch path.
- Cache successful extract responses, while indexing remains an explicit operation.
- Store provenance and extraction metadata.
- Do not return full pages to the agent by default.
- Search chunks, but collapse results by document so one page cannot occupy every result slot.
- Introduce embeddings and rerankers only after a benchmark proves the gain.

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

Fetches, extracts, chunks and indexes up to 50 explicit URLs into separate OpenSearch `documents` and `chunks` indexes.

### Crawl

```text
POST /v1/crawl
```

Runs a bounded breadth-first crawl using the same SSRF, redirect, MIME, size and robots protections as extraction. Defaults to same-origin discovery and is hard-limited by request depth/page budgets.

### Search

```text
POST /v1/search
```

Runs a BM25 baseline over title, heading path and chunk text. Results are collapsed by document ID and return an extractive passage as the description.

## Local stack

```bash
docker compose up -d --build
```

The Compose stack includes:

- FastAPI API
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

Search covers only content already present in the SupraCrawl index. A separate discovery provider can still be used while building the corpus.

## Roadmap

### Phase 1 — Extraction — certified

- FastAPI service
- HTTPX fetcher
- URL normalization and SSRF protection
- deterministic DOM cleanup
- Mozilla Readability worker
- Playwright fallback for JS-heavy pages
- structural chunking
- context-budgeted output
- extraction quality signals
- Redis caching
- Hermes `web_extract`
- deterministic, live E2E and real-Hermes contract gates

### Phase 2 — Search / index — current milestone

- OpenSearch `documents` + `chunks` indexes
- explicit URL indexer
- bounded same-origin crawler
- stable canonical document IDs
- stale-chunk cleanup on reindex
- BM25 baseline
- document diversity via result collapse
- extractive snippets
- `/v1/search`
- Hermes `web_search`
- dedicated deterministic, live E2E and real-Hermes contract gates

### Phase 2B — Relevance upgrades, only if benchmarks justify them

- multilingual-E5-small dense retrieval baseline
- BM25 + dense fusion with RRF
- optional BGE reranker over a small top-N
- language-aware relevance evaluation

### Phase 3 — Quality and scale

- benchmark corpus and relevance judgments
- continuous crawling and refresh policies
- per-domain extraction rules
- observability and backpressure
- persistent object storage for originals/provenance where needed
- optional model-assisted cleanup for hard residual cases only

## Verification policy

A phase is not considered complete because it compiles or because a unit suite is green. Each milestone must pass its deterministic suite, container checks, live E2E acceptance matrix and pinned real-Hermes contract test before it is merged to `main` and declared certified.
