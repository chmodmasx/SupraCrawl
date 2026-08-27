# SupraCrawl

SupraCrawl is a self-hosted web retrieval and extraction backend designed for LLM agents, with Hermes Agent as the first integration target.

Its guiding principle is simple:

> Fetch once, clean once, index once, and send the model only what answers the query.

## Goals

SupraCrawl is intended to provide a local, controllable alternative to hosted web-extraction services while optimizing for **quality per token**, not raw HTML throughput.

The project separates three concerns:

1. **Discovery** — find relevant URLs.
2. **Extraction** — convert a URL into clean, structured content.
3. **Retrieval** — search already processed content and return only the most relevant passages.

The first milestone focuses on extraction. Whole-web crawling is explicitly out of scope for the MVP.

## Target architecture

```text
                    Hermes Agent
                        |
             web_search / web_extract
                        |
               SupraCrawl provider
                        |
                  FastAPI Gateway
              +---------+---------+
              |                   |
         Retrieval API       Extraction API
              |                   |
        OpenSearch          Redis / Scheduler
       BM25 + vectors              |
              |              HTTP-first fetch
         hybrid fusion             |
              |               clean enough?
          reranker            /           \
              |             yes            no
         top chunks          |              |
              |        deterministic     Playwright
              |          extraction        fallback
              |              \             /
              +---------------+-----------+
                              |
                       canonicalize
                       dedupe / lang
                         chunking
                              |
                         embedding
                              |
                         OpenSearch
```

## Design rules

- HTTP fetch first; browser rendering only as fallback.
- No generative LLM in the normal extraction path.
- Preserve headings, lists, tables, code blocks and useful links.
- Treat `rel=canonical` as a signal, not an absolute truth.
- Block SSRF destinations and revalidate every redirect.
- Cache successful fetch/extract results.
- Store provenance and extraction metadata.
- Do not return full pages to the agent by default.
- Prefer query-aware passage selection and explicit context budgets.
- Introduce embeddings and rerankers only after a benchmark proves the gain.

## Roadmap

### Phase 1 — Extraction

- FastAPI service
- HTTPX fetcher
- URL normalization and SSRF protection
- deterministic DOM cleanup
- Mozilla Readability worker
- Playwright fallback for JS-heavy pages
- structural chunking
- context-budgeted output
- extraction quality signals
- tests and CI

### Phase 2 — Search / index

- OpenSearch document + chunk indexes
- canonicalization and duplicate detection
- BM25 baseline
- multilingual embeddings
- hybrid retrieval / RRF
- page/domain diversity
- extractive snippets
- Hermes `web_search` provider

### Phase 3 — Quality and scale

- benchmark corpus and relevance judgments
- optional BGE reranking over top-N only
- continuous crawling and refresh policies
- per-domain extraction rules
- observability and backpressure
- optional model-assisted cleanup for hard residual cases only

## Status

Initial implementation is under development.
