# Architecture

## Product boundary

SupraCrawl intentionally separates **discovery**, **extraction**, and **retrieval**. The first production milestone replaces `web_extract`; discovery can remain on an existing Hermes search provider while SupraCrawl proves extraction quality.

The full target is:

```text
Hermes
  -> SupraCrawl provider
      -> FastAPI
          -> extraction pipeline
              -> robots + SSRF validation
              -> HTTP fetch
              -> Readability
              -> quality gate
              -> Playwright fallback only when needed
              -> canonicalization
              -> structural chunking
              -> context budget
          -> retrieval pipeline (Phase 2)
              -> OpenSearch BM25
              -> multilingual embeddings
              -> RRF/hybrid fusion
              -> optional reranker over top-N
              -> page/domain diversity
              -> compact snippets + chunks
```

## Why the browser is a fallback

A browser is slower, more memory-intensive and materially expands the network/security surface. Static HTML therefore takes the normal HTTP path. Browser rendering is attempted only when deterministic extraction is too short/low-quality or the page appears to be an SPA shell.

The browser worker is isolated from the API process. It lazily starts Chromium, uses a new context per render, blocks service workers and validates every HTTP(S) request made by the page. Production deployments should additionally enforce egress policy at the container/network layer; application-level DNS checks alone cannot eliminate every DNS-rebinding race.

## Extraction ensemble

1. Fetch with strict byte, timeout and redirect limits.
2. Parse metadata and canonical hints.
3. Send static HTML to Mozilla Readability.
4. Fall back to Trafilatura if the worker is unavailable or returns nothing.
5. Score extraction quality.
6. Escalate to Playwright only below the quality threshold.
7. Convert the result to structural Markdown.
8. Chunk on heading boundaries before token-size boundaries.
9. Select chunks according to query relevance when a query exists.
10. Return an empty `raw_content` field by default.

No generative model is used in this path.

## Security boundary

URLs are untrusted input. Current controls include:

- HTTP(S)-only URLs;
- no URL credentials;
- DNS resolution before requests;
- block loopback/private/link-local/multicast/reserved/unspecified addresses;
- redirect revalidation;
- MIME allowlist for the HTML pipeline;
- maximum response bytes;
- conservative robots handling;
- browser request interception for subresources.

A hardened deployment should also deny RFC1918/link-local/cloud-metadata destinations at the network/firewall layer. That protects against time-of-check/time-of-use and DNS-rebinding variants that pure application validation cannot fully close.

## Phase 2 index model

Documents and chunks will be separate logical records. Documents hold canonical identity, fetch/extraction metadata and content hashes. Chunks hold section paths, passage text, token counts and optional embeddings. Retrieval occurs over chunks, then groups/diversifies by document and domain before returning results.

Baseline order:

```text
BM25
  -> evaluate
BM25 + multilingual dense retrieval
  -> RRF
  -> evaluate
optional reranker over top-N
  -> evaluate
```

Embeddings and reranking are not accepted as improvements until the benchmark demonstrates gains in MRR/Recall@n/nDCG and quality-per-token.
