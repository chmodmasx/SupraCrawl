from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from . import __version__
from .cache import JsonCache
from .chunking import approx_tokens, chunk_markdown, select_chunks
from .config import get_settings
from .crawler import Crawler
from .extractor import Extractor, content_hash
from .fetcher import FetchError, HttpFetcher
from .indexer import Indexer
from .models import (
    CrawlPage,
    CrawlRequest,
    CrawlResponse,
    ExtractedDocument,
    ExtractMetadata,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    IndexItem,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .search import OpenSearchStore, SearchBackendError
from .security import UnsafeUrlError

settings = get_settings()
cache = JsonCache(settings.redis_url, settings.cache_ttl_s)
fetcher = HttpFetcher(settings)
extractor = Extractor(settings, fetcher)
search_store = OpenSearchStore(settings)
indexer = Indexer(settings, extractor, search_store)
crawler = Crawler(fetcher, extractor, indexer)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await cache.close()
    await search_store.close()


app = FastAPI(title="SupraCrawl", version=__version__, lifespan=lifespan)


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    documents: list[ExtractedDocument] = []
    max_tokens = request.max_context_tokens or settings.max_context_tokens

    for input_url in request.urls:
        url = str(input_url)
        cache_key = cache.key("extract", f"{url}|{request.query or ''}|{max_tokens}")
        if not request.force_refresh:
            cached = await cache.get(cache_key)
            if cached:
                try:
                    cached_document = ExtractedDocument.model_validate(cached)
                except ValidationError:
                    # A stale/corrupt-but-valid JSON object must behave as a cache miss.
                    pass
                else:
                    documents.append(cached_document)
                    continue

        try:
            fetched, result = await extractor.extract(url)
        except UnsafeUrlError as exc:
            documents.append(
                ExtractedDocument(
                    url=url,
                    error=f"Unsafe URL: {exc}",
                )
            )
            continue
        except FetchError as exc:
            documents.append(
                ExtractedDocument(
                    url=url,
                    error=f"Fetch failed: {exc}",
                )
            )
            continue

        chunks = chunk_markdown(result.markdown)
        selected = select_chunks(
            chunks,
            query=request.query,
            max_tokens=max_tokens,
            max_chunks=settings.max_chunks_per_document,
        )
        content = "\n\n---\n\n".join(chunk.text for chunk in selected).strip()
        context_tokens = approx_tokens(content) if content else 0

        document = ExtractedDocument(
            url=fetched.final_url,
            title=result.title,
            content=content,
            raw_content="",
            metadata=ExtractMetadata(
                canonical_url=result.canonical_url,
                final_url=fetched.final_url,
                fetched_at=fetched.fetched_at,
                rendered=result.rendered,
                extractor=result.extractor,
                extraction_quality=result.quality,
                content_hash=content_hash(result.markdown),
                http_status=fetched.status_code,
                content_type=fetched.content_type,
                chunks_total=len(chunks),
                chunks_selected=len(selected),
                context_tokens_approx=context_tokens,
                provenance=[
                    {
                        "source_url": fetched.final_url,
                        "fetch_url": fetched.fetch_url,
                        "selected_chunk_ordinals": [chunk.ordinal for chunk in selected],
                    }
                ],
            ),
        )
        await cache.set(cache_key, document.model_dump(mode="json"))
        documents.append(document)

    return ExtractResponse(documents=documents)


@app.post("/v1/index", response_model=IndexResponse)
async def index_urls(request: IndexRequest) -> IndexResponse:
    items: list[IndexItem] = []
    for input_url in request.urls:
        outcome = await indexer.index_url(str(input_url))
        items.append(IndexItem.model_validate(asdict(outcome)))
    return IndexResponse(items=items)


@app.post("/v1/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest) -> CrawlResponse:
    outcomes = await crawler.crawl(
        seeds=[str(seed) for seed in request.seeds],
        max_pages=request.max_pages,
        max_depth=request.max_depth,
        same_origin=request.same_origin,
    )
    pages = [CrawlPage.model_validate(asdict(outcome)) for outcome in outcomes]
    return CrawlResponse(
        pages_visited=len(pages),
        pages_indexed=sum(1 for page in pages if page.indexed),
        pages=pages,
    )


@app.post("/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    try:
        raw_results = await search_store.search(request.query, request.limit)
    except SearchBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    results = [SearchResult.model_validate(result) for result in raw_results]
    return SearchResponse(results=results)
