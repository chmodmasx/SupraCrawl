from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

SearchMode = Literal["bm25", "hybrid"]


class ExtractRequest(BaseModel):
    urls: list[HttpUrl] = Field(min_length=1, max_length=10)
    query: str | None = Field(default=None, max_length=2000)
    max_context_tokens: int | None = Field(default=None, ge=128, le=32_000)
    force_refresh: bool = False


class ExtractMetadata(BaseModel):
    canonical_url: str
    final_url: str
    fetched_at: str
    language: str | None = None
    rendered: bool = False
    extractor: str
    extraction_quality: float = Field(ge=0.0, le=1.0)
    content_hash: str
    http_status: int
    content_type: str
    chunks_total: int
    chunks_selected: int
    context_tokens_approx: int
    provenance: list[dict[str, Any]] = Field(default_factory=list)


class ExtractedDocument(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    raw_content: str = ""
    metadata: ExtractMetadata | None = None
    error: str | None = None


class ExtractResponse(BaseModel):
    success: bool = True
    documents: list[ExtractedDocument]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    mode: SearchMode | None = None


class SearchResult(BaseModel):
    title: str
    url: str
    description: str
    position: int = Field(ge=1)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    success: bool = True
    results: list[SearchResult]
    mode_requested: SearchMode = "bm25"
    mode_used: SearchMode = "bm25"
    degraded: bool = False
    degradation_reason: str | None = None


class IndexRequest(BaseModel):
    urls: list[HttpUrl] = Field(min_length=1, max_length=50)


class IndexItem(BaseModel):
    url: str
    indexed: bool
    document_id: str | None = None
    content_hash: str | None = None
    chunks_indexed: int = Field(default=0, ge=0)
    vector_indexed: bool | None = None
    vector_chunks_indexed: int = Field(default=0, ge=0)
    vector_error: str | None = None
    error: str | None = None


class IndexResponse(BaseModel):
    success: bool = True
    items: list[IndexItem]


class CrawlRequest(BaseModel):
    seeds: list[HttpUrl] = Field(min_length=1, max_length=10)
    max_pages: int = Field(default=25, ge=1, le=100)
    max_depth: int = Field(default=1, ge=0, le=3)
    same_origin: bool = True


class CrawlPage(BaseModel):
    url: str
    depth: int = Field(ge=0)
    indexed: bool
    document_id: str | None = None
    content_hash: str | None = None
    chunks_indexed: int = Field(default=0, ge=0)
    error: str | None = None


class CrawlResponse(BaseModel):
    success: bool = True
    pages_visited: int = Field(ge=0)
    pages_indexed: int = Field(ge=0)
    pages: list[CrawlPage]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "SupraCrawl"
    version: str
