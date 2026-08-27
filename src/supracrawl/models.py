from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


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
    title: str
    content: str
    raw_content: str = ""
    metadata: ExtractMetadata


class ExtractResponse(BaseModel):
    success: bool = True
    documents: list[ExtractedDocument]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "SupraCrawl"
    version: str
