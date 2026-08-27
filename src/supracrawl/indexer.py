from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_markdown
from .config import Settings
from .extractor import Extraction, Extractor, content_hash
from .fetcher import FetchError, FetchResult
from .search import OpenSearchStore, SearchBackendError
from .security import UnsafeUrlError


@dataclass(slots=True)
class IndexOutcome:
    url: str
    indexed: bool
    document_id: str | None = None
    content_hash: str | None = None
    chunks_indexed: int = 0
    error: str | None = None


class Indexer:
    def __init__(self, settings: Settings, extractor: Extractor, store: OpenSearchStore) -> None:
        self.settings = settings
        self.extractor = extractor
        self.store = store

    async def index_url(self, url: str) -> IndexOutcome:
        try:
            fetched, extraction = await self.extractor.extract(url)
        except UnsafeUrlError as exc:
            return IndexOutcome(url=url, indexed=False, error=f"Unsafe URL: {exc}")
        except FetchError as exc:
            return IndexOutcome(url=url, indexed=False, error=f"Fetch failed: {exc}")

        return await self.index_extraction(fetched, extraction)

    async def index_extraction(
        self,
        fetched: FetchResult,
        extraction: Extraction,
    ) -> IndexOutcome:
        digest = content_hash(extraction.markdown)
        chunks = chunk_markdown(
            extraction.markdown,
            target_tokens=self.settings.index_chunk_target_tokens,
        )
        if not chunks:
            return IndexOutcome(
                url=fetched.final_url,
                indexed=False,
                content_hash=digest,
                error="No indexable content extracted",
            )

        try:
            doc_id, chunks_indexed = await self.store.index_document(
                fetched=fetched,
                extraction=extraction,
                chunks=chunks,
                content_hash=digest,
            )
        except SearchBackendError as exc:
            return IndexOutcome(
                url=fetched.final_url,
                indexed=False,
                content_hash=digest,
                error=str(exc),
            )

        return IndexOutcome(
            url=fetched.final_url,
            indexed=True,
            document_id=doc_id,
            content_hash=digest,
            chunks_indexed=chunks_indexed,
        )
