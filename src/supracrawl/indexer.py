from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_markdown
from .config import Settings
from .embeddings import DenseEmbedder, EmbeddingBackendError, build_passage_text
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
    vector_indexed: bool | None = None
    vector_chunks_indexed: int = 0
    vector_error: str | None = None
    error: str | None = None


class Indexer:
    def __init__(
        self,
        settings: Settings,
        extractor: Extractor,
        store: OpenSearchStore,
        embedder: DenseEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.extractor = extractor
        self.store = store
        self.embedder = embedder
        if settings.dense_enabled and self.embedder is None:
            self.embedder = DenseEmbedder(
                model_name=settings.dense_model_name,
                dimension=settings.dense_dimension,
                query_prefix=settings.dense_query_prefix,
                passage_prefix=settings.dense_passage_prefix,
            )

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

        vector_indexed: bool | None = None
        vector_chunks_indexed = 0
        vector_error: str | None = None
        if self.settings.dense_enabled:
            vector_indexed = False
            try:
                if self.embedder is None:
                    raise EmbeddingBackendError("dense embedder is not configured")
                passages = [
                    build_passage_text(
                        title=extraction.title,
                        section_path=chunk.section_path,
                        text=chunk.text,
                        prefix=self.settings.dense_passage_prefix,
                    )
                    for chunk in chunks
                ]
                vectors = await self.embedder.embed_passages(passages)
                vector_doc_id, vector_chunks_indexed = await self.store.index_vector_document(
                    fetched=fetched,
                    extraction=extraction,
                    chunks=chunks,
                    content_hash=digest,
                    vectors=vectors,
                )
                if vector_doc_id != doc_id:
                    raise SearchBackendError(
                        "vector document identity does not match lexical identity"
                    )
                vector_indexed = True
            except (EmbeddingBackendError, SearchBackendError, ValueError) as exc:
                # Lexical indexing is authoritative. A vector-side failure is
                # reported but must not erase a successful BM25 write. Hybrid
                # reads independently reject stale vectors by current content hash.
                vector_error = str(exc)
                vector_chunks_indexed = 0

        return IndexOutcome(
            url=fetched.final_url,
            indexed=True,
            document_id=doc_id,
            content_hash=digest,
            chunks_indexed=chunks_indexed,
            vector_indexed=vector_indexed,
            vector_chunks_indexed=vector_chunks_indexed,
            vector_error=vector_error,
        )
