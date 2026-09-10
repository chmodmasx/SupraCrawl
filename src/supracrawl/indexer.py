from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

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


@dataclass(frozen=True, slots=True)
class FreshIndexHit:
    document_id: str
    content_hash: str
    fetched_at: str
    age_s: float


@dataclass(frozen=True, slots=True)
class RevalidationIndexHit:
    document_id: str
    content_hash: str
    fetched_at: str
    age_s: float
    etag: str | None = None
    last_modified: str | None = None

    @property
    def has_validator(self) -> bool:
        return bool(self.etag or self.last_modified)


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

    async def revalidation_document(
        self,
        url: str,
        *,
        now: datetime | None = None,
    ) -> RevalidationIndexHit | None:
        body = {
            "size": 1,
            "track_total_hits": False,
            "_source": [
                "document_id",
                "content_hash",
                "fetched_at",
                "url",
                "etag",
                "last_modified",
            ],
            "query": {"term": {"url": url}},
            "sort": [{"fetched_at": {"order": "desc"}}],
        }
        try:
            await self.store.ensure_indices()
            response = await self.store._request_with_index_recovery(
                "POST",
                f"/{self.settings.opensearch_documents_index}/_search",
                json=body,
            )
        except SearchBackendError:
            return None

        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        hits_payload = payload.get("hits")
        if not isinstance(hits_payload, dict):
            return None
        hits = hits_payload.get("hits")
        if not isinstance(hits, list) or not hits:
            return None
        hit = hits[0]
        if not isinstance(hit, dict):
            return None
        source = hit.get("_source")
        if not isinstance(source, dict):
            return None

        document_id = source.get("document_id")
        digest = source.get("content_hash")
        fetched_at = source.get("fetched_at")
        if not all(isinstance(value, str) and value for value in (document_id, digest, fetched_at)):
            return None

        try:
            indexed_at = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        if indexed_at.tzinfo is None:
            return None
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            return None
        age_s = max(
            0.0,
            (current.astimezone(UTC) - indexed_at.astimezone(UTC)).total_seconds(),
        )

        etag = source.get("etag")
        last_modified = source.get("last_modified")
        return RevalidationIndexHit(
            document_id=document_id,
            content_hash=digest,
            fetched_at=fetched_at,
            age_s=round(age_s, 3),
            etag=etag if isinstance(etag, str) and etag else None,
            last_modified=(
                last_modified if isinstance(last_modified, str) and last_modified else None
            ),
        )

    async def fresh_document(
        self,
        url: str,
        refresh_after_s: int,
        *,
        now: datetime | None = None,
    ) -> FreshIndexHit | None:
        if refresh_after_s <= 0:
            return None

        hit = await self.revalidation_document(url, now=now)
        if hit is None or hit.age_s >= refresh_after_s:
            return None

        return FreshIndexHit(
            document_id=hit.document_id,
            content_hash=hit.content_hash,
            fetched_at=hit.fetched_at,
            age_s=hit.age_s,
        )

    async def touch_revalidated_document(
        self,
        hit: RevalidationIndexHit,
        fetched: FetchResult,
    ) -> bool:
        if not fetched.not_modified or fetched.status_code != 304:
            return False

        doc = {
            "fetched_at": fetched.fetched_at,
            "http_status": fetched.status_code,
        }
        if fetched.etag:
            doc["etag"] = fetched.etag
        if fetched.last_modified:
            doc["last_modified"] = fetched.last_modified

        try:
            await self.store.ensure_indices()
            response = await self.store._request_with_index_recovery(
                "POST",
                f"/{self.settings.opensearch_documents_index}/_update/{hit.document_id}"
                "?refresh=wait_for",
                json={"doc": doc},
            )
        except SearchBackendError:
            return False
        return response.status_code < 400

    async def _persist_http_validators(self, document_id: str, fetched: FetchResult) -> None:
        doc: dict[str, str] = {}
        if fetched.etag:
            doc["etag"] = fetched.etag
        if fetched.last_modified:
            doc["last_modified"] = fetched.last_modified
        if not doc:
            return

        mapping = {
            "properties": {
                "etag": {"type": "keyword", "ignore_above": 2048},
                "last_modified": {"type": "keyword", "ignore_above": 2048},
            }
        }
        try:
            response = await self.store._request(
                "PUT",
                f"/{self.settings.opensearch_documents_index}/_mapping",
                json=mapping,
            )
            if response.status_code >= 400:
                return
            response = await self.store._request_with_index_recovery(
                "POST",
                f"/{self.settings.opensearch_documents_index}/_update/{document_id}"
                "?refresh=wait_for",
                json={"doc": doc},
            )
            if response.status_code >= 400:
                return
        except SearchBackendError:
            return

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

        await self._persist_http_validators(doc_id, fetched)

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
