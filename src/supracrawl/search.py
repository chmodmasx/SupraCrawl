from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from .chunking import Chunk
from .config import Settings
from .extractor import Extraction
from .fetcher import FetchResult
from .urls import normalize_url


class SearchBackendError(RuntimeError):
    pass


@dataclass(slots=True)
class _DocumentLockState:
    lock: asyncio.Lock
    users: int = 0


def document_id(url: str) -> str:
    normalized = normalize_url(url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_identity_url(final_url: str, canonical_url: str | None) -> str:
    """Use canonical URLs for identity only when they stay on the fetched host.

    A page-controlled cross-origin canonical is useful provenance, but it must not
    be able to overwrite another site's document identity in the local index.
    """
    normalized_final = normalize_url(final_url)
    if not canonical_url:
        return normalized_final

    try:
        normalized_canonical = normalize_url(canonical_url)
    except ValueError:
        return normalized_final

    final_host = (urlsplit(normalized_final).hostname or "").lower()
    canonical_host = (urlsplit(normalized_canonical).hostname or "").lower()
    if final_host and canonical_host == final_host:
        return normalized_canonical
    return normalized_final


def _compact_text(text: str, max_chars: int = 600) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 2].rstrip() + " …"


class OpenSearchStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._indices_ready = False
        self._vector_index_ready = False
        self._indices_lock = asyncio.Lock()
        self._vector_index_lock = asyncio.Lock()
        self._document_locks_guard = asyncio.Lock()
        self._document_locks: dict[str, _DocumentLockState] = {}

    async def close(self) -> None:
        return None

    def _auth(self) -> tuple[str, str] | None:
        if self.settings.opensearch_username and self.settings.opensearch_password:
            return (self.settings.opensearch_username, self.settings.opensearch_password)
        return None

    @asynccontextmanager
    async def _document_lock(self, doc_id: str) -> AsyncIterator[None]:
        async with self._document_locks_guard:
            state = self._document_locks.get(doc_id)
            if state is None:
                state = _DocumentLockState(lock=asyncio.Lock())
                self._document_locks[doc_id] = state
            state.users += 1

        try:
            async with state.lock:
                yield
        finally:
            async with self._document_locks_guard:
                state.users -= 1
                if state.users == 0 and not state.lock.locked():
                    self._document_locks.pop(doc_id, None)

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        base_url = self.settings.opensearch_url
        if not base_url:
            raise SearchBackendError("OpenSearch is not configured")

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.opensearch_timeout_s,
                verify=self.settings.opensearch_verify_tls,
                auth=self._auth(),
            ) as client:
                return await client.request(method, base_url.rstrip("/") + path, **kwargs)
        except httpx.HTTPError as exc:
            self._indices_ready = False
            self._vector_index_ready = False
            raise SearchBackendError(f"OpenSearch request failed: {exc}") from exc

    async def _request_with_index_recovery(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        response = await self._request(method, path, **kwargs)
        if response.status_code != 404:
            return response

        # OpenSearch may have restarted with a fresh volume or an operator may
        # have deleted an index while this process still considered it ready.
        self._indices_ready = False
        await self.ensure_indices()
        return await self._request(method, path, **kwargs)

    async def _request_with_vector_index_recovery(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        response = await self._request(method, path, **kwargs)
        if response.status_code != 404:
            return response

        self._vector_index_ready = False
        await self.ensure_vector_index()
        return await self._request(method, path, **kwargs)

    @staticmethod
    def _documents_mapping() -> dict[str, Any]:
        return {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "document_id": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "canonical_url": {"type": "keyword"},
                    "identity_url": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content_hash": {"type": "keyword"},
                    "fetched_at": {"type": "date"},
                    "content_type": {"type": "keyword"},
                    "http_status": {"type": "integer"},
                    "rendered": {"type": "boolean"},
                    "extractor": {"type": "keyword"},
                    "extraction_quality": {"type": "float"},
                    "chunk_count": {"type": "integer"},
                },
            },
        }

    @staticmethod
    def _chunks_mapping() -> dict[str, Any]:
        return {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "document_id": {"type": "keyword"},
                    "content_hash": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "title": {"type": "text"},
                    "section_path": {"type": "text"},
                    "text": {"type": "text"},
                    "ordinal": {"type": "integer"},
                    "approx_tokens": {"type": "integer"},
                    "fetched_at": {"type": "date"},
                },
            },
        }

    def _vector_chunks_mapping(self) -> dict[str, Any]:
        return {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "_meta": {
                    "embedding_model": self.settings.dense_model_name,
                    "embedding_dimension": self.settings.dense_dimension,
                    "vector_engine": "lucene",
                    "vector_method": "flat",
                    "vector_space": "cosinesimil",
                },
                "dynamic": "strict",
                "properties": {
                    "document_id": {"type": "keyword"},
                    "content_hash": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "title": {"type": "text"},
                    "section_path": {"type": "text"},
                    "text": {"type": "text"},
                    "ordinal": {"type": "integer"},
                    "approx_tokens": {"type": "integer"},
                    "fetched_at": {"type": "date"},
                    "embedding_model": {"type": "keyword"},
                    "embedding_dimension": {"type": "integer"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.settings.dense_dimension,
                        "method": {
                            "name": "flat",
                            "engine": "lucene",
                            "space_type": "cosinesimil",
                        },
                    },
                },
            },
        }

    async def ensure_indices(self, validate: bool = False) -> None:
        if self._indices_ready and not validate:
            return

        async with self._indices_lock:
            if self._indices_ready and not validate:
                return

            definitions = (
                (self.settings.opensearch_documents_index, self._documents_mapping()),
                (self.settings.opensearch_chunks_index, self._chunks_mapping()),
            )
            for index_name, mapping in definitions:
                response = await self._request("HEAD", f"/{index_name}")
                if response.status_code == 404:
                    response = await self._request("PUT", f"/{index_name}", json=mapping)
                    if response.status_code not in {200, 201}:
                        raise SearchBackendError(
                            f"Unable to create OpenSearch index {index_name}: "
                            f"HTTP {response.status_code}"
                        )
                elif response.status_code >= 400:
                    raise SearchBackendError(
                        f"Unable to inspect OpenSearch index {index_name}: "
                        f"HTTP {response.status_code}"
                    )

            self._indices_ready = True

    def _validate_vector_mapping_payload(self, payload: Any) -> None:
        index_name = self.settings.opensearch_vector_chunks_index
        if not isinstance(payload, dict):
            raise SearchBackendError("OpenSearch vector mapping returned invalid JSON")
        index_payload = payload.get(index_name)
        if not isinstance(index_payload, dict):
            raise SearchBackendError(
                "OpenSearch vector mapping is missing the configured index"
            )
        mappings = index_payload.get("mappings")
        if not isinstance(mappings, dict):
            raise SearchBackendError("OpenSearch vector mapping is missing mappings")
        meta = mappings.get("_meta")
        properties = mappings.get("properties")
        if not isinstance(meta, dict) or not isinstance(properties, dict):
            raise SearchBackendError(
                "OpenSearch vector mapping is missing provenance metadata"
            )

        expected_meta = {
            "embedding_model": self.settings.dense_model_name,
            "embedding_dimension": self.settings.dense_dimension,
            "vector_engine": "lucene",
            "vector_method": "flat",
            "vector_space": "cosinesimil",
        }
        for key, expected in expected_meta.items():
            if meta.get(key) != expected:
                raise SearchBackendError(
                    f"OpenSearch vector mapping mismatch for {key}: "
                    f"got {meta.get(key)!r}, expected {expected!r}"
                )

        embedding = properties.get("embedding")
        if not isinstance(embedding, dict):
            raise SearchBackendError("OpenSearch vector mapping has no embedding field")
        if embedding.get("type") != "knn_vector":
            raise SearchBackendError("OpenSearch embedding field is not knn_vector")
        if embedding.get("dimension") != self.settings.dense_dimension:
            raise SearchBackendError(
                "OpenSearch embedding field dimension does not match configured model"
            )
        method = embedding.get("method")
        if not isinstance(method, dict):
            raise SearchBackendError("OpenSearch embedding field has no vector method")
        expected_method = {
            "name": "flat",
            "engine": "lucene",
            "space_type": "cosinesimil",
        }
        for key, expected in expected_method.items():
            if method.get(key) != expected:
                raise SearchBackendError(
                    f"OpenSearch vector method mismatch for {key}: "
                    f"got {method.get(key)!r}, expected {expected!r}"
                )

    async def ensure_vector_index(self, validate: bool = False) -> None:
        if self._vector_index_ready and not validate:
            return

        async with self._vector_index_lock:
            if self._vector_index_ready and not validate:
                return

            index_name = self.settings.opensearch_vector_chunks_index
            response = await self._request("HEAD", f"/{index_name}")
            if response.status_code == 404:
                response = await self._request(
                    "PUT",
                    f"/{index_name}",
                    json=self._vector_chunks_mapping(),
                )
                if response.status_code not in {200, 201}:
                    raise SearchBackendError(
                        f"Unable to create OpenSearch vector index {index_name}: "
                        f"HTTP {response.status_code}"
                    )
            elif response.status_code >= 400:
                raise SearchBackendError(
                    f"Unable to inspect OpenSearch vector index {index_name}: "
                    f"HTTP {response.status_code}"
                )

            response = await self._request("GET", f"/{index_name}/_mapping")
            if response.status_code >= 400:
                raise SearchBackendError(
                    f"Unable to validate OpenSearch vector index {index_name}: "
                    f"HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise SearchBackendError("OpenSearch vector mapping returned invalid JSON") from exc
            self._validate_vector_mapping_payload(payload)
            self._vector_index_ready = True

    def _validated_vector(self, vector: Sequence[float]) -> list[float]:
        try:
            normalized = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise SearchBackendError("vector contains a non-numeric value") from exc
        if len(normalized) != self.settings.dense_dimension:
            raise SearchBackendError(
                f"vector dimension mismatch: got {len(normalized)}, "
                f"expected {self.settings.dense_dimension}"
            )
        if not all(math.isfinite(value) for value in normalized):
            raise SearchBackendError("vector contains a non-finite value")
        if not any(value != 0.0 for value in normalized):
            raise SearchBackendError("vector must not be all zeros")
        return normalized

    async def index_document(
        self,
        fetched: FetchResult,
        extraction: Extraction,
        chunks: list[Chunk],
        content_hash: str,
    ) -> tuple[str, int]:
        canonical_url = normalize_url(extraction.canonical_url or fetched.final_url)
        identity_url = canonical_identity_url(fetched.final_url, canonical_url)
        doc_id = document_id(identity_url)

        # Bulk replacement plus stale-chunk cleanup is one logical operation.
        # Serialize it per document so concurrent refreshes cannot delete each
        # other's newly written chunks. Different documents still index concurrently.
        async with self._document_lock(doc_id):
            return await self._index_document_locked(
                fetched=fetched,
                extraction=extraction,
                chunks=chunks,
                content_hash=content_hash,
                canonical_url=canonical_url,
                identity_url=identity_url,
                doc_id=doc_id,
            )

    async def _index_document_locked(
        self,
        fetched: FetchResult,
        extraction: Extraction,
        chunks: list[Chunk],
        content_hash: str,
        canonical_url: str,
        identity_url: str,
        doc_id: str,
    ) -> tuple[str, int]:
        # Validate on writes: OpenSearch's bulk API can otherwise auto-create a
        # deleted index with dynamic mappings before a 404 can be observed.
        await self.ensure_indices(validate=True)

        document_source = {
            "document_id": doc_id,
            "url": fetched.final_url,
            "canonical_url": canonical_url,
            "identity_url": identity_url,
            "title": extraction.title,
            "content_hash": content_hash,
            "fetched_at": fetched.fetched_at,
            "content_type": fetched.content_type,
            "http_status": fetched.status_code,
            "rendered": extraction.rendered,
            "extractor": extraction.extractor,
            "extraction_quality": extraction.quality,
            "chunk_count": len(chunks),
        }

        lines: list[str] = [
            json.dumps(
                {
                    "index": {
                        "_index": self.settings.opensearch_documents_index,
                        "_id": doc_id,
                    }
                },
                separators=(",", ":"),
            ),
            json.dumps(document_source, separators=(",", ":"), ensure_ascii=False),
        ]

        for chunk in chunks:
            chunk_id = f"{doc_id}:{content_hash[:16]}:{chunk.ordinal}"
            chunk_source = {
                "document_id": doc_id,
                "content_hash": content_hash,
                "url": fetched.final_url,
                "title": extraction.title,
                "section_path": " > ".join(part for part in chunk.section_path if part),
                "text": chunk.text,
                "ordinal": chunk.ordinal,
                "approx_tokens": chunk.approx_tokens,
                "fetched_at": fetched.fetched_at,
            }
            lines.extend(
                [
                    json.dumps(
                        {
                            "index": {
                                "_index": self.settings.opensearch_chunks_index,
                                "_id": chunk_id,
                            }
                        },
                        separators=(",", ":"),
                    ),
                    json.dumps(chunk_source, separators=(",", ":"), ensure_ascii=False),
                ]
            )

        bulk_payload = "\n".join(lines) + "\n"
        response = await self._request_with_index_recovery(
            "POST",
            "/_bulk?refresh=wait_for",
            content=bulk_payload.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        if response.status_code >= 400:
            raise SearchBackendError(f"OpenSearch bulk index failed: HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise SearchBackendError("OpenSearch bulk index returned invalid JSON") from exc
        if not isinstance(body, dict) or body.get("errors") is True:
            raise SearchBackendError("OpenSearch bulk index reported item errors")

        stale_query = {
            "query": {
                "bool": {
                    "filter": [{"term": {"document_id": doc_id}}],
                    "must_not": [{"term": {"content_hash": content_hash}}],
                }
            }
        }
        response = await self._request(
            "POST",
            f"/{self.settings.opensearch_chunks_index}/_delete_by_query"
            "?refresh=true&conflicts=proceed",
            json=stale_query,
        )
        if response.status_code >= 400:
            raise SearchBackendError(
                f"OpenSearch stale-chunk cleanup failed: HTTP {response.status_code}"
            )

        return doc_id, len(chunks)

    async def index_vector_document(
        self,
        fetched: FetchResult,
        extraction: Extraction,
        chunks: list[Chunk],
        content_hash: str,
        vectors: Sequence[Sequence[float]],
    ) -> tuple[str, int]:
        if len(vectors) != len(chunks):
            raise SearchBackendError(
                f"vector count mismatch: got {len(vectors)}, expected {len(chunks)}"
            )
        validated_vectors = [self._validated_vector(vector) for vector in vectors]
        canonical_url = normalize_url(extraction.canonical_url or fetched.final_url)
        identity_url = canonical_identity_url(fetched.final_url, canonical_url)
        doc_id = document_id(identity_url)

        async with self._document_lock(doc_id):
            await self.ensure_vector_index(validate=True)
            lines: list[str] = []
            for chunk, vector in zip(chunks, validated_vectors, strict=True):
                chunk_id = f"{doc_id}:{content_hash[:16]}:{chunk.ordinal}"
                source = {
                    "document_id": doc_id,
                    "content_hash": content_hash,
                    "url": fetched.final_url,
                    "title": extraction.title,
                    "section_path": " > ".join(part for part in chunk.section_path if part),
                    "text": chunk.text,
                    "ordinal": chunk.ordinal,
                    "approx_tokens": chunk.approx_tokens,
                    "fetched_at": fetched.fetched_at,
                    "embedding_model": self.settings.dense_model_name,
                    "embedding_dimension": self.settings.dense_dimension,
                    "embedding": vector,
                }
                lines.extend(
                    [
                        json.dumps(
                            {
                                "index": {
                                    "_index": self.settings.opensearch_vector_chunks_index,
                                    "_id": chunk_id,
                                }
                            },
                            separators=(",", ":"),
                        ),
                        json.dumps(source, separators=(",", ":"), ensure_ascii=False),
                    ]
                )

            if lines:
                response = await self._request_with_vector_index_recovery(
                    "POST",
                    "/_bulk?refresh=wait_for",
                    content=("\n".join(lines) + "\n").encode("utf-8"),
                    headers={"Content-Type": "application/x-ndjson"},
                )
                if response.status_code >= 400:
                    raise SearchBackendError(
                        f"OpenSearch vector bulk index failed: HTTP {response.status_code}"
                    )
                try:
                    body = response.json()
                except ValueError as exc:
                    raise SearchBackendError(
                        "OpenSearch vector bulk index returned invalid JSON"
                    ) from exc
                if not isinstance(body, dict) or body.get("errors") is True:
                    raise SearchBackendError("OpenSearch vector bulk index reported item errors")

                # The bulk API can auto-create an index if it disappears after the
                # pre-write validation. Revalidate after the write so a race cannot
                # silently leave dynamic or incompatible vector mappings behind.
                await self.ensure_vector_index(validate=True)

            stale_query = {
                "query": {
                    "bool": {
                        "filter": [{"term": {"document_id": doc_id}}],
                        "must_not": [{"term": {"content_hash": content_hash}}],
                    }
                }
            }
            response = await self._request(
                "POST",
                f"/{self.settings.opensearch_vector_chunks_index}/_delete_by_query"
                "?refresh=true&conflicts=proceed",
                json=stale_query,
            )
            if response.status_code >= 400:
                raise SearchBackendError(
                    f"OpenSearch stale-vector cleanup failed: HTTP {response.status_code}"
                )

        return doc_id, len(chunks)

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        await self.ensure_indices()
        body = {
            "size": limit,
            "track_total_hits": False,
            "_source": [
                "document_id",
                "url",
                "title",
                "section_path",
                "text",
                "ordinal",
                "fetched_at",
            ],
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^4", "section_path^2", "text"],
                                "type": "best_fields",
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^7", "section_path^4", "text^2"],
                                "type": "phrase",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
            "collapse": {"field": "document_id"},
        }
        response = await self._request_with_index_recovery(
            "POST",
            f"/{self.settings.opensearch_chunks_index}/_search",
            json=body,
        )
        if response.status_code >= 400:
            raise SearchBackendError(f"OpenSearch search failed: HTTP {response.status_code}")

        try:
            payload = response.json()
            hits = payload["hits"]["hits"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SearchBackendError("OpenSearch search returned an invalid response") from exc
        if not isinstance(hits, list):
            raise SearchBackendError("OpenSearch search returned invalid hits")

        return self._format_hits(hits)

    async def dense_search(
        self,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        vector = self._validated_vector(query_vector)
        await self.ensure_vector_index()
        candidate_count = min(10_000, max(100, limit * 50))
        body = {
            "size": limit,
            "track_total_hits": False,
            "_source": [
                "document_id",
                "url",
                "title",
                "section_path",
                "text",
                "ordinal",
                "fetched_at",
                "embedding_model",
                "embedding_dimension",
            ],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": candidate_count,
                    }
                }
            },
            "collapse": {"field": "document_id"},
        }
        response = await self._request_with_vector_index_recovery(
            "POST",
            f"/{self.settings.opensearch_vector_chunks_index}/_search",
            json=body,
        )
        if response.status_code >= 400:
            raise SearchBackendError(
                f"OpenSearch dense search failed: HTTP {response.status_code}"
            )
        try:
            payload = response.json()
            hits = payload["hits"]["hits"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SearchBackendError("OpenSearch dense search returned an invalid response") from exc
        if not isinstance(hits, list):
            raise SearchBackendError("OpenSearch dense search returned invalid hits")
        return self._format_hits(hits, vector_metadata=True)

    @staticmethod
    def _format_hits(
        hits: list[Any],
        *,
        vector_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for position, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
                continue
            source = hit["_source"]
            url = source.get("url")
            if not isinstance(url, str) or not url:
                continue
            title = source.get("title") if isinstance(source.get("title"), str) else ""
            text = source.get("text") if isinstance(source.get("text"), str) else ""
            section_path = (
                source.get("section_path") if isinstance(source.get("section_path"), str) else ""
            )
            score = hit.get("_score")
            metadata = {
                "document_id": source.get("document_id"),
                "section_path": section_path,
                "chunk_ordinal": source.get("ordinal"),
                "fetched_at": source.get("fetched_at"),
            }
            if vector_metadata:
                metadata.update(
                    {
                        "embedding_model": source.get("embedding_model"),
                        "embedding_dimension": source.get("embedding_dimension"),
                    }
                )
            results.append(
                {
                    "title": title,
                    "url": url,
                    "description": _compact_text(text),
                    "position": position,
                    "score": float(score) if isinstance(score, (int, float)) else None,
                    "metadata": metadata,
                }
            )
        return results
