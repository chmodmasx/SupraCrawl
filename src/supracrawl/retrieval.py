from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .embeddings import DenseEmbedder, EmbeddingBackendError
from .fusion import reciprocal_rank_fusion
from .models import SearchMode
from .search import OpenSearchStore, SearchBackendError


@dataclass(slots=True)
class SearchExecution:
    results: list[dict[str, Any]]
    mode_requested: SearchMode
    mode_used: SearchMode
    degraded: bool = False
    degradation_reason: str | None = None


def _compact_text(text: str, max_chars: int = 600) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 2].rstrip() + " …"


def _result_identity(result: dict[str, Any]) -> str:
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        document_id = metadata.get("document_id")
        if isinstance(document_id, str) and document_id:
            return document_id
    url = result.get("url")
    return url if isinstance(url, str) else ""


class SearchService:
    def __init__(
        self,
        settings: Settings,
        store: OpenSearchStore,
        embedder: DenseEmbedder,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder

    async def search(
        self,
        query: str,
        limit: int,
        *,
        mode: SearchMode | None = None,
    ) -> SearchExecution:
        requested_mode: SearchMode = mode or self.settings.search_mode
        candidate_limit = min(20, max(10, limit))

        # The lexical path is the authoritative backbone. Its failure is never
        # hidden by the optional dense path.
        lexical = await self.store.search(query, candidate_limit)

        if requested_mode == "bm25":
            return SearchExecution(
                results=self._as_bm25(lexical, limit),
                mode_requested="bm25",
                mode_used="bm25",
            )

        if not self.settings.dense_enabled:
            return self._degraded(
                lexical,
                limit,
                "hybrid requested but dense retrieval is disabled",
            )

        try:
            dense = await self._dense_search(query, candidate_limit)
            dense = await self._filter_current_dense_results(dense)
        except (EmbeddingBackendError, SearchBackendError) as exc:
            return self._degraded(
                lexical,
                limit,
                f"hybrid vector path unavailable: {exc}",
            )

        if not dense:
            return self._degraded(
                lexical,
                limit,
                "hybrid vector path returned no current candidates",
            )

        return SearchExecution(
            results=self._fuse(lexical, dense, limit),
            mode_requested="hybrid",
            mode_used="hybrid",
        )

    def _as_bm25(
        self,
        results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for position, raw in enumerate(results[:limit], start=1):
            result = dict(raw)
            metadata = dict(result.get("metadata") or {})
            metadata["retrieval_mode"] = "bm25"
            result["metadata"] = metadata
            result["position"] = position
            formatted.append(result)
        return formatted

    def _degraded(
        self,
        lexical: list[dict[str, Any]],
        limit: int,
        reason: str,
    ) -> SearchExecution:
        return SearchExecution(
            results=self._as_bm25(lexical, limit),
            mode_requested="hybrid",
            mode_used="bm25",
            degraded=True,
            degradation_reason=reason,
        )

    async def _dense_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        vector = await self.embedder.embed_query(query)
        vector = self.store._validated_vector(vector)
        await self.store.ensure_vector_index()

        candidate_count = min(10_000, max(100, limit * 50))
        index_name = self.settings.opensearch_vector_chunks_index
        body = {
            "size": limit,
            "track_total_hits": False,
            "_source": [
                "document_id",
                "content_hash",
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
        response = await self.store._request_with_vector_index_recovery(
            "POST",
            f"/{index_name}/_search",
            json=body,
        )
        if response.status_code >= 400:
            raise SearchBackendError(
                f"OpenSearch hybrid dense search failed: HTTP {response.status_code}"
            )
        try:
            payload = response.json()
            hits = payload["hits"]["hits"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SearchBackendError(
                "OpenSearch hybrid dense search returned an invalid response"
            ) from exc
        if not isinstance(hits, list):
            raise SearchBackendError("OpenSearch hybrid dense search returned invalid hits")

        results: list[dict[str, Any]] = []
        for position, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict):
                continue
            source = hit.get("_source")
            if not isinstance(source, dict):
                continue
            document_id = source.get("document_id")
            content_hash = source.get("content_hash")
            url = source.get("url")
            if not isinstance(document_id, str) or not document_id:
                continue
            if not isinstance(content_hash, str) or not content_hash:
                continue
            if not isinstance(url, str) or not url:
                continue

            title = source.get("title") if isinstance(source.get("title"), str) else ""
            text = source.get("text") if isinstance(source.get("text"), str) else ""
            section_path = (
                source.get("section_path")
                if isinstance(source.get("section_path"), str)
                else ""
            )
            score = hit.get("_score")
            results.append(
                {
                    "title": title,
                    "url": url,
                    "description": _compact_text(text),
                    "position": position,
                    "score": float(score) if isinstance(score, (int, float)) else None,
                    "metadata": {
                        "document_id": document_id,
                        "content_hash": content_hash,
                        "section_path": section_path,
                        "chunk_ordinal": source.get("ordinal"),
                        "fetched_at": source.get("fetched_at"),
                        "embedding_model": source.get("embedding_model"),
                        "embedding_dimension": source.get("embedding_dimension"),
                    },
                }
            )
        return results

    async def _filter_current_dense_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expected_hashes: dict[str, str] = {}
        for result in results:
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            document_id = metadata.get("document_id")
            content_hash = metadata.get("content_hash")
            if isinstance(document_id, str) and isinstance(content_hash, str):
                expected_hashes[document_id] = content_hash

        if not expected_hashes:
            return []

        index_name = self.settings.opensearch_documents_index
        response = await self.store._request_with_index_recovery(
            "POST",
            f"/{index_name}/_mget",
            json={"ids": list(expected_hashes)},
        )
        if response.status_code >= 400:
            raise SearchBackendError(
                f"OpenSearch current-content validation failed: HTTP {response.status_code}"
            )
        try:
            payload = response.json()
            documents = payload["docs"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SearchBackendError(
                "OpenSearch current-content validation returned an invalid response"
            ) from exc
        if not isinstance(documents, list):
            raise SearchBackendError(
                "OpenSearch current-content validation returned invalid documents"
            )

        current_hashes: dict[str, str] = {}
        for document in documents:
            if not isinstance(document, dict) or document.get("found") is not True:
                continue
            document_id = document.get("_id")
            source = document.get("_source")
            if not isinstance(document_id, str) or not isinstance(source, dict):
                continue
            content_hash = source.get("content_hash")
            if isinstance(content_hash, str):
                current_hashes[document_id] = content_hash

        current: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            document_id = metadata.get("document_id")
            content_hash = metadata.get("content_hash")
            if not isinstance(document_id, str) or not isinstance(content_hash, str):
                continue
            if current_hashes.get(document_id) == content_hash:
                current.append(result)
        return current

    def _fuse(
        self,
        lexical: list[dict[str, Any]],
        dense: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        lexical_ids = [identity for result in lexical if (identity := _result_identity(result))]
        dense_ids = [identity for result in dense if (identity := _result_identity(result))]
        fused_ids = reciprocal_rank_fusion(
            [lexical_ids, dense_ids],
            k=self.settings.hybrid_rrf_k,
            limit=limit,
        )

        lexical_by_id = {_result_identity(result): result for result in lexical}
        dense_by_id = {_result_identity(result): result for result in dense}
        lexical_rank = {document_id: rank for rank, document_id in enumerate(lexical_ids, 1)}
        dense_rank = {document_id: rank for rank, document_id in enumerate(dense_ids, 1)}

        fused: list[dict[str, Any]] = []
        for position, document_id in enumerate(fused_ids, start=1):
            source = lexical_by_id.get(document_id) or dense_by_id.get(document_id)
            if source is None:
                continue
            result = dict(source)
            metadata = dict(result.get("metadata") or {})
            lexical_position = lexical_rank.get(document_id)
            dense_position = dense_rank.get(document_id)
            metadata.update(
                {
                    "retrieval_mode": "hybrid",
                    "lexical_rank": lexical_position,
                    "dense_rank": dense_position,
                }
            )
            rrf_score = 0.0
            if lexical_position is not None:
                rrf_score += 1.0 / (self.settings.hybrid_rrf_k + lexical_position)
            if dense_position is not None:
                rrf_score += 1.0 / (self.settings.hybrid_rrf_k + dense_position)
            result["metadata"] = metadata
            result["position"] = position
            result["score"] = rrf_score
            fused.append(result)
        return fused
