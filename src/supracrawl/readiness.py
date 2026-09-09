from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol

from .config import Settings
from .embeddings import EmbeddingBackendError
from .search import SearchBackendError

READINESS_SCHEMA_VERSION = 1
ReadinessComponentStatus = Literal["ready", "not_ready", "disabled", "not_required"]


@dataclass(frozen=True, slots=True)
class ComponentReadiness:
    status: ReadinessComponentStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReadinessState:
    ready: bool
    components: dict[str, ComponentReadiness]


class _SearchStore(Protocol):
    async def ensure_indices(self, validate: bool = False) -> None: ...

    async def ensure_vector_index(self, validate: bool = False) -> None: ...


class _DenseEmbedder(Protocol):
    async def embed_query(self, query: str) -> list[float]: ...


class ReadinessChecker:
    """Validate the configured serving path without changing search ranking."""

    def __init__(
        self,
        settings: Settings,
        search_store: _SearchStore,
        dense_embedder: _DenseEmbedder,
    ) -> None:
        self.settings = settings
        self.search_store = search_store
        self.dense_embedder = dense_embedder
        self._dense_runtime_ready = False
        self._dense_runtime_lock = asyncio.Lock()

    async def _ensure_dense_runtime(self) -> None:
        if self._dense_runtime_ready:
            return
        async with self._dense_runtime_lock:
            if self._dense_runtime_ready:
                return
            await self.dense_embedder.embed_query("supracrawl readiness")
            self._dense_runtime_ready = True

    async def check(self) -> ReadinessState:
        components: dict[str, ComponentReadiness] = {}

        try:
            await self.search_store.ensure_indices(validate=True)
        except SearchBackendError:
            components["lexical"] = ComponentReadiness(
                status="not_ready",
                reason="opensearch_unavailable_or_indices_invalid",
            )
        else:
            components["lexical"] = ComponentReadiness(status="ready")

        if self.settings.search_mode == "hybrid":
            if not self.settings.dense_enabled:
                components["dense"] = ComponentReadiness(
                    status="not_ready",
                    reason="dense_disabled_for_hybrid",
                )
            else:
                try:
                    await self._ensure_dense_runtime()
                except (EmbeddingBackendError, ValueError):
                    components["dense"] = ComponentReadiness(
                        status="not_ready",
                        reason="dense_runtime_unavailable",
                    )
                else:
                    try:
                        await self.search_store.ensure_vector_index(validate=True)
                    except SearchBackendError:
                        components["dense"] = ComponentReadiness(
                            status="not_ready",
                            reason="vector_index_unavailable_or_invalid",
                        )
                    else:
                        components["dense"] = ComponentReadiness(status="ready")
        else:
            components["dense"] = ComponentReadiness(status="not_required")

        if not self.settings.reranker_enabled:
            components["reranker"] = ComponentReadiness(status="disabled")
        elif not self.settings.reranker_warmup_on_startup:
            components["reranker"] = ComponentReadiness(
                status="not_ready",
                reason="reranker_startup_warmup_required",
            )
        else:
            # App startup awaits reranker warmup before serving requests. If this
            # process is reachable and startup warmup is configured, that load
            # boundary has already succeeded without readiness loading the model.
            components["reranker"] = ComponentReadiness(status="ready")

        return ReadinessState(
            ready=all(component.status != "not_ready" for component in components.values()),
            components=components,
        )
