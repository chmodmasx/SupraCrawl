from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .models import SearchMode
from .retrieval import SearchExecution, SearchService

RERANKER_MODEL_REPO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_REVISION = "1427fd6"
RERANKER_RUNTIME_ALIAS = "supracrawl/mmarco-mMiniLMv2-L12-H384-v1-qint8-avx2"
RERANKER_MODEL_FILE = "onnx/model_quint8_avx2.onnx"
RERANKER_MODEL_FILE_SHA256 = (
    "6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b"
)
RERANKER_FASTEMBED_VERSION = "0.8.0"
RERANKER_CANDIDATE_POOL_SIZE = 10
RERANKER_PROTECTED_TOP_K = 5
RERANKER_STRATEGY = "score_desc_within_frozen_top5_and_tail"
RERANKER_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    RERANKER_MODEL_FILE,
)


class RerankerBackendError(RuntimeError):
    pass


class _CrossEncoder(Protocol):
    def rerank(self, query: str, documents: list[str]) -> Any: ...


class _Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class ControlledSearchExecution:
    results: list[dict[str, Any]]
    mode_requested: SearchMode
    mode_used: SearchMode
    degraded: bool
    degradation_reason: str | None
    reranker_enabled: bool
    reranker_used: bool = False
    reranker_degraded: bool = False
    reranker_degradation_reason: str | None = None


def _candidate_text(result: dict[str, Any]) -> str:
    title = result.get("title")
    description = result.get("description")
    parts = [
        value.strip()
        for value in (title, description)
        if isinstance(value, str) and value.strip()
    ]
    if not parts:
        raise RerankerBackendError("hybrid candidate has no title or description")
    return "\n".join(parts)


def _top5_preserving_indices(scores: list[float], protected_top_k: int) -> list[int]:
    if not scores:
        return []
    if protected_top_k <= 0:
        raise ValueError("protected_top_k must be positive")

    split = min(protected_top_k, len(scores))
    score_order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    protected = set(range(split))
    top = [index for index in score_order if index in protected]
    tail = [index for index in score_order if index not in protected]
    ranking = top + tail

    if set(ranking[:split]) != protected:
        raise RerankerBackendError("reranker violated protected top-k membership")
    if set(ranking[split:]) != set(range(split, len(scores))):
        raise RerankerBackendError("reranker violated frozen tail membership")
    return ranking


class LocalCrossEncoderReranker:
    """Lazy exact Candidate 2 runtime.

    Model identity and ranking constants are fixed because Phase 3F certified
    exactly this candidate. Only activation is configurable.
    """

    def __init__(self) -> None:
        self._model: _CrossEncoder | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self) -> _CrossEncoder:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_model_sync(self) -> _CrossEncoder:
        try:
            from fastembed.common.model_description import ModelSource
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RerankerBackendError(
                "reranker requires the optional 'reranker' dependencies"
            ) from exc

        actual_fastembed = importlib.metadata.version("fastembed")
        if actual_fastembed != RERANKER_FASTEMBED_VERSION:
            raise RerankerBackendError(
                "reranker runtime mismatch: "
                f"fastembed {actual_fastembed}, expected {RERANKER_FASTEMBED_VERSION}"
            )

        try:
            model_dir = Path(
                snapshot_download(
                    repo_id=RERANKER_MODEL_REPO,
                    revision=RERANKER_REVISION,
                    allow_patterns=list(RERANKER_REQUIRED_FILES),
                )
            )
        except Exception as exc:
            raise RerankerBackendError(
                f"unable to download frozen reranker snapshot: {exc}"
            ) from exc

        model_path = model_dir / RERANKER_MODEL_FILE
        if not model_path.is_file():
            raise RerankerBackendError(f"reranker ONNX file is missing: {model_path}")
        actual_sha256 = self._sha256(model_path)
        if actual_sha256 != RERANKER_MODEL_FILE_SHA256:
            raise RerankerBackendError(
                "reranker ONNX checksum mismatch: "
                f"expected {RERANKER_MODEL_FILE_SHA256}, got {actual_sha256}"
            )

        try:
            supported = {
                item["model"] for item in TextCrossEncoder.list_supported_models()
            }
            if RERANKER_RUNTIME_ALIAS not in supported:
                TextCrossEncoder.add_custom_model(
                    model=RERANKER_RUNTIME_ALIAS,
                    model_file=RERANKER_MODEL_FILE,
                    sources=ModelSource(hf=RERANKER_MODEL_REPO),
                )
            return TextCrossEncoder(
                model_name=RERANKER_RUNTIME_ALIAS,
                specific_model_path=str(model_dir),
            )
        except Exception as exc:
            raise RerankerBackendError(
                f"unable to load frozen reranker model: {exc}"
            ) from exc

    @staticmethod
    def _score_sync(
        model: _CrossEncoder,
        query: str,
        documents: list[str],
    ) -> list[float]:
        try:
            scores = [float(score) for score in model.rerank(query, documents)]
        except Exception as exc:
            raise RerankerBackendError(f"reranker inference failed: {exc}") from exc
        if len(scores) != len(documents):
            raise RerankerBackendError(
                f"reranker score count mismatch: got {len(scores)}, expected {len(documents)}"
            )
        if not all(math.isfinite(score) for score in scores):
            raise RerankerBackendError("reranker returned non-finite scores")
        return scores

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not results:
            return []

        candidate_count = min(RERANKER_CANDIDATE_POOL_SIZE, len(results))
        candidate_results = results[:candidate_count]
        documents = [_candidate_text(result) for result in candidate_results]
        model = await self._ensure_model()
        scores = await asyncio.to_thread(self._score_sync, model, query, documents)
        order = _top5_preserving_indices(scores, RERANKER_PROTECTED_TOP_K)

        reranked: list[dict[str, Any]] = []
        for candidate_index in order:
            raw = candidate_results[candidate_index]
            result = dict(raw)
            metadata = dict(result.get("metadata") or {})
            metadata.update(
                {
                    "reranker_model": RERANKER_MODEL_REPO,
                    "reranker_revision": RERANKER_REVISION,
                    "reranker_strategy": RERANKER_STRATEGY,
                    "reranker_score": scores[candidate_index],
                    "first_stage_position": candidate_index + 1,
                }
            )
            result["metadata"] = metadata
            reranked.append(result)

        reranked.extend(dict(result) for result in results[candidate_count:])
        for position, result in enumerate(reranked, start=1):
            result["position"] = position
        return reranked


class ControlledRerankingSearchService:
    """Optional reranking layer over the certified SearchService."""

    def __init__(
        self,
        settings: Settings,
        base_service: SearchService,
        reranker: _Reranker | None,
    ) -> None:
        self.settings = settings
        self.base_service = base_service
        self.reranker = reranker

    @staticmethod
    def _wrap(
        execution: SearchExecution,
        results: list[dict[str, Any]],
        *,
        enabled: bool,
        used: bool = False,
        reranker_degraded: bool = False,
        reason: str | None = None,
    ) -> ControlledSearchExecution:
        return ControlledSearchExecution(
            results=results,
            mode_requested=execution.mode_requested,
            mode_used=execution.mode_used,
            degraded=execution.degraded,
            degradation_reason=execution.degradation_reason,
            reranker_enabled=enabled,
            reranker_used=used,
            reranker_degraded=reranker_degraded,
            reranker_degradation_reason=reason,
        )

    async def search(
        self,
        query: str,
        limit: int,
        *,
        mode: SearchMode | None = None,
    ) -> ControlledSearchExecution:
        if not self.settings.reranker_enabled:
            execution = await self.base_service.search(query, limit, mode=mode)
            return self._wrap(execution, execution.results, enabled=False)

        first_stage_limit = max(limit, RERANKER_CANDIDATE_POOL_SIZE)
        execution = await self.base_service.search(query, first_stage_limit, mode=mode)
        first_stage = execution.results

        if execution.mode_used != "hybrid":
            if execution.mode_requested == "bm25":
                return self._wrap(
                    execution,
                    first_stage[:limit],
                    enabled=True,
                )
            return self._wrap(
                execution,
                first_stage[:limit],
                enabled=True,
                reranker_degraded=True,
                reason="reranker not attempted because hybrid retrieval was unavailable",
            )

        if self.reranker is None:
            return self._wrap(
                execution,
                first_stage[:limit],
                enabled=True,
                reranker_degraded=True,
                reason="reranker enabled but runtime is not configured",
            )

        try:
            reranked = await self.reranker.rerank(query, first_stage)
        except (RerankerBackendError, ValueError) as exc:
            return self._wrap(
                execution,
                first_stage[:limit],
                enabled=True,
                reranker_degraded=True,
                reason=f"reranker unavailable: {exc}",
            )

        return self._wrap(
            execution,
            reranked[:limit],
            enabled=True,
            used=True,
        )
