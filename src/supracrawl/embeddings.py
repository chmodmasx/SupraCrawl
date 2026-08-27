from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any


class EmbeddingBackendError(RuntimeError):
    pass


def build_passage_text(
    *,
    title: str,
    section_path: Sequence[str] | str,
    text: str,
    prefix: str = "passage: ",
) -> str:
    if isinstance(section_path, str):
        section = section_path.strip()
    else:
        section = " > ".join(part.strip() for part in section_path if part.strip())

    parts = [title.strip()]
    if section:
        parts.append(section)
    parts.append(text.strip())
    body = "\n".join(part for part in parts if part)
    return prefix + body


class DenseEmbedder:
    """Lazy local embedding runtime used only by explicit vector paths.

    FastEmbed is imported lazily so the normal BM25/extraction runtime does not
    acquire the optional ONNX dependency. Model inference is moved off the event
    loop because ONNX execution is synchronous CPU work.
    """

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        if not model_name.strip():
            raise ValueError("embedding model name must not be empty")
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.model_name = model_name
        self.dimension = dimension
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    def _load_model_sync(self) -> Any:
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
        except ImportError as exc:
            raise EmbeddingBackendError(
                "dense retrieval requires the optional 'hybrid-eval' dependencies"
            ) from exc

        try:
            supported = {item["model"] for item in TextEmbedding.list_supported_models()}
            if self.model_name not in supported:
                TextEmbedding.add_custom_model(
                    model=self.model_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=self.model_name),
                    dim=self.dimension,
                    model_file="onnx/model.onnx",
                )
            return TextEmbedding(model_name=self.model_name)
        except Exception as exc:
            raise EmbeddingBackendError(
                f"unable to load embedding model {self.model_name}: {exc}"
            ) from exc

    def _validate_vector(self, raw_vector: Any) -> list[float]:
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingBackendError("embedding runtime returned an invalid vector") from exc
        if len(vector) != self.dimension:
            raise EmbeddingBackendError(
                f"embedding dimension mismatch: got {len(vector)}, expected {self.dimension}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingBackendError("embedding vector contains non-finite values")
        if not any(value != 0.0 for value in vector):
            raise EmbeddingBackendError("embedding vector must not be all zeros")
        return vector

    def _embed_sync(self, model: Any, texts: list[str], batch_size: int) -> list[list[float]]:
        try:
            raw_vectors = list(model.embed(texts, batch_size=batch_size))
        except Exception as exc:
            raise EmbeddingBackendError(f"embedding inference failed: {exc}") from exc
        if len(raw_vectors) != len(texts):
            raise EmbeddingBackendError(
                f"embedding count mismatch: got {len(raw_vectors)}, expected {len(texts)}"
            )
        return [self._validate_vector(vector) for vector in raw_vectors]

    async def embed_query(self, query: str) -> list[float]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        model = await self._ensure_model()
        vectors = await asyncio.to_thread(
            self._embed_sync,
            model,
            [self.query_prefix + query],
            1,
        )
        return vectors[0]

    async def embed_passages(
        self,
        passages: Sequence[str],
        *,
        batch_size: int = 16,
    ) -> list[list[float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        texts = [passage for passage in passages if passage.strip()]
        if len(texts) != len(passages):
            raise ValueError("passages must not contain empty strings")
        if not texts:
            return []
        model = await self._ensure_model()
        return await asyncio.to_thread(self._embed_sync, model, texts, batch_size)
