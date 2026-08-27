from __future__ import annotations

import math

import pytest

from supracrawl.embeddings import DenseEmbedder, EmbeddingBackendError, build_passage_text


class FakeEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.seen_texts: list[str] = []
        self.seen_batch_sizes: list[int] = []

    def embed(self, texts, batch_size):
        self.seen_texts.extend(texts)
        self.seen_batch_sizes.append(batch_size)
        return iter(self.vectors)


def test_build_passage_text_preserves_title_heading_and_body() -> None:
    assert build_passage_text(
        title="Vector retrieval",
        section_path=("Search", "Dense"),
        text="Exact cosine lookup.",
    ) == "passage: Vector retrieval\nSearch > Dense\nExact cosine lookup."


@pytest.mark.asyncio
async def test_embed_query_applies_e5_prefix_and_validates_dimension() -> None:
    embedder = DenseEmbedder(model_name="test/model", dimension=3)
    model = FakeEmbeddingModel([[0.1, 0.2, 0.3]])
    embedder._model = model

    vector = await embedder.embed_query("semantic search")

    assert vector == pytest.approx([0.1, 0.2, 0.3])
    assert model.seen_texts == ["query: semantic search"]
    assert model.seen_batch_sizes == [1]


@pytest.mark.asyncio
async def test_embed_passages_rejects_empty_input_item() -> None:
    embedder = DenseEmbedder(model_name="test/model", dimension=3)
    embedder._model = FakeEmbeddingModel([[0.1, 0.2, 0.3]])

    with pytest.raises(ValueError, match="empty strings"):
        await embedder.embed_passages(["valid", "   "])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.1, 0.2], "dimension mismatch"),
        ([0.1, math.nan, 0.3], "non-finite"),
        ([0.0, 0.0, 0.0], "all zeros"),
    ],
)
async def test_embedding_runtime_rejects_invalid_vectors(vector, message) -> None:
    embedder = DenseEmbedder(model_name="test/model", dimension=3)
    embedder._model = FakeEmbeddingModel([vector])

    with pytest.raises(EmbeddingBackendError, match=message):
        await embedder.embed_query("query")
