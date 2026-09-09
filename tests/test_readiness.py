from __future__ import annotations

import pytest

from supracrawl.config import Settings
from supracrawl.embeddings import EmbeddingBackendError
from supracrawl.readiness import ReadinessChecker
from supracrawl.search import SearchBackendError


class _Store:
    def __init__(self, *, lexical_fail: bool = False, vector_fail: bool = False) -> None:
        self.lexical_fail = lexical_fail
        self.vector_fail = vector_fail
        self.lexical_calls = 0
        self.vector_calls = 0

    async def ensure_indices(self, validate: bool = False) -> None:
        assert validate is True
        self.lexical_calls += 1
        if self.lexical_fail:
            raise SearchBackendError("fixture lexical failure")

    async def ensure_vector_index(self, validate: bool = False) -> None:
        assert validate is True
        self.vector_calls += 1
        if self.vector_fail:
            raise SearchBackendError("fixture vector failure")


class _Dense:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def embed_query(self, _query: str) -> list[float]:
        self.calls += 1
        if self.fail:
            raise EmbeddingBackendError("fixture dense failure")
        return [1.0]


@pytest.mark.asyncio
async def test_hybrid_readiness_validates_all_required_components_once() -> None:
    settings = Settings(search_mode="hybrid", dense_enabled=True, reranker_enabled=False)
    store = _Store()
    dense = _Dense()
    checker = ReadinessChecker(settings, store, dense)

    first = await checker.check()
    second = await checker.check()

    assert first.ready is True
    assert second.ready is True
    assert first.components["lexical"].status == "ready"
    assert first.components["dense"].status == "ready"
    assert first.components["reranker"].status == "disabled"
    assert dense.calls == 1
    assert store.lexical_calls == 2
    assert store.vector_calls == 2


@pytest.mark.asyncio
async def test_bm25_readiness_does_not_require_dense_runtime() -> None:
    settings = Settings(search_mode="bm25", dense_enabled=False)
    store = _Store()
    dense = _Dense(fail=True)
    checker = ReadinessChecker(settings, store, dense)

    state = await checker.check()

    assert state.ready is True
    assert state.components["dense"].status == "not_required"
    assert dense.calls == 0
    assert store.vector_calls == 0


@pytest.mark.asyncio
async def test_hybrid_readiness_rejects_disabled_dense_runtime() -> None:
    settings = Settings(search_mode="hybrid", dense_enabled=False)
    checker = ReadinessChecker(settings, _Store(), _Dense())

    state = await checker.check()

    assert state.ready is False
    assert state.components["dense"].reason == "dense_disabled_for_hybrid"


@pytest.mark.asyncio
async def test_hybrid_readiness_reports_dense_and_vector_failures() -> None:
    dense_state = await ReadinessChecker(
        Settings(search_mode="hybrid", dense_enabled=True),
        _Store(),
        _Dense(fail=True),
    ).check()
    vector_state = await ReadinessChecker(
        Settings(search_mode="hybrid", dense_enabled=True),
        _Store(vector_fail=True),
        _Dense(),
    ).check()

    assert dense_state.ready is False
    assert dense_state.components["dense"].reason == "dense_runtime_unavailable"
    assert vector_state.ready is False
    assert vector_state.components["dense"].reason == (
        "vector_index_unavailable_or_invalid"
    )


@pytest.mark.asyncio
async def test_readiness_reports_lexical_failure() -> None:
    state = await ReadinessChecker(
        Settings(search_mode="bm25"),
        _Store(lexical_fail=True),
        _Dense(),
    ).check()

    assert state.ready is False
    assert state.components["lexical"].reason == (
        "opensearch_unavailable_or_indices_invalid"
    )


@pytest.mark.asyncio
async def test_reranker_readiness_requires_startup_warmup_when_enabled() -> None:
    unsafe = await ReadinessChecker(
        Settings(
            search_mode="bm25",
            reranker_enabled=True,
            reranker_warmup_on_startup=False,
        ),
        _Store(),
        _Dense(),
    ).check()
    safe = await ReadinessChecker(
        Settings(
            search_mode="bm25",
            reranker_enabled=True,
            reranker_warmup_on_startup=True,
        ),
        _Store(),
        _Dense(),
    ).check()

    assert unsafe.ready is False
    assert unsafe.components["reranker"].reason == "reranker_startup_warmup_required"
    assert safe.ready is True
    assert safe.components["reranker"].status == "ready"
