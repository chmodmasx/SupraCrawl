from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import supracrawl.app as app_module
from supracrawl.config import Settings
from supracrawl.models import CrawlRequest
from supracrawl.scheduler import RefreshScheduler


class RecordingCrawler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.called = asyncio.Event()

    async def crawl(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        self.called.set()
        return []


def _enabled_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "crawl_scheduler_enabled": True,
        "crawl_scheduler_seeds": ["https://example.com/root"],
        "crawl_scheduler_interval_s": 120,
        "crawl_scheduler_max_pages": 7,
        "crawl_scheduler_max_depth": 2,
        "crawl_scheduler_same_origin": False,
        "crawl_scheduler_refresh_after_s": 3600,
        "crawl_scheduler_conditional_revalidate_leaves": True,
    }
    values.update(overrides)
    return Settings(**values)


def test_phase4d_policy_matches_preregistered_scheduler_contract() -> None:
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation/phase4d_refresh_scheduler_policy.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert policy["base_certified_sha"] == "bf52c6e73e4e4d125e36729736815214df05371f"
    assert policy["architecture_contract"]["execution_model"] == (
        "one process-local asyncio task"
    )
    assert policy["architecture_contract"]["first_run"] == "immediate after enabled startup"
    assert policy["architecture_contract"]["overlap_allowed"] is False
    assert policy["configuration_contract"]["enabled"]["default"] is False
    assert policy["configuration_contract"]["interval_s"] == {
        "field": "crawl_scheduler_interval_s",
        "default": 21600,
        "minimum": 60,
    }
    assert policy["constraints"]["crawler_algorithm_unchanged"] is True
    assert policy["constraints"]["no_existing_workflow_change"] is True


def test_phase4d_settings_default_to_disabled() -> None:
    settings = Settings()

    assert settings.crawl_scheduler_enabled is False
    assert settings.crawl_scheduler_seeds == []
    assert settings.crawl_scheduler_interval_s == 21_600
    assert settings.crawl_scheduler_refresh_after_s == 21_600
    assert settings.crawl_scheduler_conditional_revalidate_leaves is True


def test_phase4d_enabled_settings_require_seed() -> None:
    with pytest.raises(ValidationError, match="crawl_scheduler_seeds"):
        Settings(crawl_scheduler_enabled=True)


def test_phase4d_manual_crawl_request_contract_is_unchanged() -> None:
    assert list(CrawlRequest.model_fields) == [
        "seeds",
        "max_pages",
        "max_depth",
        "same_origin",
        "refresh_after_s",
        "conditional_revalidate_leaves",
    ]


@pytest.mark.asyncio
async def test_phase4d_disabled_scheduler_creates_no_task_or_crawl() -> None:
    crawler = RecordingCrawler()
    scheduler = RefreshScheduler(crawler, Settings())

    scheduler.start()
    await asyncio.sleep(0)

    assert scheduler.running is False
    assert crawler.calls == []
    await scheduler.stop()


@pytest.mark.asyncio
async def test_phase4d_enabled_scheduler_runs_immediately_with_exact_policy() -> None:
    crawler = RecordingCrawler()
    sleep_started = asyncio.Event()

    async def blocked_sleep(_delay: float) -> None:
        sleep_started.set()
        await asyncio.Event().wait()

    settings = _enabled_settings()
    scheduler = RefreshScheduler(crawler, settings, sleep=blocked_sleep)

    scheduler.start()
    await asyncio.wait_for(crawler.called.wait(), timeout=1)
    await asyncio.wait_for(sleep_started.wait(), timeout=1)

    assert crawler.calls == [
        {
            "seeds": ["https://example.com/root"],
            "max_pages": 7,
            "max_depth": 2,
            "same_origin": False,
            "refresh_after_s": 3600,
            "conditional_revalidate_leaves": True,
        }
    ]

    await scheduler.stop()
    assert scheduler.running is False


@pytest.mark.asyncio
async def test_phase4d_scheduler_never_overlaps_and_waits_after_completion() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()
    second_started = asyncio.Event()

    class ControlledCrawler:
        def __init__(self) -> None:
            self.calls = 0

        async def crawl(self, **_kwargs: Any) -> list[Any]:
            self.calls += 1
            if self.calls == 1:
                first_started.set()
                await release_first.wait()
                return []
            second_started.set()
            await asyncio.Event().wait()
            return []

    async def controlled_sleep(delay: float) -> None:
        assert delay == 120
        sleep_started.set()
        await release_sleep.wait()

    crawler = ControlledCrawler()
    scheduler = RefreshScheduler(crawler, _enabled_settings(), sleep=controlled_sleep)

    scheduler.start()
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert crawler.calls == 1
    assert sleep_started.is_set() is False

    release_first.set()
    await asyncio.wait_for(sleep_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert crawler.calls == 1

    release_sleep.set()
    await asyncio.wait_for(second_started.wait(), timeout=1)
    assert crawler.calls == 2

    await scheduler.stop()


@pytest.mark.asyncio
async def test_phase4d_cycle_exception_does_not_stop_future_cycles() -> None:
    second_started = asyncio.Event()

    class FlakyCrawler:
        def __init__(self) -> None:
            self.calls = 0

        async def crawl(self, **_kwargs: Any) -> list[Any]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("fixture failure")
            second_started.set()
            await asyncio.Event().wait()
            return []

    sleep_calls: list[float] = []

    async def immediate_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    crawler = FlakyCrawler()
    scheduler = RefreshScheduler(crawler, _enabled_settings(), sleep=immediate_sleep)

    scheduler.start()
    await asyncio.wait_for(second_started.wait(), timeout=1)

    assert crawler.calls == 2
    assert sleep_calls == [120]

    await scheduler.stop()


@pytest.mark.asyncio
async def test_phase4d_stop_cancels_active_cycle_and_prevents_future_cycles() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingCrawler:
        def __init__(self) -> None:
            self.calls = 0

        async def crawl(self, **_kwargs: Any) -> list[Any]:
            self.calls += 1
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return []

    crawler = BlockingCrawler()
    scheduler = RefreshScheduler(crawler, _enabled_settings())

    scheduler.start()
    await asyncio.wait_for(started.wait(), timeout=1)
    await scheduler.stop()

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    await asyncio.sleep(0)
    assert crawler.calls == 1
    assert scheduler.running is False


@pytest.mark.asyncio
async def test_phase4d_fastapi_lifespan_owns_scheduler_task(monkeypatch) -> None:
    events: list[str] = []

    class FakeScheduler:
        def start(self) -> None:
            events.append("scheduler-start")

        async def stop(self) -> None:
            events.append("scheduler-stop")

    class FakeResource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            events.append(self.name)

    monkeypatch.setattr(app_module, "refresh_scheduler", FakeScheduler())
    monkeypatch.setattr(app_module, "cache", FakeResource("cache-close"))
    monkeypatch.setattr(app_module, "search_store", FakeResource("search-store-close"))

    async with app_module.lifespan(app_module.app):
        events.append("inside")

    assert events == [
        "scheduler-start",
        "inside",
        "scheduler-stop",
        "cache-close",
        "search-store-close",
    ]
