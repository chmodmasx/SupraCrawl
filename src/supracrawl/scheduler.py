from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .config import Settings

logger = logging.getLogger(__name__)


class CrawlRunner(Protocol):
    async def crawl(
        self,
        seeds: list[str],
        max_pages: int,
        max_depth: int,
        same_origin: bool,
        refresh_after_s: int = 0,
        conditional_revalidate_leaves: bool = False,
    ) -> list[Any]: ...


class RefreshScheduler:
    def __init__(
        self,
        crawler: CrawlRunner,
        settings: Settings,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._crawler = crawler
        self._settings = settings
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self._settings.crawl_scheduler_enabled or self.running:
            return
        self._task = asyncio.create_task(
            self._run_loop(),
            name="supracrawl-refresh-scheduler",
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_loop(self) -> None:
        while True:
            try:
                await self._crawler.crawl(
                    seeds=[str(seed) for seed in self._settings.crawl_scheduler_seeds],
                    max_pages=self._settings.crawl_scheduler_max_pages,
                    max_depth=self._settings.crawl_scheduler_max_depth,
                    same_origin=self._settings.crawl_scheduler_same_origin,
                    refresh_after_s=self._settings.crawl_scheduler_refresh_after_s,
                    conditional_revalidate_leaves=(
                        self._settings.crawl_scheduler_conditional_revalidate_leaves
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduled crawl refresh cycle failed")
            await self._sleep(self._settings.crawl_scheduler_interval_s)
