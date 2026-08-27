from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from selectolax.parser import HTMLParser

from .extractor import Extractor
from .fetcher import FetchError, HttpFetcher
from .indexer import Indexer
from .security import UnsafeUrlError
from .urls import normalize_url


@dataclass(slots=True)
class CrawlOutcome:
    url: str
    depth: int
    indexed: bool
    document_id: str | None = None
    content_hash: str | None = None
    chunks_indexed: int = 0
    error: str | None = None


def _origin_key(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is None:
        if scheme == "http":
            port = 80
        elif scheme == "https":
            port = 443
    return scheme, hostname, port


def discover_links(html: str, base_url: str) -> list[str]:
    tree = HTMLParser(html)
    links: list[str] = []
    seen: set[str] = set()

    for node in tree.css("a"):
        href = node.attributes.get("href")
        if not href:
            continue
        candidate = urljoin(base_url, href.strip())
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in {"http", "https"}:
            continue
        try:
            normalized = normalize_url(candidate)
        except ValueError:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)

    return links


class Crawler:
    def __init__(self, fetcher: HttpFetcher, extractor: Extractor, indexer: Indexer) -> None:
        self.fetcher = fetcher
        self.extractor = extractor
        self.indexer = indexer

    async def crawl(
        self,
        seeds: list[str],
        max_pages: int,
        max_depth: int,
        same_origin: bool,
    ) -> list[CrawlOutcome]:
        queue: deque[tuple[str, int]] = deque()
        allowed_origins: set[tuple[str, str, int | None]] = set()
        for seed in seeds:
            try:
                normalized = normalize_url(seed)
            except ValueError:
                normalized = seed
            queue.append((normalized, 0))
            allowed_origins.add(_origin_key(normalized))

        visited: set[str] = set()
        outcomes: list[CrawlOutcome] = []

        while queue and len(visited) < max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                fetched = await self.fetcher.fetch_html(url)
                extraction = await self.extractor.extract_fetched(fetched)
            except UnsafeUrlError as exc:
                outcomes.append(
                    CrawlOutcome(
                        url=url,
                        depth=depth,
                        indexed=False,
                        error=f"Unsafe URL: {exc}",
                    )
                )
                continue
            except FetchError as exc:
                outcomes.append(
                    CrawlOutcome(
                        url=url,
                        depth=depth,
                        indexed=False,
                        error=f"Fetch failed: {exc}",
                    )
                )
                continue

            indexed = await self.indexer.index_extraction(fetched, extraction)
            outcomes.append(
                CrawlOutcome(
                    url=indexed.url,
                    depth=depth,
                    indexed=indexed.indexed,
                    document_id=indexed.document_id,
                    content_hash=indexed.content_hash,
                    chunks_indexed=indexed.chunks_indexed,
                    error=indexed.error,
                )
            )

            if depth >= max_depth:
                continue

            for link in discover_links(fetched.html, fetched.final_url):
                if same_origin and _origin_key(link) not in allowed_origins:
                    continue
                if link not in visited:
                    queue.append((link, depth + 1))

        return outcomes
