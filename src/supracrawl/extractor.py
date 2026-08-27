import hashlib
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
import trafilatura
from selectolax.parser import HTMLParser

from .config import Settings
from .fetcher import FetchResult, HttpFetcher
from .urls import normalize_url


@dataclass(slots=True)
class Extraction:
    title: str
    markdown: str
    canonical_url: str
    extractor: str
    quality: float
    rendered: bool


class Extractor:
    def __init__(self, settings: Settings, fetcher: HttpFetcher) -> None:
        self.settings = settings
        self.fetcher = fetcher

    async def extract(self, url: str) -> tuple[FetchResult, Extraction]:
        fetched = await self.fetcher.fetch_html(url)
        extraction = await self.extract_fetched(fetched)
        return fetched, extraction

    async def extract_fetched(self, fetched: FetchResult) -> Extraction:
        metadata = self._metadata(fetched.html, fetched.final_url)

        extraction = await self._worker_extract(fetched.html, fetched.final_url, render=False)
        if extraction is None:
            extraction = self._trafilatura_extract(fetched.html, fetched.final_url)

        extraction.title = extraction.title or metadata["title"]
        extraction.canonical_url = metadata["canonical_url"]
        extraction.quality = self._quality(extraction.markdown, fetched.html)

        if (
            self.settings.browser_enabled
            and self._needs_browser(extraction.markdown, fetched.html, extraction.quality)
        ):
            rendered = await self._worker_extract("", fetched.final_url, render=True)
            if rendered:
                rendered.title = rendered.title or extraction.title
                rendered.canonical_url = extraction.canonical_url
                rendered.quality = self._quality(rendered.markdown, rendered.markdown)
                rendered_is_better = (
                    rendered.quality > extraction.quality
                    or len(rendered.markdown) > len(extraction.markdown)
                )
                if rendered_is_better:
                    extraction = rendered

        return extraction

    async def _worker_extract(self, html: str, url: str, render: bool) -> Extraction | None:
        endpoint = "/render-extract" if render else "/extract"
        payload = {"url": url} if render else {"url": url, "html": html}
        try:
            timeout = self.settings.extractor_worker_timeout_s
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.settings.extractor_worker_url.rstrip("/") + endpoint,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        if not isinstance(data, dict):
            return None
        markdown_value = data.get("markdown")
        if not isinstance(markdown_value, str):
            return None
        markdown = markdown_value.strip()
        if not markdown:
            return None
        title_value = data.get("title")
        title = title_value.strip() if isinstance(title_value, str) else ""
        return Extraction(
            title=title,
            markdown=markdown,
            canonical_url=normalize_url(url),
            extractor="readability" + ("+playwright" if render else ""),
            quality=0.0,
            rendered=render,
        )

    def _trafilatura_extract(self, html: str, url: str) -> Extraction:
        markdown = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            include_formatting=True,
            favor_precision=True,
        ) or ""
        return Extraction(
            title="",
            markdown=markdown.strip(),
            canonical_url=normalize_url(url),
            extractor="trafilatura",
            quality=0.0,
            rendered=False,
        )

    def _metadata(self, html: str, final_url: str) -> dict[str, str]:
        tree = HTMLParser(html)
        title_node = tree.css_first("title")
        title = title_node.text(strip=True) if title_node else ""

        canonical = ""
        for node in tree.css("link"):
            rel = (node.attributes.get("rel") or "").lower().split()
            href = node.attributes.get("href")
            if "canonical" in rel and href:
                canonical = urljoin(final_url, href)
                break

        try:
            canonical_url = normalize_url(canonical or final_url)
        except ValueError:
            canonical_url = normalize_url(final_url)
        return {"title": title, "canonical_url": canonical_url}

    def _quality(self, markdown: str, html: str) -> float:
        text = re.sub(r"\s+", " ", markdown).strip()
        if not text:
            return 0.0
        length_score = min(1.0, len(text) / max(self.settings.min_useful_chars, 1))
        ratio = len(text) / max(len(html), 1)
        ratio_score = min(1.0, ratio / 0.08)
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        unique_ratio = len(set(lines)) / max(len(lines), 1)
        return round(
            max(0.0, min(1.0, 0.55 * length_score + 0.25 * ratio_score + 0.20 * unique_ratio)),
            4,
        )

    def _needs_browser(self, markdown: str, html: str, quality: float) -> bool:
        lower = html.lower()
        spa_shell = any(
            marker in lower
            for marker in ("id=\"__next\"", "id=\"root\"", "ng-version=", "data-reactroot")
        )
        return (
            len(markdown.strip()) < self.settings.min_useful_chars
            or quality < 0.35
            or (spa_shell and quality < 0.6)
        )


def content_hash(markdown: str) -> str:
    normalized = re.sub(r"\s+", " ", markdown).strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()
