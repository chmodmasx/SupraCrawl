from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin

import httpx

from .config import Settings
from .robots import RobotsPolicy
from .security import UnsafeUrlError, validate_public_url


class FetchError(RuntimeError):
    pass


@dataclass(slots=True)
class FetchResult:
    fetch_url: str
    final_url: str
    status_code: int
    content_type: str
    html: str
    fetched_at: str


class HttpFetcher:
    def __init__(self, settings: Settings, robots: RobotsPolicy | None = None) -> None:
        self.settings = settings
        self.robots = robots or RobotsPolicy()

    async def fetch_html(self, url: str) -> FetchResult:
        fetch_url = url
        current = url
        headers = {
            "User-Agent": self.settings.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "gzip, deflate",
        }

        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_s,
            follow_redirects=False,
            headers=headers,
        ) as client:
            for redirect_count in range(self.settings.max_redirects + 1):
                await validate_public_url(current)
                if self.settings.obey_robots_txt:
                    allowed = await self.robots.allowed(current, self.settings.user_agent)
                    if not allowed:
                        raise FetchError("Blocked by robots.txt")

                try:
                    async with client.stream("GET", current) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise FetchError("Redirect response without Location header")
                            if redirect_count >= self.settings.max_redirects:
                                raise FetchError("Maximum redirect count exceeded")
                            current = urljoin(current, location)
                            continue

                        if response.status_code >= 400:
                            raise FetchError(f"HTTP {response.status_code}")

                        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        if content_type not in {"text/html", "application/xhtml+xml"}:
                            raise FetchError(f"Unsupported content type: {content_type or 'unknown'}")

                        declared = response.headers.get("content-length")
                        if declared:
                            try:
                                if int(declared) > self.settings.max_html_bytes:
                                    raise FetchError("Response exceeds configured HTML size limit")
                            except ValueError:
                                pass

                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > self.settings.max_html_bytes:
                                raise FetchError("Response exceeds configured HTML size limit")

                        encoding = response.encoding or "utf-8"
                        try:
                            html = body.decode(encoding, errors="replace")
                        except LookupError:
                            html = body.decode("utf-8", errors="replace")

                        return FetchResult(
                            fetch_url=fetch_url,
                            final_url=current,
                            status_code=response.status_code,
                            content_type=content_type,
                            html=html,
                            fetched_at=datetime.now(UTC).isoformat(),
                        )
                except UnsafeUrlError:
                    raise
                except httpx.HTTPError as exc:
                    raise FetchError(str(exc)) from exc

        raise FetchError("Unable to fetch URL")
