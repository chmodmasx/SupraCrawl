import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from .security import validate_public_url


@dataclass
class _CachedRobots:
    expires_at: float
    parser: RobotFileParser


class RobotsPolicy:
    def __init__(self, ttl_s: int = 3600, timeout_s: float = 5.0) -> None:
        self.ttl_s = ttl_s
        self.timeout_s = timeout_s
        self._cache: dict[str, _CachedRobots] = {}

    async def allowed(self, url: str, user_agent: str) -> bool:
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cached = self._cache.get(origin)
        now = time.monotonic()
        if cached and cached.expires_at > now:
            return cached.parser.can_fetch(user_agent, url)

        parser = await self._fetch(origin)
        self._cache[origin] = _CachedRobots(now + self.ttl_s, parser)
        return parser.can_fetch(user_agent, url)

    async def _fetch(self, origin: str) -> RobotFileParser:
        current = urljoin(origin, "/robots.txt")
        parser = RobotFileParser()
        parser.set_url(current)

        async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=False) as client:
            for _ in range(3):
                await validate_public_url(current)
                try:
                    response = await client.get(current, headers={"User-Agent": "SupraCrawl/0.1"})
                except httpx.HTTPError:
                    parser.parse(["User-agent: *", "Disallow: /"])
                    return parser

                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        break
                    current = urljoin(current, location)
                    continue
                if 400 <= response.status_code < 500:
                    parser.parse([])
                    return parser
                if response.status_code >= 500:
                    parser.parse(["User-agent: *", "Disallow: /"])
                    return parser

                text = response.text[:512_000]
                parser.parse(text.splitlines())
                return parser

        parser.parse(["User-agent: *", "Disallow: /"])
        return parser
