import hashlib

import orjson
from redis.asyncio import Redis


class JsonCache:
    def __init__(self, url: str | None, ttl_s: int) -> None:
        self.ttl_s = ttl_s
        self.redis = Redis.from_url(url, decode_responses=False) if url else None

    @staticmethod
    def key(prefix: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"supracrawl:{prefix}:{digest}"

    async def get(self, key: str) -> dict | None:
        if not self.redis:
            return None
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            decoded = orjson.loads(value)
            return decoded if isinstance(decoded, dict) else None
        except Exception:
            # Cache corruption or Redis outages must never break extraction.
            return None

    async def set(self, key: str, value: dict) -> None:
        if not self.redis or self.ttl_s <= 0:
            return
        try:
            await self.redis.set(key, orjson.dumps(value), ex=self.ttl_s)
        except Exception:
            return

    async def close(self) -> None:
        if self.redis:
            await self.redis.aclose()
