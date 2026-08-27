from __future__ import annotations

import os
from typing import Any

import httpx
from agent.web_search_provider import WebSearchProvider


class SupraCrawlWebSearchProvider(WebSearchProvider):
    @property
    def name(self) -> str:
        return "supracrawl"

    @property
    def display_name(self) -> str:
        return "SupraCrawl"

    def is_available(self) -> bool:
        # Hermes calls this frequently; never perform network I/O here.
        return bool(os.getenv("SUPRACRAWL_URL", "").strip())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]:
        base_url = os.environ["SUPRACRAWL_URL"].rstrip("/")
        timeout = float(os.getenv("SUPRACRAWL_TIMEOUT_S", "20"))
        payload: dict[str, Any] = {"urls": urls}

        # SupraCrawl supports query-aware in-page passage selection. Hermes does
        # not require this kwarg today, but forwarding it is harmless when a
        # caller/provider extension supplies it.
        query = kwargs.get("query")
        if isinstance(query, str) and query.strip():
            payload["query"] = query

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/v1/extract", json=payload)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"success": False, "error": f"SupraCrawl request failed: {exc}"}

        documents = body.get("documents")
        if not isinstance(documents, list):
            return {"success": False, "error": "SupraCrawl returned an invalid response"}

        return {"success": True, "data": documents}
