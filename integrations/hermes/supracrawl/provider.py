from __future__ import annotations

import math
from typing import Any

import httpx
from agent.web_search_provider import WebSearchProvider, get_provider_env


class SupraCrawlWebSearchProvider(WebSearchProvider):
    @property
    def name(self) -> str:
        return "supracrawl"

    @property
    def display_name(self) -> str:
        return "SupraCrawl"

    def is_available(self) -> bool:
        # Hermes calls this frequently; get_provider_env() is config-aware and
        # performs no network I/O.
        return bool(get_provider_env("SUPRACRAWL_URL"))

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        base_url = get_provider_env("SUPRACRAWL_URL").rstrip("/")
        if not base_url:
            return self._errors(urls, "SUPRACRAWL_URL is not configured")

        timeout_raw = get_provider_env("SUPRACRAWL_TIMEOUT_S") or "20"
        try:
            timeout = float(timeout_raw)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except ValueError:
            return self._errors(urls, "SUPRACRAWL_TIMEOUT_S must be a positive finite number")

        payload: dict[str, Any] = {"urls": urls}

        # SupraCrawl supports query-aware in-page passage selection. Hermes
        # may forward additional kwargs; unknown fields are intentionally
        # ignored to preserve forward compatibility with the provider ABC.
        query = kwargs.get("query")
        if isinstance(query, str) and query.strip():
            payload["query"] = query

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/v1/extract", json=payload)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return self._errors(urls, f"SupraCrawl request failed: {exc}")

        if not isinstance(body, dict):
            return self._errors(urls, "SupraCrawl returned an invalid response")
        documents = body.get("documents")
        if not isinstance(documents, list):
            return self._errors(urls, "SupraCrawl returned an invalid response")
        if not all(isinstance(document, dict) for document in documents):
            return self._errors(urls, "SupraCrawl returned invalid document entries")

        # Hermes' WebSearchProvider.extract() contract is a list of document
        # dicts. Do not wrap this in the search-style success/data envelope.
        return documents

    @staticmethod
    def _errors(urls: list[str], message: str) -> list[dict[str, Any]]:
        return [
            {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": message,
            }
            for url in urls
        ]
