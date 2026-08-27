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
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        base_url = get_provider_env("SUPRACRAWL_URL").rstrip("/")
        if not base_url:
            return {"success": False, "error": "SUPRACRAWL_URL is not configured"}

        timeout, timeout_error = self._timeout()
        if timeout_error:
            return {"success": False, "error": timeout_error}

        safe_limit = max(1, min(int(limit), 20))
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{base_url}/v1/search",
                    json={"query": query, "limit": safe_limit},
                )
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return {"success": False, "error": f"SupraCrawl search failed: {exc}"}

        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            return {"success": False, "error": "SupraCrawl returned an invalid search response"}

        web: list[dict[str, Any]] = []
        for fallback_position, item in enumerate(body["results"], start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            description = item.get("description")
            position = item.get("position")
            if not isinstance(url, str) or not url:
                continue
            web.append(
                {
                    "title": title if isinstance(title, str) else "",
                    "url": url,
                    "description": description if isinstance(description, str) else "",
                    "position": position if isinstance(position, int) else fallback_position,
                }
            )

        return {"success": True, "data": {"web": web}}

    async def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        base_url = get_provider_env("SUPRACRAWL_URL").rstrip("/")
        if not base_url:
            return self._errors(urls, "SUPRACRAWL_URL is not configured")

        timeout, timeout_error = self._timeout()
        if timeout_error:
            return self._errors(urls, timeout_error)

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
    def _timeout() -> tuple[float, str | None]:
        timeout_raw = get_provider_env("SUPRACRAWL_TIMEOUT_S") or "20"
        try:
            timeout = float(timeout_raw)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except ValueError:
            return 0.0, "SUPRACRAWL_TIMEOUT_S must be a positive finite number"
        return timeout, None

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
