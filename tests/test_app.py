import httpx
import pytest

import supracrawl.app as app_module
from supracrawl.fetcher import FetchError


@pytest.mark.asyncio
async def test_extract_keeps_per_url_failure_inside_success_envelope(monkeypatch) -> None:
    async def fail_extract(_url: str):
        raise FetchError("fixture failure")

    monkeypatch.setattr(app_module.extractor, "extract", fail_extract)
    monkeypatch.setattr(app_module.cache, "redis", None)

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/extract",
            json={"urls": ["https://example.com/one", "https://example.com/two"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert len(body["documents"]) == 2
    assert all(item["error"] == "Fetch failed: fixture failure" for item in body["documents"])
