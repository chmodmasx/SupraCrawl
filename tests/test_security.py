import pytest

from supracrawl.security import UnsafeUrlError, validate_public_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "file:///etc/passwd",
    ],
)
async def test_private_and_non_http_destinations_are_rejected(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        await validate_public_url(url)
