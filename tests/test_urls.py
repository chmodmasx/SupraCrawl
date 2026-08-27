from supracrawl.urls import normalize_url


def test_normalize_url_removes_tracking_and_fragment() -> None:
    url = "HTTPS://Example.COM:443/path?utm_source=x&b=2&a=1#section"
    assert normalize_url(url) == "https://example.com/path?a=1&b=2"
