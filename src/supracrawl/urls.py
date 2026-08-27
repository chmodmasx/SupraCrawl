from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "igshid",
}


def normalize_url(url: str) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".").lower()

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc

    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    host_for_netloc = f"[{host}]" if ":" in host and not host.startswith("[") else host
    netloc = host_for_netloc if port is None or default_port else f"{host_for_netloc}:{port}"

    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lower = key.lower()
        if lower.startswith("utm_") or lower in _TRACKING_KEYS:
            continue
        query.append((key, value))
    query.sort()

    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, urlencode(query, doseq=True), ""))
