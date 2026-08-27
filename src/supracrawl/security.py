import asyncio
import ipaddress
from urllib.parse import urlsplit


class UnsafeUrlError(ValueError):
    pass


def _is_forbidden_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


async def validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("Only http:// and https:// URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URLs containing credentials are not allowed")
    if not parsed.hostname:
        raise UnsafeUrlError("URL must contain a hostname")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain", "metadata.google.internal"}:
        raise UnsafeUrlError("Local or metadata hostnames are not allowed")
    if hostname.endswith((".localhost", ".local")):
        raise UnsafeUrlError("Local hostnames are not allowed")

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_forbidden_ip(str(literal)):
            raise UnsafeUrlError("Private or non-routable IP addresses are not allowed")
        return

    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError("URL contains an invalid port") from exc
    port = explicit_port or (443 if parsed.scheme == "https" else 80)

    loop = asyncio.get_running_loop()
    try:
        answers = await loop.getaddrinfo(hostname, port)
    except OSError as exc:
        raise UnsafeUrlError(f"Hostname could not be resolved: {hostname}") from exc

    addresses = {answer[4][0] for answer in answers}
    if not addresses:
        raise UnsafeUrlError(f"Hostname resolved to no addresses: {hostname}")
    if any(_is_forbidden_ip(address) for address in addresses):
        raise UnsafeUrlError("Hostname resolves to a private or non-routable address")
