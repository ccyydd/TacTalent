"""Credential-free loopback proxy URL validation."""

from urllib.parse import urlsplit


def validate_api_proxy_url(value: object) -> str | None:
    """Validate a credential-free HTTP CONNECT proxy on loopback.

    Args:
        value: Proxy URL, or None to retain the default transport.

    Returns:
        A canonical loopback URL, or None.

    Raises:
        ValueError: The value is not a permitted loopback proxy URL.
    """
    if value is None:
        return None
    message = "API proxy must be http://127.0.0.1:PORT or http://localhost:PORT"
    if not isinstance(value, str) or any(ord(char) <= 32 for char in value):
        raise ValueError(message)
    try:
        proxy = urlsplit(value)
        port = proxy.port
    except ValueError:
        raise ValueError(message) from None
    if (
        proxy.scheme != "http"
        or proxy.hostname not in {"127.0.0.1", "localhost"}
        or proxy.username is not None
        or proxy.password is not None
        or port is None
        or not 1 <= port <= 65535
        or proxy.path
        or "?" in value
        or "#" in value
    ):
        raise ValueError(message)
    return f"http://{proxy.hostname}:{port}"
