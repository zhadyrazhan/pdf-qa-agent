"""Safely downloads a PDF from a URL.

A server-side fetch of a user-supplied URL is a classic SSRF vector: without
checks, a link like ``http://169.254.169.254/latest/meta-data`` would make the
server hit internal infrastructure instead of an external PDF. Defenses here
(best-effort, not an absolute guarantee — see "Known limitations" in the README):

  - only http/https schemes are allowed;
  - the host is resolved via DNS and rejected if any resolved address is
    private, loopback, link-local, multicast, reserved, or unspecified;
  - after HTTP redirects, the final host is resolved and checked again (a
    redirect to an internal address doesn't slip through);
  - download size is capped (Content-Length check plus a streamed count, so a
    missing/wrong header can't be used to bypass it);
  - the downloaded content must start with the ``%PDF-`` signature.

To guard against DNS rebinding, a custom httpcore transport re-resolves the
hostname immediately before opening the TCP connection and connects to that
verified IP, while keeping the original hostname for TLS/SNI. Proxy
environment variables are disabled so the request can't be silently redirected
through an external proxy.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpcore
import httpx
from httpcore._backends.sync import SyncBackend
from httpx._transports.default import ResponseStream, map_httpcore_exceptions

_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

DEFAULT_MAX_PDF_BYTES = 1 * 1024 * 1024  # 1 MB — matches the API's file upload limit
DEFAULT_FETCH_TIMEOUT = 30.0
DEFAULT_MAX_REDIRECTS = 5
PDF_MAGIC = b"%PDF-"

ALLOWED_SCHEMES = ("http", "https")


class UnsafeURLError(ValueError):
    """URL rejected by the SSRF check (disallowed scheme or address)."""


class PDFTooLargeError(ValueError):
    """The downloaded file exceeds the allowed size."""


class PDFFetchError(RuntimeError):
    """Network error, HTTP error, or the content doesn't look like a PDF."""


def is_url(source: str) -> bool:
    """True for anything shaped like `scheme://...` (http, https, ftp, file, ...).

    Deliberately broader than just http/https: a non-http(s) scheme (e.g. ftp://,
    file://) should be routed into fetch_pdf_bytes() so _validate_host() rejects it
    with a clear "unsupported scheme" error, rather than falling through to being
    treated as a local filesystem path (which produces a confusing pdfium error and
    can leak the server's working directory in it).
    """
    return bool(_URL_SCHEME_RE.match(source))


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_public_ip(hostname: str) -> str:
    """Resolve a hostname and return a validated public address.

    This function is called by the network transport immediately before opening
    the socket. That closes the usual validation-then-resolve DNS rebinding gap:
    the socket is opened to the exact address that was just checked.
    """
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host: {hostname}") from exc

    resolved_ips = {sockaddr[0] for _, _, _, _, sockaddr in addrinfo}
    unsafe_ips = [ip for ip in resolved_ips if not _is_public_ip(ip)]
    if unsafe_ips:
        raise UnsafeURLError(f"{hostname} resolves to a non-public address — refusing to fetch")
    if not resolved_ips:
        raise UnsafeURLError(f"Could not resolve host: {hostname}")
    return sorted(resolved_ips)[0]


def _validate_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Unsupported URL scheme {parsed.scheme!r} — only http/https are allowed")
    if not parsed.hostname:
        raise UnsafeURLError("URL has no hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError("URL contains an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise UnsafeURLError("URL contains an invalid port")
    _resolve_public_ip(parsed.hostname)


class _PublicOnlyNetworkBackend(SyncBackend):
    """Connect to the just-validated public IP while retaining TLS SNI.

    httpcore passes the original hostname to ``start_tls`` after this method
    returns, so certificates are still verified for ``example.com`` rather than
    for the numeric IP address. Redirects trigger another connection and are
    therefore resolved and validated independently.
    """

    def connect_tcp(self, host: str, port: int, timeout: float | None = None, local_address: str | None = None, socket_options=None):
        public_ip = _resolve_public_ip(host)
        return super().connect_tcp(public_ip, port, timeout, local_address, socket_options)


class _PublicOnlyHTTPTransport(httpx.BaseTransport):
    """httpx transport with public-IP-pinned connections and no proxy env vars."""

    def __init__(self):
        import ssl

        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=5.0,
            network_backend=_PublicOnlyNetworkBackend(),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        req = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with map_httpcore_exceptions():
            response = self._pool.handle_request(req)
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=ResponseStream(response.stream),
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._pool.close()


def fetch_pdf_bytes(
    url: str,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
    client: httpx.Client | None = None,
) -> bytes:
    """Downloads a PDF from a URL with SSRF checks and a size cap.

    `client` lets tests pass a preconfigured httpx.Client (e.g. with a
    MockTransport) without touching the network.
    """
    _validate_host(url)

    owns_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=True,
        max_redirects=DEFAULT_MAX_REDIRECTS,
        timeout=timeout,
        trust_env=False,
        transport=_PublicOnlyHTTPTransport(),
    )
    try:
        try:
            with http_client.stream("GET", url) as response:
                # Re-check the final host after redirects too, or the first
                # check could be bypassed by redirecting to an internal address.
                _validate_host(str(response.url))

                if response.status_code >= 400:
                    raise PDFFetchError(f"Failed to fetch PDF: HTTP {response.status_code} for {url}")

                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise PDFFetchError("Origin returned an invalid Content-Length header") from exc
                    if declared_size < 0:
                        raise PDFFetchError("Origin returned an invalid Content-Length header")
                    if declared_size > max_bytes:
                        raise PDFTooLargeError(
                            "File needs to be less than 1 MB"
                        )

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise PDFTooLargeError(
                            "File needs to be less than 1 MB"
                        )
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise PDFFetchError(f"Network error fetching {url}: {exc}") from exc
    finally:
        if owns_client:
            http_client.close()

    content = b"".join(chunks)
    if not content.startswith(PDF_MAGIC):
        raise PDFFetchError(f"Content at {url} does not look like a PDF (missing {PDF_MAGIC!r} header)")

    return content
