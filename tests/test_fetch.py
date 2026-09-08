"""Тесты загрузки PDF по URL: SSRF-проверки, лимит размера, проверка сигнатуры PDF.

Полностью офлайн — DNS-резолвинг подменяется через monkeypatch (без реального сетевого
резолва), а HTTP-запросы идут через httpx.MockTransport (без реальных обращений к сети).
"""
import httpx
import pytest

from pdf_qa_agent.fetch import (
    PDFFetchError,
    PDFTooLargeError,
    UnsafeURLError,
    fetch_pdf_bytes,
    is_url,
)

PDF_BYTES = b"%PDF-1.4\n%mock pdf content\n%%EOF"


def _fake_addrinfo(ip: str):
    """Форма возврата socket.getaddrinfo, достаточная для _validate_host."""
    return [(2, 1, 6, "", (ip, 0))]


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch):
    """По умолчанию все тесты резолвят любой хост в публичный IP — без реального DNS.
    Отдельные тесты переопределяют это через monkeypatch внутри теста."""
    monkeypatch.setattr(
        "pdf_qa_agent.fetch.socket.getaddrinfo",
        lambda host, port: _fake_addrinfo("93.184.216.34"),
    )


def test_is_url_detects_http_and_https():
    assert is_url("http://example.com/doc.pdf")
    assert is_url("https://example.com/doc.pdf")
    assert not is_url("/local/path/doc.pdf")
    assert not is_url("doc.pdf")
    assert not is_url("C:\\Users\\me\\doc.pdf")


def test_is_url_also_detects_non_http_schemes_so_they_hit_scheme_validation():
    # ftp://, file:// etc. must still be recognized as "URL-shaped" so extraction
    # routes them into fetch_pdf_bytes()/_validate_host() (-> clear "unsupported
    # scheme" error) instead of silently treating them as a local file path.
    assert is_url("ftp://example.com/doc.pdf")
    assert is_url("file:///etc/passwd")


def test_rejects_non_http_scheme(monkeypatch):
    with pytest.raises(UnsafeURLError, match="scheme"):
        fetch_pdf_bytes("ftp://example.com/doc.pdf")


def test_rejects_url_resolving_to_loopback(monkeypatch):
    monkeypatch.setattr(
        "pdf_qa_agent.fetch.socket.getaddrinfo",
        lambda host, port: _fake_addrinfo("127.0.0.1"),
    )
    with pytest.raises(UnsafeURLError, match="non-public"):
        fetch_pdf_bytes("http://localhost/doc.pdf")


def test_rejects_url_resolving_to_private_network(monkeypatch):
    monkeypatch.setattr(
        "pdf_qa_agent.fetch.socket.getaddrinfo",
        lambda host, port: _fake_addrinfo("10.0.0.5"),
    )
    with pytest.raises(UnsafeURLError, match="non-public"):
        fetch_pdf_bytes("http://internal.example/doc.pdf")


def test_rejects_url_resolving_to_link_local_metadata_endpoint(monkeypatch):
    monkeypatch.setattr(
        "pdf_qa_agent.fetch.socket.getaddrinfo",
        lambda host, port: _fake_addrinfo("169.254.169.254"),
    )
    with pytest.raises(UnsafeURLError, match="non-public"):
        fetch_pdf_bytes("http://169.254.169.254/latest/meta-data")


def test_rejects_unresolvable_host(monkeypatch):
    import socket

    def _raise(host, port):
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr("pdf_qa_agent.fetch.socket.getaddrinfo", _raise)
    with pytest.raises(UnsafeURLError, match="Could not resolve"):
        fetch_pdf_bytes("http://does-not-resolve.invalid/doc.pdf")


def test_rejects_urls_with_embedded_credentials():
    with pytest.raises(UnsafeURLError, match="credentials"):
        fetch_pdf_bytes("https://user:password@example.com/doc.pdf")


def test_rejects_invalid_content_length_header():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_BYTES, headers={"content-length": "not-a-number"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(PDFFetchError, match="Content-Length"):
        fetch_pdf_bytes("https://example.com/report.pdf", client=client)


def test_fetches_pdf_bytes_successfully():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_BYTES, headers={"content-length": str(len(PDF_BYTES))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch_pdf_bytes("https://example.com/report.pdf", client=client)

    assert result == PDF_BYTES


def test_rejects_when_content_length_header_exceeds_max():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_BYTES, headers={"content-length": "999999999"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(PDFTooLargeError):
        fetch_pdf_bytes("https://example.com/report.pdf", max_bytes=100, client=client)


def test_rejects_when_streamed_body_exceeds_max_without_content_length_header():
    big_body = PDF_BYTES + b"0" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big_body)  # без content-length

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(PDFTooLargeError):
        fetch_pdf_bytes("https://example.com/report.pdf", max_bytes=50, client=client)


def test_rejects_non_pdf_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not a pdf</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(PDFFetchError, match="does not look like a PDF"):
        fetch_pdf_bytes("https://example.com/report.pdf", client=client)


def test_raises_on_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(PDFFetchError, match="404"):
        fetch_pdf_bytes("https://example.com/missing.pdf", client=client)


def test_raises_pdf_fetch_error_on_network_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(PDFFetchError, match="Network error"):
        fetch_pdf_bytes("https://example.com/report.pdf", client=client)


def test_host_is_revalidated_after_response_received(monkeypatch):
    """fetch_pdf_bytes проверяет хост дважды: до запроса (по URL) и после ответа
    (по response.url, который отражает финальный адрес после редиректов). Здесь мы
    подменяем DNS так, будто между двумя проверками резолвинг изменился на приватный
    адрес (упрощённая модель редиректа/DNS rebinding) — вторая проверка должна поймать
    это и отклонить запрос, даже когда исходный хост выглядел публичным."""
    calls = {"n": 0}

    def fake_getaddrinfo(host, port):
        calls["n"] += 1
        return _fake_addrinfo("93.184.216.34") if calls["n"] == 1 else _fake_addrinfo("10.0.0.1")

    monkeypatch.setattr("pdf_qa_agent.fetch.socket.getaddrinfo", fake_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_BYTES)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(UnsafeURLError, match="non-public"):
        fetch_pdf_bytes("https://example.com/report.pdf", client=client)

    assert calls["n"] == 2  # обе проверки хоста действительно выполнились
