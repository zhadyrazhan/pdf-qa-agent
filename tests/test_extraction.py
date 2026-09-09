"""Tests for PDF extraction: text layer directly, VLM OCR with a mocked client
(no real calls to the OpenAI API)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pdf_qa_agent.extraction import PDFExtractor, build_context
from pdf_qa_agent.retry import RetryConfig
from pdf_qa_agent.schemas import ExtractedPage


def _mock_client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.parse.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=ExtractedPage(text=text)))]
    )
    return client


def test_text_layer_extraction_used_when_text_present(text_pdf):
    client = MagicMock()  # should never be called
    extractor = PDFExtractor(client=client)

    pages = extractor.extract(text_pdf)

    assert len(pages) == 2
    assert all(p.source == "text_layer" for p in pages)
    assert "Total revenue" in pages[0].text
    assert "Jane Doe" in pages[1].text
    client.chat.completions.parse.assert_not_called()


def test_vlm_ocr_fallback_used_when_no_text_layer(blank_pdf):
    client = _mock_client_returning("# Recognized scanned text")
    extractor = PDFExtractor(client=client, retry_config=RetryConfig(max_retries=1))

    pages = extractor.extract(blank_pdf)

    assert len(pages) == 1
    assert pages[0].source == "vlm_ocr"
    assert pages[0].text == "# Recognized scanned text"
    client.chat.completions.parse.assert_called_once()
    # Confirm the page was sent as base64 PNG alongside the text prompt
    call_kwargs = client.chat.completions.parse.call_args.kwargs
    content = call_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"


def test_build_context_includes_page_numbers_and_source():
    from pdf_qa_agent.schemas import PageContent

    pages = [
        PageContent(page=1, text="Hello", source="text_layer"),
        PageContent(page=2, text="World", source="vlm_ocr"),
    ]
    context = build_context(pages)

    assert "Страница 1" in context
    assert "text_layer" in context
    assert "Страница 2" in context
    assert "vlm_ocr" in context
    assert "Hello" in context and "World" in context


def test_text_layer_min_chars_threshold_is_configurable(blank_pdf):
    # text_layer_min_chars=0 -> even an empty string "passes" as a text layer,
    # so the VLM should never be called.
    client = MagicMock()
    extractor = PDFExtractor(client=client, text_layer_min_chars=0)

    pages = extractor.extract(blank_pdf)

    assert pages[0].source == "text_layer"
    client.chat.completions.parse.assert_not_called()


def test_extract_from_url_fetches_bytes_and_parses_them(text_pdf, monkeypatch):
    """PDFExtractor.extract() accepts an http(s) URL: downloads bytes via
    pdf_qa_agent.fetch.fetch_pdf_bytes and loads them into pdfium directly from
    memory, without writing to disk. fetch_pdf_bytes is mocked, no real network."""
    pdf_bytes = text_pdf.read_bytes()
    fake_fetch = MagicMock(return_value=pdf_bytes)
    monkeypatch.setattr("pdf_qa_agent.extraction.fetch_pdf_bytes", fake_fetch)

    client = MagicMock()  # text layer is present -> VLM should not be called
    extractor = PDFExtractor(client=client, max_download_bytes=123, fetch_timeout=7.0)

    pages = extractor.extract("https://example.com/report.pdf")

    assert len(pages) == 2
    assert "Total revenue" in pages[0].text
    fake_fetch.assert_called_once_with(
        "https://example.com/report.pdf", max_bytes=123, timeout=7.0
    )
    client.chat.completions.parse.assert_not_called()


def test_extract_from_url_propagates_fetch_errors(monkeypatch):
    from pdf_qa_agent.fetch import UnsafeURLError

    def _raise(*args, **kwargs):
        raise UnsafeURLError("blocked: resolves to a private address")

    monkeypatch.setattr("pdf_qa_agent.extraction.fetch_pdf_bytes", _raise)
    extractor = PDFExtractor(client=MagicMock())

    with pytest.raises(UnsafeURLError):
        extractor.extract("http://169.254.169.254/doc.pdf")
