"""Тесты извлечения PDF: текстовый слой напрямую, VLM OCR — с замоканным клиентом
(без реальных обращений к Anthropic API)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from pdf_qa_agent.extraction import PDFExtractor, build_context
from pdf_qa_agent.retry import RetryConfig
from pdf_qa_agent.schemas import ExtractedPage


def _mock_client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.parse.return_value = SimpleNamespace(parsed_output=ExtractedPage(text=text))
    return client


def test_text_layer_extraction_used_when_text_present(text_pdf):
    client = MagicMock()  # не должен вызываться вовсе
    extractor = PDFExtractor(client=client)

    pages = extractor.extract(text_pdf)

    assert len(pages) == 2
    assert all(p.source == "text_layer" for p in pages)
    assert "Total revenue" in pages[0].text
    assert "Jane Doe" in pages[1].text
    client.messages.parse.assert_not_called()


def test_vlm_ocr_fallback_used_when_no_text_layer(blank_pdf):
    client = _mock_client_returning("# Recognized scanned text")
    extractor = PDFExtractor(client=client, retry_config=RetryConfig(max_retries=1))

    pages = extractor.extract(blank_pdf)

    assert len(pages) == 1
    assert pages[0].source == "vlm_ocr"
    assert pages[0].text == "# Recognized scanned text"
    client.messages.parse.assert_called_once()
    # Убедимся, что странице передан именно PNG в base64 и текстовый промпт
    call_kwargs = client.messages.parse.call_args.kwargs
    content = call_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
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
    # text_layer_min_chars=0 -> даже пустая строка "проходит" как текстовый слой,
    # VLM вызываться не должен.
    client = MagicMock()
    extractor = PDFExtractor(client=client, text_layer_min_chars=0)

    pages = extractor.extract(blank_pdf)

    assert pages[0].source == "text_layer"
    client.messages.parse.assert_not_called()
