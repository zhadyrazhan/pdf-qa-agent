"""Извлечение текста и структуры из PDF: текстовый слой + VLM OCR fallback для сканов.

Стратегия на страницу:
  1. Сначала пробуем текстовый слой PDF (pypdfium2) — дёшево и точно для text-based PDF.
  2. Если текстового слоя нет или он короче TEXT_LAYER_MIN_CHARS (страница — скан или
     таблица, отрисованная как изображение), страница рендерится в PNG и распознаётся
     мультимодальной моделью (Claude Vision) с OCR-промптом, сохраняющим структуру
     документа в markdown.

Вызовы VLM обёрнуты retry/backoff-логикой (pdf_qa_agent.retry), т.к. это единственная
сетевая операция в пайплайне извлечения и она подвержена временным сбоям API.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Callable, List

import anthropic
import pypdfium2 as pdfium

from pdf_qa_agent.fetch import DEFAULT_FETCH_TIMEOUT, DEFAULT_MAX_PDF_BYTES, fetch_pdf_bytes, is_url
from pdf_qa_agent.retry import RetryConfig, with_retry
from pdf_qa_agent.schemas import ExtractedPage, PageContent

DEFAULT_MODEL = "claude-opus-5"
TEXT_LAYER_MIN_CHARS = 20  # ниже этого порога страница считается сканом без текстового слоя
DEFAULT_RENDER_SCALE = 2.5

OCR_PROMPT_TEMPLATE = (
    "Это страница {page_num} из {total_pages} документа. Похоже, что обычное выделение "
    "текста здесь не работает (страница — это изображение/скан или таблица без текстового "
    "слоя). Распознай (OCR) весь текст на странице и верни его в виде markdown, сохраняя "
    "структуру документа: заголовки ('# ...'), абзацы, списки, таблицы (в markdown-формате "
    "с '|'), подписи таблиц и рисунков. Транскрибируй текст как есть, ничего не добавляй от "
    "себя и не описывай изображение."
)


class PDFExtractor:
    """Извлекает содержимое PDF постранично: текстовый слой либо VLM OCR."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        text_layer_min_chars: int = TEXT_LAYER_MIN_CHARS,
        render_scale: float = DEFAULT_RENDER_SCALE,
        retry_config: RetryConfig | None = None,
        log: Callable[[str], None] | None = None,
        max_download_bytes: int = DEFAULT_MAX_PDF_BYTES,
        fetch_timeout: float = DEFAULT_FETCH_TIMEOUT,
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.text_layer_min_chars = text_layer_min_chars
        self.render_scale = render_scale
        self._log = log or (lambda msg: print(msg, file=sys.stderr))
        self.max_download_bytes = max_download_bytes
        self.fetch_timeout = fetch_timeout
        # Оборачиваем сетевой вызов ретраями один раз в конструкторе, чтобы retry_config
        # можно было настраивать per-instance (в т.ч. в тестах, с sleep_fn=no-op).
        self._call_vlm_with_retry = with_retry(retry_config)(self._call_vlm)

    def extract(self, pdf_source: str | Path) -> List[PageContent]:
        pdf = self._open(pdf_source)
        pages: List[PageContent] = []
        total_pages = len(pdf)

        for i in range(total_pages):
            page = pdf[i]
            page_num = i + 1
            text = page.get_textpage().get_text_range().strip()

            if len(text) >= self.text_layer_min_chars:
                pages.append(PageContent(page=page_num, text=text, source="text_layer"))
            else:
                self._log(f"  page {page_num}: no text layer, running VLM OCR...")
                image_b64 = self._render_page_b64(page)
                extracted = self._call_vlm_with_retry(image_b64, page_num, total_pages)
                pages.append(PageContent(page=page_num, text=extracted.text, source="vlm_ocr"))

        return pages

    def _open(self, pdf_source: str | Path) -> pdfium.PdfDocument:
        source_str = str(pdf_source)
        if is_url(source_str):
            self._log(f"Fetching PDF from URL: {source_str}")
            pdf_bytes = fetch_pdf_bytes(
                source_str, max_bytes=self.max_download_bytes, timeout=self.fetch_timeout
            )
            return pdfium.PdfDocument(pdf_bytes)
        return pdfium.PdfDocument(source_str)

    def _render_page_b64(self, page: "pdfium.PdfPage") -> str:
        bitmap = page.render(scale=self.render_scale)
        image = bitmap.to_pil()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def _call_vlm(self, image_b64: str, page_num: int, total_pages: int) -> ExtractedPage:
        prompt = OCR_PROMPT_TEMPLATE.format(page_num=page_num, total_pages=total_pages)
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=8000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            output_format=ExtractedPage,
        )
        return response.parsed_output


def build_context(pages: List[PageContent]) -> str:
    """Собирает извлечённые страницы в единый текстовый контекст документа."""
    return "\n\n".join(
        f"--- Страница {p.page} (источник: {p.source}) ---\n{p.text}" for p in pages
    )
