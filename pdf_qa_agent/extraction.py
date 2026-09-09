"""Extracts text from a PDF: text layer first, VLM OCR fallback for scans.

Per page: try the PDF's text layer (pypdfium2) first — cheap and accurate for
text-based PDFs. If it's missing or shorter than TEXT_LAYER_MIN_CHARS (a scan or
a table rendered as an image), render the page to PNG and OCR it with an OpenAI
vision model, preserving document structure as markdown.

VLM calls are wrapped in retry/backoff (pdf_qa_agent.retry) since they're the
only network operation in the extraction pipeline.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Callable, List

from openai import OpenAI
import pypdfium2 as pdfium

from pdf_qa_agent.fetch import DEFAULT_FETCH_TIMEOUT, DEFAULT_MAX_PDF_BYTES, fetch_pdf_bytes, is_url
from pdf_qa_agent.retry import RetryConfig, with_retry
from pdf_qa_agent.schemas import ExtractedPage, PageContent

DEFAULT_MODEL = "gpt-4o-mini"
TEXT_LAYER_MIN_CHARS = 20  # below this, a page is treated as a scan with no text layer
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
    """Extracts PDF content page by page: text layer or VLM OCR."""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = DEFAULT_MODEL,
        text_layer_min_chars: int = TEXT_LAYER_MIN_CHARS,
        render_scale: float = DEFAULT_RENDER_SCALE,
        retry_config: RetryConfig | None = None,
        log: Callable[[str], None] | None = None,
        max_download_bytes: int = DEFAULT_MAX_PDF_BYTES,
        fetch_timeout: float = DEFAULT_FETCH_TIMEOUT,
    ):
        self.client = client or OpenAI()
        self.model = model
        self.text_layer_min_chars = text_layer_min_chars
        self.render_scale = render_scale
        self._log = log or (lambda msg: print(msg, file=sys.stderr))
        self.max_download_bytes = max_download_bytes
        self.fetch_timeout = fetch_timeout
        # Wrapped once here so retry_config stays configurable per-instance
        # (e.g. in tests, with sleep_fn as a no-op).
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
                try:
                    extracted = self._call_vlm_with_retry(image_b64, page_num, total_pages)
                    pages.append(PageContent(page=page_num, text=extracted.text, source="vlm_ocr"))
                except Exception as exc:
                    # A single page's OCR call (rate limit exhausted, timeout, malformed
                    # model output) shouldn't discard every other page already extracted —
                    # mark this page as failed and keep going.
                    self._log(f"  page {page_num}: VLM OCR failed, marking page as failed: {exc!r}")
                    pages.append(
                        PageContent(page=page_num, text=f"[OCR failed for this page: {exc}]", source="ocr_failed")
                    )

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
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            response_format=ExtractedPage,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured OCR result")
        return parsed


def build_context(pages: List[PageContent]) -> str:
    """Joins extracted pages into a single document text context."""
    return "\n\n".join(
        f"--- Страница {p.page} (источник: {p.source}) ---\n{p.text}" for p in pages
    )
