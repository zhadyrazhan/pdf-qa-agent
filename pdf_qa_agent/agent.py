"""PDF-QA-Agent answers questions grounded in a PDF document's content."""
from __future__ import annotations

from pathlib import Path
from typing import List

from openai import OpenAI

from pdf_qa_agent.extraction import DEFAULT_MODEL, PDFExtractor, build_context
from pdf_qa_agent.retry import RetryConfig, with_retry
from pdf_qa_agent.schemas import AgentAnswer, PageContent

SYSTEM_PROMPT = (
    "Ты — интеллектуальный агент, отвечающий на вопросы пользователя строго на основе "
    "содержимого предоставленного PDF-документа (эссе, научная статья или отчёт). Текст "
    "документа ниже уже извлечён: часть страниц — из текстового слоя PDF, часть — "
    "распознана OCR-моделью со сканов. Отвечай, опираясь ТОЛЬКО на этот текст. "
    "Если информации для ответа в документе недостаточно, поставь answerable=false "
    "и явно скажи об этом в answer, не придумывай факты. Для каждого использованного "
    "факта указывай номер страницы и короткую дословную цитату в citations."
)


class PDFQAAgent:
    """Extracts a PDF's content (text layer or VLM OCR) and answers questions about it."""

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = DEFAULT_MODEL,
        retry_config: RetryConfig | None = None,
        extractor: PDFExtractor | None = None,
    ):
        self.client = client or OpenAI()
        self.model = model
        self.extractor = extractor or PDFExtractor(
            client=self.client, model=model, retry_config=retry_config
        )
        self.pages: List[PageContent] = []
        # Wrapped once here so retry_config stays configurable (e.g. in tests,
        # skipping real delays via sleep_fn).
        self._ask_with_retry = with_retry(retry_config)(self._ask_impl)

    def load(self, pdf_path: str | Path) -> List[PageContent]:
        self.pages = self.extractor.extract(pdf_path)
        return self.pages

    def build_context(self) -> str:
        return build_context(self.pages)

    def ask(self, question: str) -> AgentAnswer:
        if not self.pages:
            raise RuntimeError("No document loaded: call agent.load(pdf_path) first.")
        return self._ask_with_retry(question)

    def _ask_impl(self, question: str) -> AgentAnswer:
        user_prompt = f"Документ:\n{self.build_context()}\n\nВопрос пользователя: {question}"
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=AgentAnswer,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured answer")
        return parsed
