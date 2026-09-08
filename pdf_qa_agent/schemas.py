"""Pydantic-схемы, используемые агентом для извлечения и структурированных ответов."""
from typing import List, Literal

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Результат VLM OCR одной отсканированной страницы."""

    text: str = Field(
        description=(
            "Полный текст страницы с сохранением структуры в виде markdown "
            "(заголовки как '# ...', абзацы, списки, подписи таблиц/рисунков)"
        )
    )


class PageContent(BaseModel):
    """Извлечённое содержимое одной страницы документа, независимо от источника."""

    page: int
    text: str
    source: Literal["text_layer", "vlm_ocr"]


class Citation(BaseModel):
    page: int = Field(description="Номер страницы источника в документе")
    excerpt: str = Field(description="Короткая дословная цитата из документа, подтверждающая ответ")


class AgentAnswer(BaseModel):
    answer: str = Field(description="Ответ на вопрос пользователя на основе документа")
    answerable: bool = Field(
        description="True, если в документе достаточно информации для ответа, иначе False"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Страницы и цитаты из документа, использованные для ответа",
    )
