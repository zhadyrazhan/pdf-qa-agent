"""Pydantic schemas used by the agent for extraction and structured answers."""
from typing import List, Literal

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """VLM OCR result for a single scanned page."""

    text: str = Field(
        description=(
            "Full page text, preserving structure as markdown "
            "(headings as '# ...', paragraphs, lists, table/figure captions)"
        )
    )


class PageContent(BaseModel):
    """A single page's extracted content, regardless of source."""

    page: int
    text: str
    source: Literal["text_layer", "vlm_ocr", "ocr_failed"]


class Citation(BaseModel):
    page: int = Field(description="Source page number in the document")
    excerpt: str = Field(description="Short verbatim quote from the document supporting the answer")


class AgentAnswer(BaseModel):
    answer: str = Field(description="Answer to the user's question based on the document")
    answerable: bool = Field(
        description="True if the document has enough information to answer, otherwise False"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Pages and quotes from the document used to support the answer",
    )
