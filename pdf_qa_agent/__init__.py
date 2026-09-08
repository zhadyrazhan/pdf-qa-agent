"""PDF Q&A Agent — отвечает на вопросы пользователя по содержимому PDF-документа.

Работает как с text-based PDF (текстовый слой), так и со сканами (VLM OCR fallback).
"""
from pdf_qa_agent.agent import PDFQAAgent
from pdf_qa_agent.schemas import AgentAnswer, Citation, ExtractedPage, PageContent

__all__ = [
    "PDFQAAgent",
    "AgentAnswer",
    "Citation",
    "ExtractedPage",
    "PageContent",
]
