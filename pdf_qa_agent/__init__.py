"""PDF-QA-Agent answers questions about a PDF document's content.

Handles both text-based PDFs (text layer) and scans (VLM OCR fallback).
"""
from dotenv import load_dotenv

load_dotenv()

from pdf_qa_agent.agent import PDFQAAgent
from pdf_qa_agent.schemas import AgentAnswer, Citation, ExtractedPage, PageContent

__all__ = [
    "PDFQAAgent",
    "AgentAnswer",
    "Citation",
    "ExtractedPage",
    "PageContent",
]
