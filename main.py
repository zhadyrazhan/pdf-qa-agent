"""HTTP API for uploading PDFs and asking grounded questions about them.

Run with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from pdf_qa_agent.agent import PDFQAAgent
from pdf_qa_agent.extraction import DEFAULT_MODEL, PDFExtractor
from pdf_qa_agent.fetch import PDF_MAGIC, PDFFetchError, PDFTooLargeError, UnsafeURLError
from pdf_qa_agent.schemas import AgentAnswer

load_dotenv()

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 1 * 1024 * 1024
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",")
    if origin.strip()
]
# Browsers commonly resolve localhost and 127.0.0.1 differently. Keep both
# available for direct API access; the Next rewrite still avoids CORS in normal use.
for local_origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
    if local_origin not in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.append(local_origin)


@dataclass
class DocumentSession:
    name: str
    agent: PDFQAAgent


class CitationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page: int
    excerpt: str


class PageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page: int
    text: str
    source: str


class DocumentResponse(BaseModel):
    id: str
    name: str
    pages: list[PageResponse]
    text_layer_pages: int
    ocr_pages: int


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DocumentFromURLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000, description="Direct http(s) link to a PDF file")


class AnswerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    answer: str
    answerable: bool
    citations: list[CitationResponse]


app = FastAPI(title="PDF Q&A Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

documents: dict[str, DocumentSession] = {}


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


def _parse_pdf(source: str | Path, model: str) -> PDFQAAgent:
    client = _client()
    # max_download_bytes keeps the URL path capped at the same size as file uploads
    # (MAX_UPLOAD_BYTES); local paths ignore this since they don't go through fetch_pdf_bytes.
    extractor = PDFExtractor(client=client, model=model, max_download_bytes=MAX_UPLOAD_BYTES)
    agent = PDFQAAgent(client=client, model=model, extractor=extractor)
    agent.load(source)
    return agent


def _document_name_from_url(url: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1]
    return name or url


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/documents", response_model=DocumentResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF document to parse")],
) -> DocumentResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            size = 0
            header = b""
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File needs to be less than 1 MB")
                if not header:
                    header = chunk[: len(PDF_MAGIC)]
                temp_file.write(chunk)

            if not header.startswith(PDF_MAGIC):
                raise HTTPException(status_code=415, detail="Uploaded file is not a PDF")

        agent = await run_in_threadpool(_parse_pdf, temp_path, os.getenv("PDF_QA_MODEL", DEFAULT_MODEL))
        document_id = uuid.uuid4().hex
        documents[document_id] = DocumentSession(name=filename, agent=agent)
        pages = [PageResponse.model_validate(page) for page in agent.pages]
        return DocumentResponse(
            id=document_id,
            name=filename,
            pages=pages,
            text_layer_pages=sum(page.source == "text_layer" for page in agent.pages),
            ocr_pages=sum(page.source == "vlm_ocr" for page in agent.pages),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to parse uploaded PDF %r", filename)
        raise HTTPException(status_code=422, detail="Could not parse PDF") from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        await file.close()


@app.post("/api/documents/from-url", response_model=DocumentResponse)
async def upload_document_from_url(request: DocumentFromURLRequest) -> DocumentResponse:
    """Fetches a PDF from a user-supplied URL and parses it.

    This is a server-side fetch of an attacker-controllable URL — the classic SSRF
    shape — so pdf_qa_agent.fetch enforces scheme/host allowlisting (public IPs only,
    checked again after redirects) and a size cap before any bytes reach pdfium. See
    pdf_qa_agent/fetch.py for the full threat model and known limitations.
    """
    try:
        agent = await run_in_threadpool(
            _parse_pdf, request.url, os.getenv("PDF_QA_MODEL", DEFAULT_MODEL)
        )
    except UnsafeURLError:
        raise HTTPException(status_code=400, detail="URL must resolve to a public HTTP(S) address")
    except PDFTooLargeError:
        raise HTTPException(status_code=413, detail="File needs to be less than 1 MB")
    except PDFFetchError:
        raise HTTPException(status_code=422, detail="Could not fetch a PDF from that URL")
    except Exception:
        logger.exception("Failed to parse PDF from URL %r", request.url)
        raise HTTPException(status_code=422, detail="Could not parse PDF from that URL")

    document_id = uuid.uuid4().hex
    name = _document_name_from_url(request.url)
    documents[document_id] = DocumentSession(name=name, agent=agent)
    pages = [PageResponse.model_validate(page) for page in agent.pages]
    return DocumentResponse(
        id=document_id,
        name=name,
        pages=pages,
        text_layer_pages=sum(page.source == "text_layer" for page in agent.pages),
        ocr_pages=sum(page.source == "vlm_ocr" for page in agent.pages),
    )


@app.post("/api/documents/{document_id}/questions", response_model=AnswerResponse)
async def ask_question(document_id: str, request: QuestionRequest) -> AnswerResponse:
    session = documents.get(document_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Document not found or server session expired")
    try:
        result: AgentAnswer = await run_in_threadpool(session.agent.ask, request.question)
        return AnswerResponse.model_validate(result)
    except Exception as exc:
        logger.exception("Failed to answer question for document %r", document_id)
        raise HTTPException(status_code=502, detail="Could not answer the question") from exc
