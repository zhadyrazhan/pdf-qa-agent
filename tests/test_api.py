"""Tests for the FastAPI endpoints (main.py): file upload, URL upload, Q&A.

Fully offline: `_parse_pdf` (the only place that calls the OpenAI API and
pdf_qa_agent.fetch) is replaced via monkeypatch with a fake agent — these tests
cover routing, request validation, and error-to-HTTP-status mapping, not the
extraction/OCR logic itself (already covered by test_extraction.py and
test_fetch.py)."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main as api_main
from pdf_qa_agent.fetch import PDFFetchError, PDFTooLargeError, UnsafeURLError
from pdf_qa_agent.schemas import AgentAnswer, Citation, PageContent

MINIMAL_PDF = b"%PDF-1.4\n%mock\n%%EOF"


class FakeAgent:
    def __init__(self, pages, answer: AgentAnswer | None = None):
        self.pages = pages
        self._answer = answer or AgentAnswer(answer="mock answer", answerable=True)

    def ask(self, question: str) -> AgentAnswer:
        return self._answer


@pytest.fixture
def client():
    api_main.documents.clear()
    return TestClient(api_main.app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document_rejects_non_pdf_filename(client):
    response = client.post(
        "/api/documents", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_upload_document_rejects_non_pdf_content(client):
    response = client.post(
        "/api/documents", files={"file": ("report.pdf", b"not a pdf", "application/pdf")}
    )
    assert response.status_code == 415


def test_upload_document_success(client, monkeypatch):
    fake_pages = [
        PageContent(page=1, text="Hello", source="text_layer"),
        PageContent(page=2, text="World", source="vlm_ocr"),
    ]
    monkeypatch.setattr(api_main, "_parse_pdf", lambda source, model: FakeAgent(fake_pages))

    response = client.post(
        "/api/documents", files={"file": ("report.pdf", MINIMAL_PDF, "application/pdf")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "report.pdf"
    assert len(body["pages"]) == 2
    assert body["text_layer_pages"] == 1
    assert body["ocr_pages"] == 1
    assert body["id"] in api_main.documents


def test_upload_document_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 10)
    response = client.post(
        "/api/documents",
        files={"file": ("report.pdf", MINIMAL_PDF, "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "File needs to be less than 1 MB"


def test_upload_from_url_success(client, monkeypatch):
    fake_pages = [PageContent(page=1, text="Remote content", source="text_layer")]
    calls = {}

    def fake_parse_pdf(source, model):
        calls["source"] = source
        return FakeAgent(fake_pages)

    monkeypatch.setattr(api_main, "_parse_pdf", fake_parse_pdf)

    response = client.post(
        "/api/documents/from-url", json={"url": "https://example.com/reports/q3.pdf"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "q3.pdf"
    assert body["text_layer_pages"] == 1
    assert calls["source"] == "https://example.com/reports/q3.pdf"


def test_upload_from_url_rejects_unsafe_url(client, monkeypatch):
    def _raise(source, model):
        raise UnsafeURLError("blocked: resolves to a private address")

    monkeypatch.setattr(api_main, "_parse_pdf", _raise)

    response = client.post(
        "/api/documents/from-url", json={"url": "http://169.254.169.254/meta"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "URL must resolve to a public HTTP(S) address"


def test_upload_from_url_rejects_oversized_pdf(client, monkeypatch):
    def _raise(source, model):
        raise PDFTooLargeError("PDF exceeds max size")

    monkeypatch.setattr(api_main, "_parse_pdf", _raise)

    response = client.post("/api/documents/from-url", json={"url": "https://example.com/big.pdf"})

    assert response.status_code == 413
    assert response.json()["detail"] == "File needs to be less than 1 MB"


def test_upload_from_url_maps_fetch_error_to_422(client, monkeypatch):
    def _raise(source, model):
        raise PDFFetchError("Network error fetching https://example.com/missing.pdf")

    monkeypatch.setattr(api_main, "_parse_pdf", _raise)

    response = client.post("/api/documents/from-url", json={"url": "https://example.com/missing.pdf"})

    assert response.status_code == 422


def test_upload_from_url_rejects_empty_url(client):
    response = client.post("/api/documents/from-url", json={"url": ""})
    assert response.status_code == 422  # pydantic min_length validation


def test_ask_question_returns_404_for_unknown_document(client):
    response = client.post("/api/documents/does-not-exist/questions", json={"question": "hi?"})
    assert response.status_code == 404


def test_ask_question_success(client, monkeypatch):
    fake_pages = [PageContent(page=1, text="x", source="text_layer")]
    answer = AgentAnswer(
        answer="42",
        answerable=True,
        citations=[Citation(page=1, excerpt="the answer is 42")],
    )
    monkeypatch.setattr(api_main, "_parse_pdf", lambda source, model: FakeAgent(fake_pages, answer))

    upload = client.post(
        "/api/documents", files={"file": ("report.pdf", MINIMAL_PDF, "application/pdf")}
    )
    document_id = upload.json()["id"]

    response = client.post(f"/api/documents/{document_id}/questions", json={"question": "what is it?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "42"
    assert body["citations"][0]["page"] == 1
