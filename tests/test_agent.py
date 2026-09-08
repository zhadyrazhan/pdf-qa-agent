"""Тесты PDFQAAgent: сборка контекста, вызов модели, retry при сбоях API —
всё с замоканным клиентом Anthropic, без сети."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from pdf_qa_agent.agent import PDFQAAgent
from pdf_qa_agent.retry import RetryConfig, RetryExhaustedError
from pdf_qa_agent.schemas import AgentAnswer, Citation, PageContent


def _mock_client_returning(answer: AgentAnswer) -> MagicMock:
    client = MagicMock()
    client.messages.parse.return_value = SimpleNamespace(parsed_output=answer)
    return client


def _agent_with_loaded_pages(client: MagicMock, retry_config: RetryConfig | None = None) -> PDFQAAgent:
    agent = PDFQAAgent(client=client, retry_config=retry_config)
    agent.pages = [
        PageContent(page=1, text="Revenue was 1,234,567 USD in 2024.", source="text_layer"),
        PageContent(page=2, text="Founded in 1998 by Jane Doe.", source="vlm_ocr"),
    ]
    return agent


def test_ask_raises_if_document_not_loaded():
    agent = PDFQAAgent(client=MagicMock())
    with pytest.raises(RuntimeError):
        agent.ask("What is the revenue?")


def test_ask_returns_structured_answer_with_citations():
    expected = AgentAnswer(
        answer="Revenue was 1,234,567 USD.",
        answerable=True,
        citations=[Citation(page=1, excerpt="Revenue was 1,234,567 USD in 2024.")],
    )
    client = _mock_client_returning(expected)
    agent = _agent_with_loaded_pages(client)

    result = agent.ask("What was the revenue?")

    assert result.answerable is True
    assert result.citations[0].page == 1
    client.messages.parse.assert_called_once()


def test_ask_sends_full_context_and_question_to_model():
    client = _mock_client_returning(AgentAnswer(answer="x", answerable=True))
    agent = _agent_with_loaded_pages(client)

    agent.ask("Who founded the company?")

    call_kwargs = client.messages.parse.call_args.kwargs
    user_message = call_kwargs["messages"][0]["content"]
    assert "Jane Doe" in user_message
    assert "1,234,567" in user_message
    assert "Who founded the company?" in user_message
    assert call_kwargs["output_format"] is AgentAnswer


def test_ask_marks_unanswerable_when_model_says_so():
    expected = AgentAnswer(answer="Информации в документе недостаточно.", answerable=False, citations=[])
    client = _mock_client_returning(expected)
    agent = _agent_with_loaded_pages(client)

    result = agent.ask("What is the CEO's salary in 2030?")

    assert result.answerable is False
    assert result.citations == []


def test_ask_retries_on_transient_error_then_succeeds():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    expected = AgentAnswer(answer="ok", answerable=True)
    client = MagicMock()
    client.messages.parse.side_effect = [
        anthropic.APIConnectionError(request=request),
        SimpleNamespace(parsed_output=expected),
    ]
    agent = _agent_with_loaded_pages(client, retry_config=RetryConfig(max_retries=3, base_delay=0.001))

    result = agent.ask("question")

    assert result.answerable is True
    assert client.messages.parse.call_count == 2


def test_ask_raises_retry_exhausted_after_persistent_failures():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = MagicMock()
    client.messages.parse.side_effect = anthropic.APIConnectionError(request=request)
    agent = _agent_with_loaded_pages(client, retry_config=RetryConfig(max_retries=2, base_delay=0.001))

    with pytest.raises(RetryExhaustedError):
        agent.ask("question")

    assert client.messages.parse.call_count == 2
