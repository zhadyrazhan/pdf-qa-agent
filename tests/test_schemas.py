import pytest
from pydantic import ValidationError

from pdf_qa_agent.schemas import AgentAnswer, Citation, PageContent


def test_page_content_rejects_invalid_source():
    with pytest.raises(ValidationError):
        PageContent(page=1, text="hi", source="handwritten")  # noqa: not a valid Literal


def test_agent_answer_defaults_to_empty_citations():
    answer = AgentAnswer(answer="42", answerable=True)
    assert answer.citations == []


def test_citation_requires_page_and_excerpt():
    with pytest.raises(ValidationError):
        Citation(page=1)  # missing excerpt
