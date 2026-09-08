# PDF Q&A Agent

PDF Q&A Agent extracts text from PDF documents—including scanned pages through VLM OCR—and answers questions with structured, page-cited responses grounded in the document.

## How it works

For each PDF page, the agent:

1. Uses the PDF text layer when it contains enough text.
2. Renders pages with no usable text layer to PNG and sends them to Claude for structured OCR.
3. Combines all extracted pages into one context.
4. Sends the context and the user’s question to Claude, requiring an `AgentAnswer` response with:
   - `answer` — the response text;
   - `answerable` — whether the document contains enough information;
   - `citations` — source page numbers and direct excerpts.

Anthropic API calls use exponential backoff with jitter for connection errors, timeouts, rate limits, and temporary server errors.

## Project structure

```text
pdf_qa_agent/          Python package
  schemas.py           Pydantic response and extraction models
  retry.py             Retry and exponential-backoff utilities
  extraction.py        PDF text extraction and VLM OCR fallback
  agent.py             PDFQAAgent: load() -> ask()
  cli.py               Command-line interface

eval/
  goldenset.json       14 questions about 3 real documents
  run_eval.py          Golden-set evaluation runner
  eval_report.json     Generated evaluation report

pdf-files/             Source PDFs used by the golden set (gitignored)
tests/                 Offline pytest suite with mocked API calls
pdf_qa_agent.py        Backward-compatible CLI entry point
```

## Installation

```bash
cd /Users/zhadyrazhan/Documents/llm-engineer/pdf-qa-agent
python3 -m pip install -r requirements.txt
```

Set an Anthropic API key before using the agent:

```bash
export ANTHROPIC_API_KEY=...
```

The test suite is fully offline and does not require an API key.

## Usage

Ask one question:

```bash
python3 -m pdf_qa_agent.cli path/to/document.pdf \
  -q "What is this document about?"
```

Start interactive mode:

```bash
python3 -m pdf_qa_agent.cli path/to/document.pdf
```

The Python API can be used directly:

```python
from pdf_qa_agent import PDFQAAgent

agent = PDFQAAgent()
agent.load("report.pdf")
result = agent.ask("What was the company’s revenue in 2024?")

print(result.answer)
print(result.answerable)
print(result.citations)
```

## Tests

```bash
python3 -m pytest tests/ -v
```

The tests cover retry behavior, text-layer extraction, OCR fallback, context construction, structured agent responses, Pydantic validation, and golden-set integrity.

## Evaluation and golden set

The golden set in [`eval/goldenset.json`](eval/goldenset.json) contains 14 questions covering three real PDFs in [`pdf-files/`](pdf-files/):

| Document | Scenario | Coverage |
|---|---|---|
| `kaztelecom.pdf` | Financial report with both text and scanned pages | Mixed extraction and exact financial values |
| `G25_Central_Asia_reading_(1)-1-4.pdf` | Four-page scanned book excerpt | VLM OCR and English-text questions |
| `1_-_mu-stat-grafiki_-2002-10-11.pdf` | Two pages containing charts | Chart captions/data and unsupported questions |

Some questions are intentionally unanswerable. They verify that the agent returns `answerable=false` instead of inventing information.

Run the full evaluation:

```bash
python3 eval/run_eval.py
```

Run selected cases or use a custom PDF root:

```bash
python3 eval/run_eval.py --ids book-author,charts-fig7-subject
python3 eval/run_eval.py --pdf-root /custom/path/to/pdf-files
```

The runner writes `eval/eval_report.json` and reports:

- `answerable_accuracy` — correctness of the `answerable` decision;
- `keyword_accuracy` — whether an expected word or number appears in answerable responses;
- `citation_accuracy` — whether an expected source page was cited;
- `pass_rate` — the percentage passing answerability and keyword checks.

The command exits with a nonzero status when `pass_rate < 1.0`, so it can be used in CI.

## Limitations

- The complete document is placed into one model context; very large PDFs need chunking or retrieval.
- OCR quality depends on scan quality, especially for dense numeric tables.
- Encrypted, damaged, or unsupported PDFs may fail during loading.
