"""CLI для PDF Q&A агента.

Использование:
    python3 -m pdf_qa_agent.cli <путь_к_pdf_или_URL> -q "Вопрос?"
    python3 -m pdf_qa_agent.cli <путь_к_pdf_или_URL>              # интерактивный режим
    python3 -m pdf_qa_agent.cli https://example.com/report.pdf -q "О чём отчёт?"
"""
import argparse
import json
import sys
from pathlib import Path

from pdf_qa_agent.agent import PDFQAAgent
from pdf_qa_agent.extraction import DEFAULT_MODEL
from pdf_qa_agent.fetch import PDFFetchError, PDFTooLargeError, UnsafeURLError, is_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Интеллектуальный агент для ответов на вопросы по PDF (текст или скан, локальный файл или URL)."
    )
    parser.add_argument("pdf_source", help="Путь к PDF-документу или http(s) URL")
    parser.add_argument("-q", "--question", help="Разовый вопрос (без него — интерактивный режим)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Модель Claude (по умолчанию {DEFAULT_MODEL})")
    args = parser.parse_args()

    source = args.pdf_source
    if not is_url(source) and not Path(source).exists():
        print(f"Error: file not found: {source}", file=sys.stderr)
        sys.exit(1)

    agent = PDFQAAgent(model=args.model)

    print(f"Loading and extracting: {source}", file=sys.stderr)
    try:
        agent.load(source)
    except (UnsafeURLError, PDFTooLargeError, PDFFetchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    n_text = sum(1 for p in agent.pages if p.source == "text_layer")
    n_vlm = sum(1 for p in agent.pages if p.source == "vlm_ocr")
    print(
        f"Extracted {len(agent.pages)} pages ({n_text} text-layer, {n_vlm} via VLM OCR)",
        file=sys.stderr,
    )

    if args.question:
        result = agent.ask(args.question)
        print(json.dumps({"question": args.question, **result.model_dump()}, indent=2, ensure_ascii=False))
        return

    print("Интерактивный режим. Введите вопрос (или 'exit' для выхода).", file=sys.stderr)
    while True:
        try:
            question = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit", "выход"):
            break
        result = agent.ask(question)
        print(json.dumps({"question": question, **result.model_dump()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
