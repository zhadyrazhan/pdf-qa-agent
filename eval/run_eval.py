"""Runs PDF-QA-Agent against the golden set and reports quality metrics.

The golden set (goldenset.json) is a set of questions against three real
documents of varying quality (a text-based report, a scanned book, a scan with
complex diagrams), each with an expected answerable flag, keywords expected in
the answer, and pages the agent should cite.

Usage:
    python3 eval/run_eval.py                       # full golden set
    python3 eval/run_eval.py --ids book-author,charts-fig7-subject
    python3 eval/run_eval.py --pdf-root /custom/path/to/pdf-files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdf_qa_agent.agent import PDFQAAgent  # noqa: E402
from pdf_qa_agent.extraction import DEFAULT_MODEL  # noqa: E402
from pdf_qa_agent.retry import RetryConfig  # noqa: E402
from pdf_qa_agent.schemas import AgentAnswer  # noqa: E402

GOLDENSET_PATH = Path(__file__).parent / "goldenset.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"
# The golden-set paths are relative to the pdf-files directory shipped alongside the project.
DEFAULT_PDF_ROOT = Path(__file__).resolve().parents[1] / "pdf-files"


def _normalize(text: str) -> str:
    """Normalizes text for comparison: lowercase, no spaces/dots/commas.

    Lets numbers match regardless of thousands-separator style
    (1.529.951.443 / 1 529 951 443 / 1529951443) and ignores word casing.
    """
    return re.sub(r"[\s.,]", "", text.strip().lower())


def _keyword_hit(answer_text: str, expected_keywords: List[str]) -> bool:
    if not expected_keywords:
        return True
    normalized_answer = _normalize(answer_text)
    return any(_normalize(kw) in normalized_answer for kw in expected_keywords)


def _citation_hit(result: AgentAnswer, expected_pages: List[int]) -> bool:
    if not expected_pages:
        return True
    cited_pages = {c.page for c in result.citations}
    return bool(cited_pages & set(expected_pages))


def load_goldenset(path: Path, ids_filter: List[str] | None) -> List[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if ids_filter:
        items = [it for it in items if it["id"] in ids_filter]
    return items


def run_eval(
    goldenset: List[dict],
    pdf_root: Path,
    model: str,
    max_retries: int,
) -> dict:
    retry_config = RetryConfig(max_retries=max_retries)
    agents_by_pdf: Dict[str, PDFQAAgent] = {}
    results = []

    for item in goldenset:
        pdf_rel = item["pdf"]
        pdf_path = pdf_root / pdf_rel

        if pdf_rel not in agents_by_pdf:
            print(f"[load] {pdf_rel}", file=sys.stderr)
            if not pdf_path.exists():
                print(f"  !! file not found: {pdf_path} -- skipping all questions for it", file=sys.stderr)
                agents_by_pdf[pdf_rel] = None
            else:
                agent = PDFQAAgent(model=model, retry_config=retry_config)
                t0 = time.time()
                agent.load(pdf_path)
                n_text = sum(1 for p in agent.pages if p.source == "text_layer")
                n_vlm = sum(1 for p in agent.pages if p.source == "vlm_ocr")
                print(
                    f"  extracted {len(agent.pages)} pages "
                    f"({n_text} text-layer, {n_vlm} VLM OCR) in {time.time() - t0:.1f}s",
                    file=sys.stderr,
                )
                agents_by_pdf[pdf_rel] = agent

        agent = agents_by_pdf[pdf_rel]
        if agent is None:
            results.append({**item, "error": "pdf_not_found"})
            continue

        print(f"[ask]  {item['id']}: {item['question']}", file=sys.stderr)
        try:
            answer = agent.ask(item["question"])
        except Exception as exc:  # noqa: BLE001 - record any run error in the report
            print(f"  !! error: {exc!r}", file=sys.stderr)
            results.append({**item, "error": repr(exc)})
            continue

        answerable_match = answer.answerable == item["expected_answerable"]
        keyword_ok = _keyword_hit(answer.answer, item["expected_keywords"]) if item["expected_answerable"] else True
        citation_ok = _citation_hit(answer, item["expected_pages"]) if item["expected_answerable"] else True
        passed = answerable_match and keyword_ok

        results.append(
            {
                **item,
                "actual_answer": answer.answer,
                "actual_answerable": answer.answerable,
                "actual_citations": [c.model_dump() for c in answer.citations],
                "answerable_match": answerable_match,
                "keyword_hit": keyword_ok,
                "citation_hit": citation_ok,
                "passed": passed,
            }
        )
        print(f"  -> answerable={answer.answerable} passed={passed}", file=sys.stderr)

    return summarize(results)


def summarize(results: List[dict]) -> dict:
    total = len(results)
    scored = [r for r in results if "error" not in r]
    errored = total - len(scored)

    passed = sum(1 for r in scored if r["passed"])
    answerable_correct = sum(1 for r in scored if r["answerable_match"])

    answerable_expected = [r for r in scored if r["expected_answerable"]]
    keyword_correct = sum(1 for r in answerable_expected if r["keyword_hit"])
    with_expected_pages = [r for r in answerable_expected if r["expected_pages"]]
    citation_correct = sum(1 for r in with_expected_pages if r["citation_hit"])

    summary = {
        "total_questions": total,
        "errored": errored,
        "scored": len(scored),
        "passed": passed,
        "pass_rate": round(passed / len(scored), 3) if scored else None,
        "answerable_accuracy": round(answerable_correct / len(scored), 3) if scored else None,
        "keyword_accuracy": (
            round(keyword_correct / len(answerable_expected), 3) if answerable_expected else None
        ),
        "citation_accuracy": (
            round(citation_correct / len(with_expected_pages), 3) if with_expected_pages else None
        ),
    }
    return {"summary": summary, "results": results}


def print_report(report: dict) -> None:
    summary = report["summary"]
    print("\n=== Eval summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n=== Per-question ===")
    for r in report["results"]:
        if "error" in r:
            print(f"  [ERR ] {r['id']}: {r['error']}")
            continue
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['id']} (answerable_match={r['answerable_match']}, keyword_hit={r['keyword_hit']}, citation_hit={r['citation_hit']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Runs PDF-QA-Agent against the golden set.")
    parser.add_argument("--goldenset", type=Path, default=GOLDENSET_PATH)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT, help="Base directory for relative PDF paths in the golden set")
    parser.add_argument("--ids", help="Comma-separated list of ids to run only those")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    ids_filter = args.ids.split(",") if args.ids else None
    goldenset = load_goldenset(args.goldenset, ids_filter)
    if not goldenset:
        print("No golden set items matched — nothing to run.", file=sys.stderr)
        sys.exit(1)

    report = run_eval(goldenset, args.pdf_root, args.model, args.max_retries)

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to: {args.out}", file=sys.stderr)

    print_report(report)

    if report["summary"]["pass_rate"] is not None and report["summary"]["pass_rate"] < 1.0:
        sys.exit(2)


if __name__ == "__main__":
    main()
