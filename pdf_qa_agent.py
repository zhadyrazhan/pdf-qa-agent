"""
PDF Q&A Agent 

This script provides a command-line interface for interacting with a PDF Q&A agent.
It allows users to ask questions about the content of PDF documents and receive answers
based on the extracted text and structure of the PDFs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pdf_qa_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
