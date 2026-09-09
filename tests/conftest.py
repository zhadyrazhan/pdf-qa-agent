import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fpdf import FPDF


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """Simple two-page text-based PDF with a real text layer."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.multi_cell(0, 10, "Annual Report 2024\nTotal revenue: 1,234,567 USD.\nNet profit: 89,012 USD.")
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.multi_cell(0, 10, "Page two.\nThe company was founded in 1998 by Jane Doe.")
    out = tmp_path / "report.pdf"
    pdf.output(str(out))
    return out


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    """Single blank (no text) page PDF — simulates a scan."""
    pdf = FPDF()
    pdf.add_page()
    out = tmp_path / "scan.pdf"
    pdf.output(str(out))
    return out
