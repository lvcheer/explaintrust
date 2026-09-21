import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "docs/explaintrust-technical-brief.md"
PDF = ROOT / "output/pdf/explaintrust-technical-brief.pdf"


def test_technical_brief_pdf_matches_source_and_has_two_pages():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest().encode()
    payload = PDF.read_bytes()

    assert payload.startswith(b"%PDF-")
    assert b"source-sha256:" + digest in payload
    assert b"/Count 2" in payload
