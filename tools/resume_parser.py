"""
tools/resume_parser.py
-----------------------
Extracts raw text and strips PII. Structured extraction is handled by the LLM.
"""

import os
from typing import Any
from guardrails import strip_pii

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def parse_resume(file_path: str) -> dict[str, Any]:
    """
    Parse a PDF resume and return sanitized raw text.
    """
    if not PDF_AVAILABLE:
        return {
            "status": "error",
            "message": "PyPDF2 not installed. Run: pip install pypdf2",
        }

    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    try:
        raw_text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Basic integrity check
            if len(reader.pages) == 0:
                 return {"status": "error", "message": "PDF file appears to be empty."}
                 
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
    except Exception as e:
        return {"status": "error", "message": f"PDF Extraction failed: {str(e)}"}

    if not raw_text.strip():
        return {"status": "error", "message": "No text could be extracted from the PDF."}

    # Strip PII before the agent ever sees it
    sanitized_text = strip_pii(raw_text)

    return {
        "status": "success",
        "raw_text": sanitized_text,
        "character_count": len(sanitized_text),
    }