"""
tools/resume_parser.py
-----------------------
Extracts raw text and strips PII. Structured extraction is handled by the LLM.
"""

import os
from typing import Any
from guardrails import strip_pii
from observability import log_tool_call

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def parse_resume(file_path: str, tool_context=None) -> dict[str, Any]:
    """
    Parse a PDF resume and return sanitized raw text.
    """
    if not PDF_AVAILABLE:
        result = {
            "status": "error",
            "message": "PyPDF2 not installed. Run: pip install pypdf2",
        }
        if tool_context:
            log_tool_call(tool_context.state, "context_optimizer", "parse_resume", {"file_path": file_path}, result)
        return result

    if not os.path.exists(file_path):
        result = {"status": "error", "message": f"File not found: {file_path}"}
        if tool_context:
            log_tool_call(tool_context.state, "context_optimizer", "parse_resume", {"file_path": file_path}, result)
        return result

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
        result = {"status": "error", "message": f"PDF Extraction failed: {str(e)}"}
        if tool_context:
            log_tool_call(tool_context.state, "context_optimizer", "parse_resume", {"file_path": file_path}, result)
        return result

    if not raw_text.strip():
        result = {"status": "error", "message": "No text could be extracted from the PDF."}
        if tool_context:
            log_tool_call(tool_context.state, "context_optimizer", "parse_resume", {"file_path": file_path}, result)
        return result

    # Strip PII before the agent ever sees it
    sanitized_text = strip_pii(raw_text)

    result = {
        "status": "success",
        "raw_text": sanitized_text,
        "character_count": len(sanitized_text),
    }
    if tool_context:
        log_tool_call(
            tool_context.state,
            "context_optimizer",
            "parse_resume",
            {"file_path": file_path},
            {"status": "success", "character_count": len(sanitized_text)},
        )
    return result
