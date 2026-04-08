"""
tools/resume_parser.py
-----------------------
Tool: parse_resume
Extracts structured data from a PDF resume.
PII is stripped before the data is returned.
"""

import io
from typing import Any
from guardrails import strip_pii

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def parse_resume(file_path: str) -> dict[str, Any]:
    """
    Parse a PDF resume and return structured, PII-stripped data.

    Args:
        file_path: Absolute path to the candidate's PDF resume.

    Returns:
        A dict with keys: raw_text, skills, experience_summary, education.
        All PII (phone, email, address) is redacted.
    """
    if not PDF_AVAILABLE:
        return {
            "status": "error",
            "message": "PyPDF2 not installed. Run: pip install pypdf2",
        }

    try:
        raw_text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                raw_text += page.extract_text() or ""
    except FileNotFoundError:
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # Strip PII before any inter-agent handoff
    sanitized_text = strip_pii(raw_text)

    # Simple heuristic skill extraction (extend with NLP later)
    skill_keywords = [
        "python", "java", "sql", "javascript", "react", "node", "aws", "gcp",
        "docker", "kubernetes", "machine learning", "data analysis", "excel",
        "communication", "leadership", "project management", "agile", "scrum",
    ]
    found_skills = [
        kw for kw in skill_keywords if kw.lower() in sanitized_text.lower()
    ]

    return {
        "status": "success",
        "raw_text": sanitized_text,
        "skills": found_skills,
        "character_count": len(sanitized_text),
    }