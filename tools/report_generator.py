"""
tools/report_generator.py
--------------------------
Tool: generate_report
Writes a structured performance report to a PDF file.
Called by the Verifier/Critic agent after validation passes.
"""

import os
import datetime
from typing import Any
from observability import log_tool_call

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_report(
    candidate_name: str,
    job_role: str,
    company: str,
    evaluated_skills: dict[str, str],
    transcript_summary: str,
    guardrail_flags: list[str],
    output_dir: str = "/tmp",
    tool_context=None,
) -> dict[str, Any]:
    """
    Generate a PDF performance report for the candidate.

    Args:
        candidate_name: Candidate's name (from sanitized resume data).
        job_role: Target job role.
        company: Target company.
        evaluated_skills: Dict of skill → rating (strong/adequate/needs_improvement).
        transcript_summary: A brief summary of the interview performance.
        guardrail_flags: Any flags raised during the session.
        output_dir: Directory to write the PDF file.

    Returns:
        Dict with status and the path to the generated PDF.
    """
    try:
        if not REPORTLAB_AVAILABLE:
            result = {
                "status": "error",
                "message": "reportlab not installed. Run: pip install reportlab",
            }
            if tool_context:
                log_tool_call(tool_context.state, "verifier_critic", "generate_report", {"job_role": job_role, "company": company}, result)
            return result

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_report_{timestamp}.pdf"
        output_path = os.path.join(output_dir, filename)

        doc = SimpleDocTemplate(output_path, pagesize=LETTER,
                                rightMargin=inch, leftMargin=inch,
                                topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("<font color='darkblue'>Interview Performance Report</font>", styles["Title"]))
        story.append(Spacer(1, 0.2 * inch))

        # Metadata
        meta_lines = [
            f"<b>Candidate:</b> {candidate_name}",
            f"<b>Target Role:</b> {job_role} at {company}",
            f"<b>Date:</b> {datetime.datetime.utcnow().strftime('%B %d, %Y')}",
        ]
        for line in meta_lines:
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Skills table
        story.append(Paragraph("<font color='darkblue'>Skill Evaluation</font>", styles["Heading2"]))
        table_data = [["Skill", "Rating"]]
        rating_colors = {
            "strong": colors.lightgreen,
            "adequate": colors.lightyellow,
            "needs_improvement": colors.salmon,
        }
        for skill, rating in evaluated_skills.items():
            table_data.append([skill, rating.replace("_", " ").title()])

        table = Table(table_data, colWidths=[3.5 * inch, 2 * inch])
        table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
        table.setStyle(table_style)
        story.append(table)
        story.append(Spacer(1, 0.3 * inch))

        # Summary
        story.append(Paragraph("<font color='darkblue'>Performance Summary</font>", styles["Heading2"]))
        story.append(Paragraph(transcript_summary, styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Guardrail flags (if any)
        if guardrail_flags:
            story.append(Paragraph("<font color='darkblue'>Session Flags</font>", styles["Heading2"]))
            for flag in guardrail_flags:
                story.append(Paragraph(f"⚠ {flag}", styles["Normal"]))

        doc.build(story)

        result = {
            "status": "success",
            "report_path": output_path,
            "filename": filename,
        }
        if tool_context:
            log_tool_call(
                tool_context.state,
                "verifier_critic",
                "generate_report",
                {"job_role": job_role, "company": company, "skill_count": len(evaluated_skills)},
                {"status": "success", "report_path": output_path},
            )
        return result

    except Exception as e:
        result = {
            "status": "error",
            "message": f"Failed to generate report: {str(e)}",
        }
        if tool_context:
            log_tool_call(tool_context.state, "verifier_critic", "generate_report", {"job_role": job_role, "company": company}, result)
        return result
