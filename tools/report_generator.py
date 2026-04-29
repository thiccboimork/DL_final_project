"""
tools/report_generator.py
--------------------------
Tool: generate_report
Writes a rich, graded performance report to a PDF file.
Called by the Verifier/Critic agent after validation passes.

Key design decisions
─────────────────────
- Every parameter has a safe default → the LLM can never crash this by
  omitting guardrail_flags or any of the new optional sections.
- guardrail_flags defaults to []  (the bug from the test run).
- strengths / work_on / expand_on / next_steps are optional; auto-generated
  from evaluated_skills when the LLM doesn't supply them.
- Overall letter grade + numeric score computed from skill ratings.
- datetime.UTC used instead of deprecated utcnow().
"""

import os
import datetime
from typing import Any
from observability import log_tool_call

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing, Rect
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

RATING_SCORE = {"strong": 100, "adequate": 65, "needs_improvement": 25}

def _letter_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B+"
    if score >= 70: return "B"
    if score >= 60: return "C+"
    if score >= 50: return "C"
    if score >= 40: return "D"
    return "F"

def _grade_verdict(score: float) -> str:
    if score >= 85: return "Outstanding — highly competitive for this role."
    if score >= 70: return "Strong candidate — minor gaps to address."
    if score >= 55: return "Promising — notable development areas remain."
    if score >= 40: return "Developing — significant preparation recommended."
    return "Not yet ready — substantial skill-building needed."

# Colours
C_HEADER   = colors.HexColor("#1A237E")   # deep indigo
C_SUBHDR   = colors.HexColor("#283593")
C_RULE     = colors.HexColor("#C5CAE9")
C_ALTROW   = colors.HexColor("#F5F5F5")
C_BAR_BG   = colors.HexColor("#E0E0E0")
C_GREEN    = colors.HexColor("#2E7D32")
C_AMBER    = colors.HexColor("#E65100")
C_RED      = colors.HexColor("#B71C1C")
RATING_COLOUR = {"strong": C_GREEN, "adequate": C_AMBER, "needs_improvement": C_RED}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _grade_bar(rating: str, w: float = 110, h: float = 9):
    """Small horizontal bar proportional to the skill rating."""
    pct = {"strong": 1.0, "adequate": 0.60, "needs_improvement": 0.22}.get(rating, 0.5)
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=C_BAR_BG, strokeColor=None))
    d.add(Rect(0, 0, w * pct, h,
               fillColor=RATING_COLOUR.get(rating, colors.grey), strokeColor=None))
    return d


def _make_styles(base: dict) -> dict:
    s = {}
    def ps(name, parent="Normal", **kw):
        s[name] = ParagraphStyle(name, parent=base[parent], **kw)
    ps("RTitle",   "Title",    textColor=colors.white, fontSize=18,
       fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=0)
    ps("RSubMeta", "Normal",   textColor=colors.HexColor("#B0BEC5"),
       fontSize=9, alignment=TA_CENTER, spaceAfter=0)
    ps("SecHdr",   "Heading2", textColor=C_HEADER, fontSize=11,
       fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=1)
    ps("Body",     "Normal",   fontSize=10, leading=14, spaceAfter=4)
    ps("Bullet",   "Normal",   fontSize=10, leading=13, leftIndent=14, spaceAfter=3)
    ps("GLetter",  "Normal",   textColor=colors.white, fontSize=30,
       fontName="Helvetica-Bold", alignment=TA_CENTER)
    ps("GScore",   "Normal",   textColor=colors.white, fontSize=10,
       fontName="Helvetica-Bold", alignment=TA_CENTER)
    ps("GVerdict", "Normal",   textColor=colors.HexColor("#ECEFF1"), fontSize=9)
    ps("FlagText", "Normal",   textColor=C_RED, fontSize=9, leftIndent=10, spaceAfter=2)
    return s


def _section(title: str, styles: dict) -> list:
    return [
        Spacer(1, 0.14 * inch),
        Paragraph(title, styles["SecHdr"]),
        HRFlowable(width="100%", thickness=0.8, color=C_RULE, spaceAfter=4),
    ]


def _bullet_list(items: list, styles: dict) -> list:
    return [Paragraph(f"• {i.strip()}", styles["Bullet"])
            for i in (items or []) if i and str(i).strip()]


# ---------------------------------------------------------------------------
# Auto-fallback bullet generators (used when LLM doesn't supply sections)
# ---------------------------------------------------------------------------

def _auto_strengths(skills: dict) -> list:
    strong = [k for k, v in skills.items() if str(v).lower() == "strong"]
    return [f"Demonstrated solid command of {s}." for s in strong] \
        or ["No skills were rated 'strong' in this session — see Work On section."]


def _auto_work_on(skills: dict) -> list:
    weak = [k for k, v in skills.items() if str(v).lower() == "needs_improvement"]
    return [f"Needs meaningful improvement in {s}. "
            f"Focus on hands-on practice and real-world examples." for s in weak] \
        or ["No critical skill gaps identified in this session."]


def _auto_expand_on(skills: dict) -> list:
    mid = [k for k, v in skills.items() if str(v).lower() == "adequate"]
    return [f"{s} shows a solid foundation — deepen with advanced practice "
            f"and concrete examples." for s in mid] \
        or ["All assessed skills are either strong or need foundational work."]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    candidate_name: str = "Candidate",
    job_role: str = "",
    company: str = "",
    evaluated_skills: dict = None,
    transcript_summary: str = "",
    guardrail_flags: list = None,      # ← safe default; was crashing when omitted
    strengths: list = None,            # ← new optional section
    work_on: list = None,              # ← new optional section
    expand_on: list = None,            # ← new optional section
    next_steps: list = None,           # ← new optional section
    output_dir: str = "reports",
    tool_context=None,
) -> dict[str, Any]:
    """
    Generate a rich graded PDF performance report for the candidate.

    Args:
        candidate_name:     Display name — use "Candidate" to avoid PII.
        job_role:           Target job title.
        company:            Target company name.
        evaluated_skills:   Dict of skill → "strong"|"adequate"|"needs_improvement".
                            Defaults to {} if omitted.
        transcript_summary: 2–3 sentence overall performance synthesis.
        guardrail_flags:    Session guardrail warnings. Defaults to [] if omitted.
        strengths:          Bullet strings for Strengths section.
                            Auto-generated from evaluated_skills if omitted.
        work_on:            Bullet strings for Work On section.
                            Auto-generated from evaluated_skills if omitted.
        expand_on:          Bullet strings for Expand On section.
                            Auto-generated from evaluated_skills if omitted.
        next_steps:         Up to 4 concrete recommended actions (optional).
        output_dir:         Directory to write the PDF. Defaults to /tmp.
        tool_context:       ADK tool context for state logging (optional).

    Returns:
        {
            "status":        "success" | "error",
            "report_path":   str,        # full path to the PDF
            "filename":      str,
            "overall_score": float,      # 0–100
            "overall_grade": str,        # A / B+ / B / C+ / C / D / F
        }
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Safe defaults — the LLM frequently omits optional params
    if evaluated_skills is None:
        evaluated_skills = {}
    if guardrail_flags is None:
        guardrail_flags = []

    if not REPORTLAB_AVAILABLE:
        result = {"status": "error",
                  "message": "reportlab not installed. Run: pip install reportlab"}
        if tool_context:
            log_tool_call(tool_context.state, "verifier_critic",
                          "generate_report", {"job_role": job_role}, result)
        return result

    try:
        # ── Compute overall score & grade ────────────────────────────────
        raw_scores = [RATING_SCORE.get(str(v).lower(), 50)
                      for v in evaluated_skills.values()]
        overall_score = round(sum(raw_scores) / len(raw_scores), 1) if raw_scores else 50.0
        overall_grade   = _letter_grade(overall_score)
        overall_verdict = _grade_verdict(overall_score)

        # ── File setup ───────────────────────────────────────────────────
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
        filename    = f"interview_report_{timestamp}.pdf"
        output_path = os.path.join(output_dir, filename)

        doc = SimpleDocTemplate(
            output_path, pagesize=LETTER,
            rightMargin=0.85 * inch, leftMargin=0.85 * inch,
            topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        )
        base_styles = getSampleStyleSheet()
        styles      = _make_styles(base_styles)
        story       = []

        # ── Header band ──────────────────────────────────────────────────
        hdr = Table(
            [[Paragraph("Interview Performance Report", styles["RTitle"])]],
            colWidths=[6.3 * inch],
        )
        hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_HEADER),
            ("TOPPADDING",    (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(hdr)

        date_str = datetime.datetime.now(datetime.UTC).strftime("%B %d, %Y")
        sub = Table(
            [[Paragraph(
                f"{candidate_name}  ·  {job_role} @ {company}  ·  {date_str}",
                styles["RSubMeta"],
            )]],
            colWidths=[6.3 * inch],
        )
        sub.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_SUBHDR),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(sub)
        story.append(Spacer(1, 0.18 * inch))

        # ── Overall grade box ─────────────────────────────────────────────
        gc = (C_GREEN  if overall_score >= 70 else
              C_AMBER  if overall_score >= 50 else
              C_RED)
        grade_row = Table(
            [[
                Paragraph(overall_grade,                styles["GLetter"]),
                Paragraph(f"{overall_score:.0f}/100",   styles["GScore"]),
                Paragraph(overall_verdict,              styles["GVerdict"]),
            ]],
            colWidths=[0.85 * inch, 1.05 * inch, 4.4 * inch],
        )
        grade_row.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), gc),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ]))
        story.append(KeepTogether([grade_row]))

        # ── Performance summary ───────────────────────────────────────────
        if transcript_summary:
            story.extend(_section("📋  Performance Summary", styles))
            story.append(Paragraph(transcript_summary, styles["Body"]))

        # ── Skill table ───────────────────────────────────────────────────
        if evaluated_skills:
            story.extend(_section("📊  Skill Evaluation", styles))
            rows = [[
                Paragraph("<b>Skill</b>",       base_styles["Normal"]),
                Paragraph("<b>Rating</b>",      base_styles["Normal"]),
                Paragraph("<b>Score Bar</b>",   base_styles["Normal"]),
            ]]
            for skill, rating in evaluated_skills.items():
                r    = str(rating).lower()
                rc   = RATING_COLOUR.get(r, colors.grey)
                hex_c = rc.hexval()[2:]
                label = r.replace("_", " ").title()
                rows.append([
                    Paragraph(str(skill), base_styles["Normal"]),
                    Paragraph(f'<font color="#{hex_c}"><b>{label}</b></font>',
                              base_styles["Normal"]),
                    _grade_bar(r),
                ])
            tbl = Table(rows, colWidths=[2.8 * inch, 1.35 * inch, 1.5 * inch],
                        repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0),  C_HEADER),
                ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",       (0, 0), (-1, 0),  9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_ALTROW]),
                ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#CFD8DC")),
                ("TOPPADDING",     (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
                ("LEFTPADDING",    (0, 0), (-1, -1), 7),
                ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(tbl)

        # ── Strengths ─────────────────────────────────────────────────────
        story.extend(_section("✅  Strengths", styles))
        story.extend(_bullet_list(strengths or _auto_strengths(evaluated_skills), styles))

        # ── Work On ───────────────────────────────────────────────────────
        story.extend(_section("🎯  Work On", styles))
        story.extend(_bullet_list(work_on or _auto_work_on(evaluated_skills), styles))

        # ── Expand On ─────────────────────────────────────────────────────
        story.extend(_section("💡  Expand On", styles))
        story.extend(_bullet_list(expand_on or _auto_expand_on(evaluated_skills), styles))

        # ── Next Steps ────────────────────────────────────────────────────
        if next_steps:
            story.extend(_section("🚀  Recommended Next Steps", styles))
            for i, step in enumerate(next_steps[:4], 1):
                story.append(Paragraph(f"{i}. {str(step).strip()}", styles["Bullet"]))

        # ── Guardrail flags ───────────────────────────────────────────────
        if guardrail_flags:
            story.extend(_section("⚠️  Session Flags", styles))
            for flag in guardrail_flags:
                story.append(Paragraph(f"⚠ {flag}", styles["FlagText"]))

        doc.build(story)

        result = {
            "status":        "success",
            "report_path":   os.path.abspath(output_path),
            "filename":      filename,
        }
        if tool_context:
            if tool_context:
                log_tool_call(tool_context.state, "verifier_critic", "generate_report", 
                          {"job_role": job_role}, result)
        return result

    except Exception as exc:
        import traceback
        result = {"status": "error",
                  "message": f"Failed to generate report: {exc}",
                  "traceback": traceback.format_exc()}
        if tool_context:
            log_tool_call(tool_context.state, "verifier_critic", "generate_report",
                          {"job_role": job_role}, result)
        return {"status": "error", "message": str(exc)}