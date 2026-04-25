"""
tests/test_guardrails_and_difficult_candidates.py
--------------------------------------------------
Scenario Testing: "breaking" the bot with difficult candidates
and verifying that guardrails hold.

Three test categories:
  A. Guardrail unit tests   — direct calls to scan_report_for_violations
  B. Difficult-candidate simulation — verify get_transcript_summary flags
     evasive/vague answers correctly so the agent knows to probe
  C. Instruction-tuning contract tests — verify the agent prompts contain
     all required behavioural commitments

Run with:  python tests/test_guardrails_and_difficult_candidates.py
"""

import sys
import os
import types as pytypes

# ── Stubs so we can import without google-adk installed ──────────────────────
adk_agents = pytypes.ModuleType("google.adk.agents")
class _Agent:
    def __init__(self, **kw): pass
adk_agents.Agent = _Agent
sys.modules.setdefault("google", pytypes.ModuleType("google"))
sys.modules.setdefault("google.adk", pytypes.ModuleType("google.adk"))
sys.modules["google.adk.agents"] = adk_agents
adk_tools = pytypes.ModuleType("google.adk.tools")
adk_tools.google_search = None
sys.modules["google.adk.tools"] = adk_tools

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from guardrails import scan_report_for_violations, strip_pii, validate_output
from agents.simulation_specialist import (
    get_transcript_summary, record_answer, SIMULATION_SPECIALIST_INSTRUCTION,
)
from agents.verifier_critic import VERIFIER_CRITIC_INSTRUCTION

PASS_SYM = "✅"
FAIL_SYM = "❌"
results = []

def check(label: str, condition: bool, detail: str = ""):
    sym = PASS_SYM if condition else FAIL_SYM
    results.append((sym, label, detail))
    print(f"  {sym}  {label}" + (f"\n      └─ {detail}" if detail else ""))


# ============================================================================
# CATEGORY A — Guardrail scan tests
# ============================================================================
print("\n━━━ A. Guardrail Scan Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

clean_report = (
    "The candidate interviewed for the Software Engineer role at Acme Corp. "
    "They demonstrated strong knowledge of distributed systems and Python. "
    "Areas for improvement include system design depth and communication clarity."
)
r = scan_report_for_violations(clean_report, job_role="Software Engineer")
check("A1: Clean report → PASS verdict", r["verdict"] == "PASS")
check("A1: Clean report → show_pdf=True", r["show_pdf"] is True)
check("A1: Clean report → no flags", len(r["flags"]) == 0)

pii_report = (
    "The candidate John Smith (john.smith@gmail.com, 555-867-5309) "
    "performed well in the Software Engineer interview."
)
r = scan_report_for_violations(pii_report, job_role="Software Engineer")
check("A2: PII in report → BLOCK verdict", r["verdict"] == "BLOCK",
      f"verdict={r['verdict']}, flags={r['flags']}")
check("A2: PII in report → show_pdf=False", r["show_pdf"] is False)
check("A2: email stripped from cleaned_text", "john.smith@gmail.com" not in r["cleaned_text"])
check("A2: phone stripped from cleaned_text", "555-867-5309" not in r["cleaned_text"])

offtopic_report = (
    "The candidate discussed politics. "
    "The Software Engineer role requires Python and system design skills."
)
r = scan_report_for_violations(offtopic_report, job_role="Software Engineer")
check("A3: Off-topic → WARN verdict", r["verdict"] == "WARN", f"verdict={r['verdict']}")
check("A3: Off-topic → show_pdf still True", r["show_pdf"] is True)

critique_report = (
    "The candidate's accent was hard to understand. "
    "Their age may be a concern for this fast-paced Data Analyst role."
)
r = scan_report_for_violations(critique_report, job_role="Data Analyst")
check("A4: Personal critique (accent/age) → BLOCK", r["verdict"] == "BLOCK",
      f"flags={r['flags']}")
check("A4: Blocking flags populated", len(r["blocking_flags"]) > 0)

salutation_report = "Dear Jane Doe, thank you for interviewing for the ML Engineer role."
r = scan_report_for_violations(salutation_report, job_role="ML Engineer")
check("A5: Salutation name PII → BLOCK", r["verdict"] == "BLOCK", f"flags={r['flags']}")

text = "Call me at 415-555-1234 or email alice@example.com"
stripped = strip_pii(text)
check("A6: strip_pii removes phone", "415-555-1234" not in stripped)
check("A6: strip_pii removes email", "alice@example.com" not in stripped)

bad_turn = "Tell me about your disability and how it affects your work ethic."
flags = validate_output(bad_turn, job_role="Product Manager")
check("A7: validate_output catches 'disability'", any("disability" in f for f in flags))


# ============================================================================
# CATEGORY B — Difficult candidate scenarios
# ============================================================================
print("\n━━━ B. Difficult Candidate Scenarios ━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

class FakeCtx:
    def __init__(self): self.state = {}

ctx = FakeCtx()
record_answer(ctx, "Tell me about distributed systems experience.",
              "I don't know, I've never worked with those.")
s = get_transcript_summary(ctx)
check("B1: 'I don't know' answer → needs_probe=True", s["turns"][0]["needs_probe"] is True)

ctx2 = FakeCtx()
record_answer(ctx2, "Describe a production outage you handled.", "Pass.")
s2 = get_transcript_summary(ctx2)
check("B2: 'Pass' deflection → needs_probe=True", s2["turns"][0]["needs_probe"] is True)

ctx3 = FakeCtx()
record_answer(ctx3, "How do you approach code reviews?", "I review code.")
s3 = get_transcript_summary(ctx3)
check("B3: Very short answer (<20 words) → needs_probe=True", s3["turns"][0]["needs_probe"] is True)

ctx4 = FakeCtx()
long_answer = (
    "In my last role I led code reviews for a team of eight engineers. "
    "My process was to focus on business logic correctness first, then edge cases, "
    "then style. I found that detailed inline comments with suggested alternatives "
    "improved team learning significantly over a six-month period."
)
record_answer(ctx4, "How do you approach code reviews?", long_answer)
s4 = get_transcript_summary(ctx4)
check("B4: Substantive answer → needs_probe=False", s4["turns"][0]["needs_probe"] is False)

ctx5 = FakeCtx()
record_answer(ctx5, "Tell me about a failure.", "I'd rather not discuss that.")
s5 = get_transcript_summary(ctx5)
check("B5: 'I'd rather not' → needs_probe=True", s5["turns"][0]["needs_probe"] is True)

ctx6 = FakeCtx()
record_answer(ctx6, "Q1: Background?", "Five years in ML engineering at two startups.")
record_answer(ctx6, "Q2: Frameworks?", "PyTorch and JAX primarily.")
record_answer(ctx6, "Q3: A project?", "Built a recommendation engine using PyTorch.")
s6 = get_transcript_summary(ctx6)
check("B6: Multi-turn memory — 3 turns accessible", len(s6["turns"]) == 3)
check("B6: Q1 answer accessible at Q3", "ML engineering" in s6["turns"][0]["answer"])

ctx7 = FakeCtx()
pii_answer = "My name is Alice Johnson, email alice@johnson.com. I worked at Google."
record_answer(ctx7, "Tell me about yourself.", pii_answer)
r7 = scan_report_for_violations(pii_answer, job_role="Engineer")
check("B7: PII in candidate answer → BLOCK if put in report verbatim", r7["verdict"] == "BLOCK")
check("B7: Cleaned text has PII stripped", "alice@johnson.com" not in r7["cleaned_text"])

ctx8 = FakeCtx()
record_answer(ctx8, "What's your opinion on the current US president?",
              "That's a great political question — I think policy X is better.")
r8 = scan_report_for_violations(
    "Candidate discussed political views on policy.", job_role="Engineer"
)
check("B8: Off-topic political content in report → WARN or BLOCK",
      r8["verdict"] in ("WARN", "BLOCK"))

ctx9 = FakeCtx()
record_answer(ctx9, "Tell me about yourself.",
              "Whatever, this interview is pointless. I don't care.")
s9 = get_transcript_summary(ctx9)
check("B9: Hostile/dismissive answer → needs_probe=True (short text)",
      s9["turns"][0]["needs_probe"] is True)

ctx10 = FakeCtx()
record_answer(ctx10, "Describe your ML experience.",
              "idk I guess I've done some stuff with neural nets or whatever")
s10 = get_transcript_summary(ctx10)
check("B10: 'idk' / 'whatever' → needs_probe=True", s10["turns"][0]["needs_probe"] is True)


# ============================================================================
# CATEGORY C — Instruction-tuning contract tests
# ============================================================================
print("\n━━━ C. Instruction-Tuning Contract Tests ━━━━━━━━━━━━━━━━━━━━━━━\n")

sim = SIMULATION_SPECIALIST_INSTRUCTION
vc  = VERIFIER_CRITIC_INSTRUCTION

check("C1: Simulation has 'encouraging but firm'", "encouraging but firm" in sim.lower())
check("C2: Simulation instructs deep technical probing", "probe deeply" in sim.lower())
check("C3: Simulation handles 'I don't know'", "don't know" in sim.lower())
check("C4: Simulation redirects off-topic questions", "scope" in sim.lower())
check("C5: Simulation references needs_probe flag", "needs_probe" in sim)
check("C6: Simulation requires get_transcript_summary before each question",
      "get_transcript_summary" in sim)
check("C7: Verifier mandates run_guardrail_scan", "run_guardrail_scan" in vc)
check("C8: Verifier blocks PDF on BLOCK verdict",
      "BLOCK" in vc and ("NOT reveal" in vc.upper() or "do not" in vc.lower()))
check("C9: Verifier has structured output headers", "Strengths" in vc and "Improvement" in vc)
check("C10: Verifier forbids personal critique",
      "personal characteristics" in vc.lower() or "personal attribute" in vc.lower())
check("C11: Verifier forbids fabricating skill ratings", "fabricat" in vc.lower())
check("C12: Verifier has WARN handling (not just BLOCK)",
      "WARN" in vc and "warning" in vc.lower())


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SCENARIO TEST RESULTS")
print("=" * 60)
passed = sum(1 for s, _, _ in results if s == PASS_SYM)
total  = len(results)
print(f"Passed: {passed}/{total}")
if passed < total:
    print("\nFailed tests:")
    for s, label, detail in results:
        if s == FAIL_SYM:
            print(f"  {FAIL_SYM} {label}")
            if detail:
                print(f"      └─ {detail}")
print("=" * 60)
sys.exit(0 if passed == total else 1)
