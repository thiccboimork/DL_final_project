from tools.resume_parser import parse_resume

# Path to a real PDF resume on your machine
TEST_PDF = r"/Users/caitl/OneDrive/Documents/Spring 2026/Evelyn Weber Resume-1.pdf"

result = parse_resume(TEST_PDF)

if result["status"] == "success":
    print("✅ Successfully parsed PDF")
    print(f"Character count: {result['character_count']}")
    print("\n--- Preview (First 200 chars) ---")
    print(result["raw_text"][:200])
    
    # Check if PII stripping worked
    if "[EMAIL REDACTED]" in result["raw_text"] or "[PHONE REDACTED]" in result["raw_text"]:
        print("\n✅ PII Guardrails are active.")
else:
    print(f"❌ Error: {result['message']}")