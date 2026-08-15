"""
Vet PDF field extraction — LLM-based, not regex-based.

Deliberately different from Attestia Claims' extract_fields.py approach:
CMS-1500 is a standardized form (fixed layout since 2012), which is why a
regex/positional-crop pipeline works there. Veterinary invoices have no
equivalent standard — format varies by clinic — so extraction here uses
Claude directly rather than hand-tuned regex rules.

KNOWN LIMITATIONS (be upfront about these, don't let anyone discover them
by testing rather than by being told):
  1. Only handles text-layer PDFs. Scanned/image PDFs return None — there
     is no OCR fallback here (Claims' pytesseract path was not ported).
  2. No accuracy evaluation has been run on this extractor. There is no
     F1/precision number for it, unlike Claims' V1/V2/V3 OCR pipeline.
  3. average_market_rate is always null — nothing in the source PDF
     supplies it, and no external price lookup is wired in yet.

Requires: pip install pdfplumber anthropic
"""

import json
from pathlib import Path

import pdfplumber
import anthropic
from dotenv import load_dotenv

# 不依赖 load_dotenv() 的自动向上查找 —— 那个机制在被其他项目
# （比如 AttestiaUnified）import 时找不到 VetGuardC 自己的 .env。
# 直接算出 .env 的绝对路径：这个文件在 VetGuardC/src/ 下，
# .env 在 VetGuardC/ 根目录，也就是上一级。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = anthropic.Anthropic()

EXTRACTION_PROMPT = """Extract the following fields from this veterinary invoice text
as a JSON object. If a field is not present, use null.

{{
  "species": "dog|cat|bird|reptile|fish|hamster|rabbit|other",
  "breed": "string or null",
  "age": "integer (years) or null",
  "diagnosis": "string — the stated reason for visit / diagnosis",
  "procedures": ["list of procedure/service names as billed"],
  "billed_amount": "float — total charged",
  "average_market_rate": null,
  "modifier": "string or null (e.g. 'emergency')"
}}

INVOICE TEXT:
{text}

Respond with ONLY the JSON object."""


def extract_vet_claim_fields(pdf_path: str, claim_id: str, model: str = "claude-haiku-4-5") -> dict | None:
    """
    Extracts VetGuard-schema claim fields from a veterinary invoice PDF.

    Returns None if the PDF has no extractable text layer (i.e. it's a
    scanned image with no OCR fallback implemented here).
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if not text.strip():
        return None  # scanned/image PDF — would need an OCR fallback, not handled here

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    fields = json.loads(raw)
    fields["claim_id"] = claim_id
    return fields


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_vet_fields.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    result = extract_vet_claim_fields(str(pdf_path), claim_id=pdf_path.stem)

    if result is None:
        print("No extractable text found — likely a scanned PDF (no OCR fallback in this module).")
        sys.exit(1)

    print(json.dumps(result, indent=2))