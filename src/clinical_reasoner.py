"""
Attestia Vet — Agent 2: Clinical Reasoner
A tool-using Claude agent for fraud types that require clinical judgment:
Phantom billing, Diagnosis mismatch, Upcoding, Vaccine padding.

The agent is given the claim and THREE TOOLS over the veterinary knowledge
bases (the same lookups the MCP server exposes). It investigates, then
returns a structured JSON verdict. No dataset-specific patterns appear in
the prompt — guidance is general veterinary billing knowledge only.

Requires ANTHROPIC_API_KEY. Without it, run() returns a clear
"unavailable" result and the pipeline continues on rules only.
"""

import json
import os
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

KB = Path(__file__).parent.parent / "knowledge_base"

with open(KB / "bundle_rules.json", encoding="utf-8") as f:
    BUNDLE_RULES = json.load(f)
with open(KB / "species_procedure_rules.json", encoding="utf-8") as f:
    SPECIES_RULES = json.load(f)
with open(KB / "clinical_whitelist.json", encoding="utf-8") as f:
    WHITELIST = json.load(f)
with open(KB / "species_exceptions.json", encoding="utf-8") as f:
    SPECIES_EXCEPTIONS = json.load(f)

MODEL_DEV   = "claude-haiku-4-5"
MODEL_FINAL = "claude-sonnet-4-6"
MODEL       = MODEL_DEV
MAX_TOOL_TURNS = 8

_client = None





def _get_client():
    global _client
    if _client is None and os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _decision_status(detected) -> str:
    if detected is True:
        return "fraud"
    if detected is False:
        return "clean"
    return "indeterminate"


def _procedure_matches(procedure: str, target: str) -> bool:
    proc_lower = procedure.lower()
    target_lower = target.lower()
    return target_lower in proc_lower or proc_lower in target_lower


def _is_vaccine_procedure(procedure: str) -> bool:
    proc_lower = procedure.lower()
    return "vaccine" in proc_lower or "core vaccine series" in proc_lower


# ── Tool implementations (shared logic with the MCP server) ───────────────────
def tool_check_species_validity(procedure: str, species: str) -> dict:
    p, s = procedure.lower(), species.lower()
    for rule in SPECIES_RULES:
        if rule["procedure"].lower() in p:
            if s in [x.lower() for x in rule.get("invalid_species", [])]:
                return {"valid": False, "rule": rule["rule"]}
            if s in [x.lower() for x in rule.get("valid_species", [])]:
                return {"valid": True, "rule": rule["rule"]}
    return {"valid": None, "rule": "No species rule on file for this procedure."}


def tool_check_bundle(procedure_1: str, procedure_2: str) -> dict:
    p1, p2 = procedure_1.lower(), procedure_2.lower()
    for rule in BUNDLE_RULES:
        r1, r2 = rule["procedure_1"].lower(), rule["procedure_2"].lower()
        if {p1, p2} >= {r1, r2} or (r1 in p1 and r2 in p2) or (r1 in p2 and r2 in p1):
            return {"violation": True, "fraud_type": rule["fraud_type"], "rule": rule["rule"]}
    return {"violation": False, "rule": "No bundling rule matches this pair."}


def tool_search_whitelist(procedure: str = "", diagnosis: str = "") -> dict:
    p, d = procedure.lower(), diagnosis.lower()
    hits = []
    for e in SPECIES_EXCEPTIONS:
        if p and e["procedure"].lower() in p:
            hits.append({"procedure": e["procedure"],
                         "species": e.get("species"),
                         "diagnosis": ", ".join(e.get("diagnosis_context", [])),
                         "rationale": e["rationale"], "source": e["source"]})
    for e in WHITELIST:
        pm = p and e["procedure"].lower() in p or p in e["procedure"].lower() if p else False
        dm = d and (e["diagnosis"].lower() in d or d in e["diagnosis"].lower()) if d else False
        if pm or dm:
            hits.append({"procedure": e["procedure"], "diagnosis": e["diagnosis"],
                         "rationale": e["rationale"], "source": e["source"]})
    return {"matches": hits[:8] or "No whitelist entries match."}


TOOLS = [
    {"name": "check_species_validity",
     "description": "Check whether a veterinary procedure is valid for a given species, per the species-procedure knowledge base.",
     "input_schema": {"type": "object", "properties": {
         "procedure": {"type": "string"}, "species": {"type": "string"}},
         "required": ["procedure", "species"]}},
    {"name": "check_bundle",
     "description": "Check whether two procedures violate veterinary bundling rules when billed together.",
     "input_schema": {"type": "object", "properties": {
         "procedure_1": {"type": "string"}, "procedure_2": {"type": "string"}},
         "required": ["procedure_1", "procedure_2"]}},
    {"name": "search_whitelist",
     "description": "Search the clinical whitelist of legitimate-but-unusual procedure/diagnosis combinations before concluding fraud.",
     "input_schema": {"type": "object", "properties": {
         "procedure": {"type": "string"}, "diagnosis": {"type": "string"}}}},
]

_DISPATCH = {
    "check_species_validity": lambda a: tool_check_species_validity(a.get("procedure",""), a.get("species","")),
    "check_bundle":           lambda a: tool_check_bundle(a.get("procedure_1",""), a.get("procedure_2","")),
    "search_whitelist":       lambda a: tool_search_whitelist(a.get("procedure",""), a.get("diagnosis","")),
}


SYSTEM_PROMPT = """You are a veterinary claims fraud analyst reviewing a single claim.

Fraud types in scope (pick the ONE best fit if fraud is present):
- PHANTOM BILLING — the procedure could not plausibly have been performed on this patient
  (anatomically or practically impossible for the species/size, or wildly inconsistent with
  the visit described).
- DIAGNOSIS MISMATCH — a real procedure, but no clinical relationship to the stated
  diagnosis or visit reason.
- UPCODING — a more intensive/expensive service than the stated condition justifies
  (judged on the service choice, not on price alone).
- VACCINE PADDING — an implausible number or combination of vaccines administered in a
  single visit relative to species and standard immunization protocols.

Investigation standard:
- Use the tools to verify species validity, bundling, and whitelist exceptions BEFORE
  concluding. If a whitelist entry covers the combination, the claim is legitimate.
- Charges above market rate are NOT fraud by themselves; pricing is only supporting
  evidence when a clinical inconsistency already exists.
- Prefer DIAGNOSIS MISMATCH when the procedure could physically be performed on the
  species but does not fit the stated diagnosis or visit reason, even if the
  procedure is unusually complex or expensive.
- Prefer UPCODING when the visit reason is minor or narrow but the billed service is
  substantially more intensive than that visit justifies.
- Prefer VACCINE PADDING when the suspicious pattern is duplicate, redundant, or
  species-inappropriate vaccine stacking within one immunization visit, even if a
  duplicate-billing or species-mismatch trace is also present.
- Reserve PHANTOM BILLING for procedures that are physically or practically impossible,
  not merely over-complex or poorly justified choices.
- confidence "high" only when the combination is clinically indefensible;
  "medium" when unusual but a rare legitimate scenario exists;
  if the procedures reasonably fit the diagnosis, species, and age, the claim is LEGITIMATE.

After investigating, respond with ONLY a JSON object, no other text:
{"fraud_detected": true|false,
 "fraud_type": "Phantom billing"|"Diagnosis mismatch"|"Upcoding"|"Vaccine padding"|null,
 "confidence": "high"|"medium"|"low",
 "explanation": "one clear sentence"}"""


def build_claim_message(claim: dict) -> str:
    return (f"CLAIM {claim['claim_id']}\n"
            f"- Species/breed: {claim.get('species')} / {claim.get('breed','?')}, age {claim.get('age','?')}\n"
            f"- Visit reason / diagnosis: {claim.get('diagnosis')}\n"
            f"- Procedures billed: {', '.join(claim.get('procedures', []))}\n"
            f"- Modifier: {claim.get('modifier') or 'none'}\n"
            f"- Billed: ${claim.get('billed_amount')} (average market rate ${claim.get('average_market_rate')})\n\n"
            f"Investigate with the tools as needed, then give your JSON verdict.")


def _base(claim, detected=False, ftype=None, conf="low", expl="", raw="", error=None):
    return {"claim_id": claim["claim_id"], "agent": "clinical_reasoner",
            "fraud_detected": detected, "fraud_type": ftype, "confidence": conf,
            "explanation": expl, "clinical_flags": [ftype] if detected else [],
            "pass_to_agent3": detected is True, "raw_response": raw,
            "decision_status": _decision_status(detected), "error": error}


def run(claim: dict, model: str = None) -> dict:
    client = _get_client()
    if client is None:
        return _base(
            claim,
            detected=None,
            conf="low",
            expl="Clinical reasoning unavailable; manual review required because "
                 "ANTHROPIC_API_KEY is not set.",
            error="ANTHROPIC_API_KEY not set."
        )
    model = model or MODEL
    messages = [{"role": "user", "content": build_claim_message(claim)}]
    text = ""
    try:
        for _ in range(MAX_TOOL_TURNS):
            resp = client.messages.create(model=model, max_tokens=700,
                                          system=SYSTEM_PROMPT, tools=TOOLS,
                                          messages=messages)
            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        out = _DISPATCH.get(block.name, lambda a: {"error": "unknown tool"})(block.input)
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": json.dumps(out)})
                messages.append({"role": "user", "content": results})
                continue
            text = "".join(b.text for b in resp.content if b.type == "text")
            break

        if not text.strip():
            # Tool-turn budget exhausted while the model was still investigating.
            # Force a final verdict with tools disabled.
            messages.append({"role": "user", "content":
                "Tool budget exhausted. Based on everything gathered so far, "
                "respond NOW with only your JSON verdict."})
            resp = client.messages.create(model=model, max_tokens=700,
                                          system=SYSTEM_PROMPT, tools=TOOLS,
                                          tool_choice={"type": "none"},
                                          messages=messages)
            text = "".join(b.text for b in resp.content if b.type == "text")

        clean = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        # Robust parse: take the first {...} JSON object in the reply
        if not clean.startswith("{"):
            import re as _re
            m = _re.search(r"\{.*\}", clean, _re.DOTALL)
            clean = m.group(0) if m else clean
        v = json.loads(clean)
        return _base(claim, bool(v.get("fraud_detected")), v.get("fraud_type"),
                     v.get("confidence", "low"), v.get("explanation", ""), text)
    except Exception as e:
        error = f"LLM reasoning failed: {e}"
        return _base(
            claim,
            detected=None,
            conf="low",
            expl=f"{error}. Manual review required.",
            raw=text,
            error=error
        )


if __name__ == "__main__":
    demo = {"claim_id": "TEST", "species": "fish", "breed": "Betta", "age": 2,
            "diagnosis": "Preventive care visit", "procedures": ["General anesthesia"],
            "billed_amount": 210.0, "average_market_rate": 200, "modifier": None}
    print(json.dumps(run(demo), indent=2))
