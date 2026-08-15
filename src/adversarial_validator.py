"""
VetGuard — Agent 3: Adversarial Validator
Novel contribution: challenges Agent 2's fraud verdict.
Only overrides if Claude can cite a specific whitelist entry.
Prevents false positives on legitimate but unusual procedure-diagnosis pairs.
Now enhanced to better handle phantom billing and species mismatch cases.
"""

import json
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = anthropic.Anthropic()
KB = Path(__file__).parent.parent / "knowledge_base"

with open(KB / "clinical_whitelist.json", "r", encoding="utf-8") as f:
    WHITELIST = json.load(f)
with open(KB / "species_exceptions.json", "r", encoding="utf-8") as f:
    SPECIES_EXCEPTIONS = json.load(f)

# Expand whitelist with species exceptions at runtime
EXTENDED_WHITELIST = WHITELIST + SPECIES_EXCEPTIONS


def _decision_status(detected) -> str:
    if detected is True:
        return "fraud"
    if detected is False:
        return "clean"
    return "indeterminate"


def _diagnosis_terms(entry: dict) -> list[str]:
    """Normalize whitelist diagnosis fields across both KB schemas."""
    value = entry.get("diagnosis_context", entry.get("diagnosis"))
    if not value:
        return []
    if isinstance(value, list):
        return [str(term) for term in value]
    return [str(value)]


def _procedure_matches(entry: dict, procedures_lower: list[str]) -> bool:
    entry_proc = entry.get("procedure", "").lower()
    return bool(entry_proc) and any(
        entry_proc in proc or proc in entry_proc for proc in procedures_lower
    )


def _diagnosis_matches(entry: dict, diagnosis_lower: str) -> bool:
    return any(
        term.lower() in diagnosis_lower or diagnosis_lower in term.lower()
        for term in _diagnosis_terms(entry)
    )


def find_relevant_whitelist_entries(claim: dict) -> list[dict]:
    """Return whitelist or exception entries relevant to the claim."""
    diagnosis_lower = claim["diagnosis"].lower()
    species_lower = claim["species"].lower()
    procedures_lower = [p.lower() for p in claim["procedures"]]

    species_hits = []
    exact_hits = []
    seen = set()

    def add_hit(bucket: list[dict], entry: dict):
        entry_key = json.dumps(entry, sort_keys=True)
        if entry_key not in seen:
            bucket.append(entry)
            seen.add(entry_key)

    for entry in EXTENDED_WHITELIST:
        proc_match = _procedure_matches(entry, procedures_lower)
        diagnosis_terms = _diagnosis_terms(entry)
        diag_match = _diagnosis_matches(entry, diagnosis_lower) if diagnosis_terms else False
        species_values = [s.lower() for s in entry.get("species", [])]

        if species_values:
            if species_lower not in species_values or not proc_match:
                continue
            if diagnosis_terms and not diag_match:
                continue
            add_hit(species_hits, entry)
            continue

        if proc_match and diag_match:
            add_hit(exact_hits, entry)
    if species_hits:
        return species_hits
    return exact_hits


def build_whitelist_context(claim: dict) -> str:
    """Find whitelist entries relevant to this claim's procedures and diagnosis, including species exceptions."""
    relevant = find_relevant_whitelist_entries(claim)

    if not relevant:
        return "No matching whitelist entries found for this procedure-diagnosis-species combination."

    lines = ["Potentially relevant whitelist entries and clinical exceptions:"]
    for e in relevant[:15]:  # Limit to 15 entries to avoid context overflow
        diagnoses = ", ".join(_diagnosis_terms(e)) or "no diagnosis context provided"
        if "species" in e:
            lines.append(
                f"- {e['procedure']} on {', '.join(e['species'])} for {diagnoses}: "
                f"{e['rationale']} (Source: {e['source']})"
            )
        else:
            lines.append(
                f"- {e['procedure']} + {diagnoses}: {e['rationale']} "
                f"(Source: {e['source']})"
            )
    return "\n".join(lines)


def _format_whitelist_entry(entry: dict) -> str:
    diagnoses = ", ".join(_diagnosis_terms(entry)) or "no diagnosis context"
    if "species" in entry:
        return f"{entry['procedure']} on {', '.join(entry['species'])} for {diagnoses}"
    return f"{entry['procedure']} + {diagnoses}"


SYSTEM_PROMPT = """You are a senior veterinary clinician reviewing a fraud analyst's decision.
Your role is to CHALLENGE the fraud verdict — construct the strongest possible clinical argument
that the claim is LEGITIMATE.

Rules you must follow:
1. You may only override the fraud verdict if you can cite a SPECIFIC entry from the provided whitelist or clinical exceptions.
2. Treat the provided WHITELIST REFERENCE as the authoritative source for specialist species exceptions.
3. For SPECIES MISMATCH fraud: Override ONLY if the cited exception shows the procedure is actually possible for that species in specialized practice.
4. For PHANTOM BILLING fraud: Override ONLY if the procedure is technically feasible for the species.
5. For UPCODING fraud: Override ONLY if the complex procedure is medically justified (e.g., senior pet needs comprehensive panel).
6. For VACCINE PADDING fraud: Override ONLY if this is a legitimate puppy/kitten series or multi-valent vaccine.
7. General clinical reasoning alone is NOT sufficient to override — you must cite the whitelist or exception.
8. If no whitelist entry applies, you must uphold the fraud verdict.
9. Be precise and conservative — false negatives (missed fraud) are more costly than false positives.

When evaluating species mismatch:
- Consider if the procedure is anatomically possible (e.g., TPLO in cats IS possible, just less common)
- Consider if specialist referral centers might perform this procedure
- Use the provided clinical exceptions as the source of truth for rare bird, reptile, and feline specialist cases
- Only override if you can cite a legitimate clinical justification

When evaluating phantom billing:
- Determine if the procedure is PHYSICALLY IMPOSSIBLE (fish anesthesia) vs. just uncommon
- Only override if there's documented clinical necessity and feasibility

Respond ONLY with a valid JSON object — no preamble, no markdown.
"""

CHALLENGE_TEMPLATE = """A fraud analyst flagged this claim. Challenge the verdict.

ORIGINAL CLAIM:
- Species: {species} ({breed})
- Diagnosis: {diagnosis}
- Procedures: {procedures}
- Fraud verdict: {fraud_type} (confidence: {confidence})
- Analyst explanation: {explanation}

WHITELIST REFERENCE (including clinical exceptions):
{whitelist_context}

INSTRUCTIONS:
Construct the strongest possible argument that this claim is LEGITIMATE.
Treat the whitelist reference as the source of truth for specialist species exceptions.
You may ONLY override if a specific whitelist entry or clinical exception supports it.
If no whitelist entry applies, uphold the fraud verdict.

For SPECIES MISMATCH cases:
- Ask: Is this procedure actually possible for this species?
- If yes and you can cite a source, consider override

For PHANTOM BILLING cases:
- Ask: Is this procedure technically feasible?
- Consider: Would a specialist referral center perform this?

For UPCODING cases:
- Ask: Is there clinical justification for the more complex procedure?
- Example: Senior pet, chronic condition, specific symptoms

For VACCINE PADDING cases:
- Ask: Is this a legitimate puppy/kitten series?
- Consider: Age of animal, vaccine types

Respond ONLY with this JSON:
{{
  "override_applied": true or false,
  "whitelist_entry_cited": "exact procedure + diagnosis + species combination from whitelist" or null,
  "override_rationale": "clinical justification citing the whitelist entry or exception" or null,
  "final_fraud_detected": true or false,
  "validator_explanation": "one sentence summary of your decision"
}}"""


def run(claim: dict, agent2_result: dict, model: str = "claude-haiku-4-5-20251001") -> dict:
    """
    Challenge Agent 2's fraud verdict.
    Only called when Agent 2 detected fraud.
    """
    relevant_entries = find_relevant_whitelist_entries(claim)
    species_exception_hits = [entry for entry in relevant_entries if entry.get("species")]
    if species_exception_hits:
        cited = _format_whitelist_entry(species_exception_hits[0])
        rationale = (
            f"Knowledge-base species exception explicitly supports {cited}, "
            "so the fraud verdict is overridden."
        )
        return {
            "claim_id": claim["claim_id"],
            "agent": "adversarial_validator",
            "override_applied": True,
            "whitelist_entry_cited": cited,
            "override_rationale": rationale,
            "final_fraud_detected": False,
            "validator_explanation": rationale,
            "raw_response": "",
            "decision_status": "clean",
            "error": None
        }

    whitelist_context = build_whitelist_context(claim)

    prompt = CHALLENGE_TEMPLATE.format(
        species=claim["species"],
        breed=claim.get("breed", "Unknown"),
        diagnosis=claim["diagnosis"],
        procedures=", ".join(claim["procedures"]),
        fraud_type=agent2_result.get("fraud_type", "Unknown"),
        confidence=agent2_result.get("confidence", "Unknown"),
        explanation=agent2_result.get("explanation", ""),
        whitelist_context=whitelist_context
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=768,  # Increased for more detailed reasoning
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        final_fraud_detected = result.get("final_fraud_detected")
        if not isinstance(final_fraud_detected, bool):
            final_fraud_detected = None

        return {
            "claim_id": claim["claim_id"],
            "agent": "adversarial_validator",
            "override_applied": result.get("override_applied", False),
            "whitelist_entry_cited": result.get("whitelist_entry_cited"),
            "override_rationale": result.get("override_rationale"),
            "final_fraud_detected": final_fraud_detected,
            "validator_explanation": result.get("validator_explanation", ""),
            "raw_response": raw,
            "decision_status": _decision_status(final_fraud_detected),
            "error": None
        }

    except json.JSONDecodeError as e:
        error = f"Adversarial validator parse failure: {e}"
        return {
            "claim_id": claim["claim_id"],
            "agent": "adversarial_validator",
            "override_applied": False,
            "whitelist_entry_cited": None,
            "override_rationale": None,
            "final_fraud_detected": None,
            "validator_explanation": f"{error}. Manual review required.",
            "raw_response": "",
            "decision_status": "indeterminate",
            "error": error
        }
        # Parse failure — conservatively uphold the fraud verdict
        return {
            "claim_id": claim["claim_id"],
            "agent": "adversarial_validator",
            "override_applied": False,
            "whitelist_entry_cited": None,
            "override_rationale": None,
            "final_fraud_detected": True,
            "validator_explanation": f"Parse error — fraud verdict upheld: {str(e)}",
            "raw_response": ""
        }
    except Exception as e:
        error = f"Adversarial validator API failure: {e}"
        return {
            "claim_id": claim["claim_id"],
            "agent": "adversarial_validator",
            "override_applied": False,
            "whitelist_entry_cited": None,
            "override_rationale": None,
            "final_fraud_detected": None,
            "validator_explanation": f"{error}. Manual review required.",
            "raw_response": "",
            "decision_status": "indeterminate",
            "error": error
        }
        return {
            "claim_id": claim["claim_id"],
            "agent": "adversarial_validator",
            "override_applied": False,
            "whitelist_entry_cited": None,
            "override_rationale": None,
            "final_fraud_detected": True,
            "validator_explanation": f"API error — fraud verdict upheld: {str(e)}",
            "raw_response": ""
        }


if __name__ == "__main__":
    # Test case: TPLO on cat (should potentially be overridden)
    test_claim = {
        "claim_id": "TEST001",
        "species": "cat",
        "breed": "Domestic Shorthair",
        "age": 7,
        "diagnosis": "Cranial cruciate ligament rupture",
        "procedures": ["TPLO surgery"],
        "billed_amount": 3675.00,
        "average_market_rate": 3500.00,
        "modifier": None
    }
    agent2_mock = {
        "fraud_detected": True,
        "fraud_type": "Species mismatch",
        "confidence": "high",
        "explanation": "TPLO surgery is a canine-specific procedure that cannot be performed on cats"
    }
    result = run(test_claim, agent2_mock)
    print(json.dumps(result, indent=2))
    
    # Test case: Echocardiogram on reptile (should potentially be overridden)
    test_claim2 = {
        "claim_id": "TEST002",
        "species": "reptile",
        "breed": "Bearded Dragon",
        "age": 5,
        "diagnosis": "Pericardial effusion",
        "procedures": ["Echocardiogram complete"],
        "billed_amount": 472.50,
        "average_market_rate": 450.00,
        "modifier": None
    }
    agent2_mock2 = {
        "fraud_detected": True,
        "fraud_type": "Phantom billing",
        "confidence": "high",
        "explanation": "Echocardiogram cannot be performed on reptiles due to anatomical limitations"
    }
    result2 = run(test_claim2, agent2_mock2)
    print(json.dumps(result2, indent=2))
