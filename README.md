# Attestia Vet

### AI-Powered Veterinary Insurance Fraud Detection using Multi-Agent Reasoning

Built with **Python**, **Claude Sonnet 4 / Haiku**, **LangGraph**, and **Streamlit**.

Attestia Vet combines deterministic billing rules, LLM-assisted clinical reasoning, and an evidence-based adversarial review step to analyze veterinary claims and produce structured, auditable fraud decisions.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [The Problem We're Solving](#the-problem-were-solving)
- [Fraud Types Covered](#fraud-types-covered)
- [How the Pipeline Actually Works](#how-the-pipeline-actually-works)
- [Agent Overview](#agent-overview)
- [Design Principles](#design-principles)
- [Why Veterinary Insurance?](#why-veterinary-insurance)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Results and Evaluation](#results-and-evaluation)
- [PDF Input Path (New)](#pdf-input-path-new)
- [Example Output](#example-output)
- [Knowledge Base](#knowledge-base)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Key Engineering Findings](#key-engineering-findings)
- [Current Limitations](#current-limitations)
- [Potential Next Steps](#potential-next-steps)
- [Future Roadmap](#future-roadmap)
- [What We Learned](#what-we-learned)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## What This Project Does

Given a single veterinary claim, the pipeline attempts to determine:

- **Is it fraudulent?** (Yes / No)
- **What type of fraud?** (one of 8 specific fraud patterns)
- **Why was it flagged?** (plain-English explanation an investigator can act on)

The system works on a single claim document with no reference or history required, matching how real veterinary insurance reviewers work.

### Note on Naming

The draft README uses the product name **Attestia Vet**. Much of the current code, UI, and API still use the internal name **VetGuard**. Both refer to the same system.

---

## The Problem We're Solving

Pet insurance is a growing market, and claims review teams still face the same core operational problem seen in other insurance domains: suspicious claims are expensive to review manually and easy to miss at scale. Existing systems fail because:

- **Manual review is slow** - inconsistent and cannot scale to millions of claims
- **Historical pattern analysis** takes months to detect fraud and misses first-time fraudsters
- **Neither approach** can evaluate a single claim in isolation, the moment it arrives

Our system fills this gap, evaluating every claim independently the moment it arrives, using only the information in that one document.

---

## Fraud Types Covered

The generated synthetic dataset covers **8 fraud categories**:

| Fraud Type | Primary Handling Path | Risk Level |
|------------|----------------------|------------|
| Duplicate Billing | Rule Checker | Low |
| Unbundling | Rule Checker | High |
| Species Mismatch | Rule Checker | Medium |
| Modifier Abuse | Rule Checker | Medium |
| Phantom Billing | Clinical Reasoner | High |
| Diagnosis Mismatch | Clinical Reasoner | Medium |
| Upcoding | Clinical Reasoner | High |
| Vaccine Padding | Clinical Reasoner | Medium |

### Important Nuance

- Some Clinical Reasoner categories also have deterministic helper logic for obvious patterns
- Some rule-detectable traces can still be intentionally routed onward when the code wants Agent 2 to assign a more specific label

---

## How the Pipeline Actually Works

The current pipeline is **conditional, not strictly linear**:

```text
Claim JSON
  |
  +--> Agent 0: Document Integrity Checker (optional, only if pdf_path is present)
  |
  +--> Agent 1: Rule Checker
         |
         +--> if rule-based fraud is finalized here: stop
         |
         +--> otherwise -> Agent 2: Clinical Reasoner
                           |
                           +--> if clean: stop
                           +--> if indeterminate/error: stop
                           +--> if fraud with enough confidence:
                                   -> Agent 3: Adversarial Validator
                                      -> final verdict
```

### Current Runtime Outcomes

The current code can return three runtime outcomes:

| Outcome | Description |
|---------|-------------|
| **Fraud** | Claim flagged as fraudulent with specific fraud type |
| **Clean** | Claim appears legitimate |
| **Indeterminate** | API key missing, LLM call fails, or response cannot be parsed |

**Important:** It is not correct to describe the current implementation as always producing a binary decision under every runtime condition.

This routing behavior comes from:

- `src/fraud_engine.py` - thin wrapper
- `src/fraud_engine_langgraph.py` - LangGraph StateGraph with conditional edges

---

## Agent Overview

| Agent | Purpose | Implementation |
|-------|---------|----------------|
| **Agent 0: Document Integrity Checker** | Optional PDF forensics before claim reasoning | Python (`pikepdf`, `Pillow`, `exiftool`) |
| **Agent 1: Rule Checker** | Deterministic checks for billing and species-rule violations | Python |
| **Agent 2: Clinical Reasoner** | Tool-using claim reasoning for clinically ambiguous fraud types | Anthropic Claude API + Python tools |
| **Agent 3: Adversarial Validator** | Conservative override step that can only rescue fraud findings by citing knowledge-base evidence | Anthropic Claude API + knowledge base |

---

## Design Principles

Unlike chatbot-style AI systems, Attestia Vet follows an **enterprise-grade design philosophy**:

- **Rule-first architecture** - deterministic issues handled first by Agent 1
- **Deterministic fraud rules remain deterministic** - no LLM hallucinations on calculations
- **LLMs perform reasoning instead of calculations** - they explain, not compute
- **Every recommendation is explainable** - human-readable decisions every time
- **Validation is independent from generation** - second opinion catches errors
- **Evidence-based overrides** - Agent 3 can only override by citing a specific `knowledge_base` entry
- **AI supports investigators instead of replacing them** - human-in-the-loop design
- **Shared tool-backed reasoning** - Clinical Reasoner and MCP server work from the same knowledge-base files

---

## Why Veterinary Insurance?

Veterinary insurance follows many of the same billing principles as human healthcare, including diagnoses, procedures, modifiers, and reimbursement rules, while **avoiding HIPAA restrictions**.

This makes it an ideal domain for developing explainable enterprise AI architectures for fraud detection.

---

## Technology Stack

| Layer | Current Repo Technology |
|-------|------------------------|
| **Orchestration** | LangGraph (StateGraph with conditional edges + `MemorySaver` checkpointing) |
| **LLMs** | Anthropic Claude Haiku / Sonnet |
| **API** | Flask |
| **UI** | Streamlit |
| **Tool Integration** | MCP (Model Context Protocol) — standalone server, compatible with Claude Desktop, Cursor, or any MCP client |
| **PDF field extraction** | Claude-based extraction (`extract_vet_fields.py`) — see [PDF Input Path](#pdf-input-path-new) |
| **Config** | `python-dotenv` |
| **Charting** | Plotly |
| **PDF Forensics** | Optional: `pikepdf`, `Pillow`, `exiftool-py` |

---

## Project Structure

```text
Attestia-Vet/
|-- AGENTS.md                        # Rules for AI coding agents working on this repo
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- .env                             # API keys (gitignored)
|
|-- run_pipeline.py                  # Main entry point: generate -> process -> evaluate
|                                    # Flags: --generate, --sample N, --full, --evaluate,
|                                    #        --fast (Haiku), --sonnet (default)
|-- run_full_batch.py                # Standalone script to run all claims via fraud_engine.py
|
|-- debug_vaccine.py                 # One-off debug script for vaccine padding cases
|-- diagnose_vp.py                   # One-off debug script for diagnosis-mismatch cases
|-- test_on_real_data.py             # Ad hoc script to spot-check reasoner accuracy
|-- test_claim.pdf                   # Sample PDF claim (diagnosis mismatch case)
|-- test_claim_clean.pdf             # Sample PDF claim (legitimate case, for false-positive checks)
|
|-- src/
|   |-- app.py                       # Streamlit dashboard (Live Demo / Performance / Audit Log)
|   |-- api.py                       # Flask API bridging web frontend to agent pipeline
|   |-- adversarial_validator.py     # Agent 3 - challenges Agent 2's verdict; can only
|   |                                # override by citing a specific knowledge_base entry
|   |-- check_pipeline_path.py       # Dev utility - path/caching sanity check
|   |-- clinical_reasoner.py         # Agent 2 - Claude-based reasoning: Phantom billing,
|   |                                # Diagnosis mismatch, Upcoding, Vaccine padding
|   |-- document_integrity_checker.py # Agent 0 - forensic PDF integrity/forgery pre-check,
|   |                                # runs only when a PDF is attached to the claim
|   |-- extract_vet_fields.py        # LLM-based PDF -> structured claim field extraction
|   |                                # (added for the PDF upload path; see limitations)
|   |-- evaluate.py                  # Computes precision/recall/F1 per fraud type + overall
|   |-- fraud_engine.py              # Thin wrapper around fraud_engine_langgraph.py;
|   |                                # single entry point used by dashboard, batch runner
|   |-- fraud_engine_langgraph.py    # LangGraph StateGraph - defines routing/orchestration
|   |                                # logic via conditional edges
|   |-- generate_claims.py           # Generates synthetic claims dataset (660 claims,
|   |                                # 8 fraud types + legitimate, seeded for reproducibility)
|   |-- rule_checker.py              # Agent 1 - deterministic checks: Duplicate billing,
|   |                                # Unbundling, Species mismatch, Modifier abuse
|   |-- upcoding_rules.json          # Reference price/complexity data (not currently wired)
|   `-- vetguard_mcp_server.py       # MCP server exposing knowledge-base lookups and agents
|                                    # as tools (compatible with Claude Desktop, Cursor, etc.)
|
|-- knowledge_base/                  # Long-term memory - source of truth for rule checks
|   |                                # and adversarial validation
|   |-- bundle_rules.json            # Procedure bundling rules (unbundling detection) - 30 rules
|   |-- species_procedure_rules.json # Species-procedure validity rules - 19 rules
|   |-- species_exceptions.json      # Rare-but-legitimate specialist exceptions - 4 entries
|   `-- clinical_whitelist.json      # Legitimate but unusual procedure-diagnosis pairs - 23 entries
|
|-- data/
|   |-- raw_claims/claims.json       # Generated dataset (gitignored - regenerate locally)
|   `-- final_results/
|       |-- results.json             # Full per-claim results (gitignored)
|       `-- metrics.json             # Aggregate + per-fraud-type metrics (gitignored)
|
`-- tests/
    |-- test_adversarial_validator.py
    |-- test_fraud_regressions.py
    `-- test_pipeline_status.py
```

### Important Notes on Structure

- `src/upcoding_rules.json` exists in the repo as reference data, but it is **not currently wired** into the active runtime path
- The generated dataset and result artifacts are local files and are **gitignored**: `data/raw_claims/claims.json`, `data/final_results/results.json`, `data/final_results/metrics.json`

---

## Quick Start

### 1. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add Your Anthropic API Key

Create or update `.env` at the project root:

```env
ANTHROPIC_API_KEY=your_real_key_here
```

The LLM-based agents (`clinical_reasoner.py`, `adversarial_validator.py`, `extract_vet_fields.py`) load this via `load_dotenv()` resolved relative to their own file location, so it works regardless of which directory you launch a script from — including when this codebase is imported into another project (see the unified integration note in [Current Limitations](#current-limitations)).

### 3. Generate the Dataset

```bash
python src/generate_claims.py
```

### 4. Run a Full Batch Locally

```bash
python run_full_batch.py
python src/evaluate.py
```

### 5. Optional Entry Points

**Dashboard:**

```bash
streamlit run src/app.py
# -> opens http://localhost:8501
```

**API:**

```bash
python src/api.py
```

**Wrapper pipeline with model selection:**

```bash
# Test on a small sample (20 claims, Sonnet by default)
python run_pipeline.py --sample 20

# Run full batch (all 660 claims)
python run_pipeline.py --full

# Use Haiku for cheaper/faster runs
python run_pipeline.py --full --fast

# Re-evaluate existing results without re-running claims
python run_pipeline.py --evaluate
```

### 6. Optional PDF-Forensics Dependencies

If you want the full Agent 0 PDF analysis path, install the extra packages:

```bash
pip install pikepdf pillow exiftool-py
```

---

## Dataset

The generator currently creates **660 claims total**:

| Class | Count |
|-------|-------|
| Legitimate claims | 200 |
| Fraudulent claims | 460 |
| **Total** | **660** |

Fraudulent claims are distributed across 8 fraud types. The generated labels `fraud_indicator` and `fraud_type` are used for dataset generation and evaluation only. They are **not** supposed to be read by the inference path when making a fraud decision.

**Primary generator file:** `src/generate_claims.py`

---

## Results and Evaluation

**Important:** This repo should **not** hard-code a single permanent metrics table in the README. End-to-end results depend on:

- The current synthetic dataset version
- The current rules and reasoning code
- The model path being used
- Whether the Anthropic API key is valid and available
- Run-to-run variance from unpinned `temperature` on the LLM calls (typically ±5–10% on LLM-judged categories)

To generate a fresh local metrics snapshot:

```bash
python src/generate_claims.py
python run_full_batch.py
python src/evaluate.py
```

**For explicit model selection from the wrapper script:**

```bash
python run_pipeline.py --generate
python run_pipeline.py --sample 20
python run_pipeline.py --full
python run_pipeline.py --full --fast
python run_pipeline.py --evaluate
```

### Most recent verified run (660 claims)

| Metric | Value |
|---|---|
| F1 | 0.987 |
| Precision | 0.985 |
| Recall | 0.989 |
| Accuracy | 0.982 |
| Coverage | 1.00 (all 660 claims scored; 0 indeterminate) |

This is a single representative run, verified end-to-end (dataset generation → batch processing → evaluation → spot-checked against individual claim audit trails), not a cherry-picked or averaged number. Always regenerate locally for an authoritative, current score.

Full breakdown is available in `data/final_results/metrics.json` after running evaluation locally.

---

## PDF Input Path (New)

In addition to structured JSON claims, the pipeline now accepts PDF invoices via `extract_vet_fields.py`, which uses Claude directly to extract claim fields from the PDF's text layer.

**Why not a regex/positional pipeline (like Attestia Claims' CMS-1500 OCR path)?** CMS-1500 is a standardized, position-fixed form — that's what makes a coordinate-based extraction pipeline (crop-to-region, regex) viable there. Veterinary invoices have no equivalent standard; format varies by clinic. An LLM-based extractor handles that variability without hand-tuned rules per clinic format.

**Known limitations of this path (see also [Current Limitations](#current-limitations)):**
- Only works on PDFs with a text layer — no OCR fallback for scanned/image PDFs.
- No accuracy benchmark has been run on this extractor (unlike Attestia Claims' OCR pipeline, which has a documented V1→V2→V3 F1 progression).
- `average_market_rate` is always `null` in extracted output — nothing in a typical invoice PDF supplies this, and no external price lookup is wired in yet.

Verified end-to-end (PDF → Agent 0 → Agents 1-3 → verdict) on two hand-designed test cases: a diagnosis-mismatch case (`test_claim.pdf`, correctly flagged as fraud) and a legitimate case (`test_claim_clean.pdf`). Note: `test_claim_clean.pdf`, because it was generated with the open-source `reportlab` library, is correctly flagged by Agent 0 as having a suspicious PDF producer — this is Agent 0 working as intended (real veterinary billing software doesn't use `reportlab`), not a false positive in the clinical reasoning layers. Use a text-editor-produced or real-world PDF to test the clinical reasoning path without triggering Agent 0.

---

## Example Output

The real saved batch output is a structured claim-level record. A simplified example looks like this:

```json
{
  "claim_id": "VET00160",
  "species": "reptile",
  "breed": "Bearded Dragon",
  "diagnosis": "Splenic mass",
  "procedures": ["Chemotherapy", "Complete blood count panel"],
  "billed_amount": 590.48,
  "average_market_rate": 585.0,
  "agent1_result": {
    "agent": "rule_checker",
    "timestamp": "2026-07-06T21:51:41.118180",
    "fraud_detected": false,
    "fraud_type": null,
    "decision_status": "clean",
    "explanation": "No rule violations detected. Passing to clinical reasoner."
  },
  "agent2_result": {
    "agent": "clinical_reasoner",
    "timestamp": "2026-07-06T21:51:52.288124",
    "fraud_detected": false,
    "fraud_type": null,
    "confidence": "low",
    "decision_status": "clean",
    "explanation": "Claim appears clinically appropriate.",
    "error": null
  },
  "agent3_result": null,
  "final_verdict": false,
  "final_fraud_type": null,
  "deciding_agent": "clinical_reasoner",
  "decision_status": "clean",
  "errors": []
}
```

### Notes on Output Structure

- Rule-based fraud can finalize after Agent 1 without Agent 2 or Agent 3
- Agent 3 only appears when Agent 2 has already flagged fraud and the graph routes the claim onward for challenge/review
- `indeterminate` can occur when the API key is missing, an LLM call fails, or a response cannot be parsed

---

## Knowledge Base

Current knowledge-base files in `knowledge_base/`:

| File | Purpose | Current Count |
|------|---------|---------------|
| `bundle_rules.json` | Procedure bundling rules (unbundling detection) | 30 rules |
| `species_procedure_rules.json` | Species-procedure validity rules (species mismatch) | 19 rules |
| `species_exceptions.json` | Rare-but-legitimate specialist procedure exceptions | 4 entries |
| `clinical_whitelist.json` | Legitimate but unusual procedure-diagnosis pairs | 23 entries |

The **adversarial validator** relies on these files when deciding whether a flagged claim has a documented legitimate exception.

---

## Key Engineering Decisions

### 1. Rule-First Architecture

Deterministic issues such as unbundling or species-rule violations are handled first by Agent 1. This prevents LLM hallucinations on calculations and ensures that clear-cut fraud is caught quickly.

### 2. Evidence-Based Overrides

Agent 3 (Adversarial Validator) is intentionally **conservative**. It is designed to **rescue** a claim only when a knowledge-base entry supports doing so. This prevents arbitrary reversals and makes the system more transparent.

### 3. Shared Tool-Backed Reasoning

The Clinical Reasoner and the MCP server both work from the **same underlying knowledge-base files** for:

- Bundle checks
- Species validity checks
- Whitelist lookups

This ensures consistency across different entry points.

### 4. Conditional Routing with LangGraph

The pipeline uses **LangGraph StateGraph** with conditional edges to route claims intelligently:

- If Agent 1 finds fraud -> stop
- If Agent 1 finds no fraud -> continue to Agent 2
- If Agent 2 finds fraud with enough confidence -> route to Agent 3 for adversarial review
- If Agent 2 finds clean or indeterminate -> stop

### 5. Optional PDF Pre-Check

Agent 0 exists and is wired into the unified engine when a claim includes a `pdf_path`. Its deepest analysis requires optional packages (`pikepdf`, `pillow`, `exiftool-py`) not installed by default.

### 6. Two Model Modes

- **Sonnet (wrapper default):** High accuracy, suitable for production-oriented runs through `run_pipeline.py`
- **Haiku (`--fast`):** Cheaper, faster, good for prototyping or internal testing

---

## Key Engineering Findings

### 1. NCCI Panel Rules Work for Veterinary Billing

Just like human healthcare, there are "comprehensive" procedures that should not be unbundled. Our rule checker detects this with high accuracy.

### 2. Species Mismatch is a Powerful Signal

Procedures that are clinically impossible for a species are a very strong fraud signal. This rule path has performed extremely well in local testing.

### 3. Tool-Backed Prompting Improves LLM Reasoning

The current reasoner benefits from tightly scoped fraud definitions, tool-backed investigation, and explicit label guidance. That structure helps the model focus on the right kind of inconsistency instead of free-form speculation.

### 4. Adversarial Validation Catches Hallucinations

The second LLM opinion (Agent 3) helps reduce false positives compared to a single-LLM approach by forcing a conservative, evidence-based challenge step.

### 5. Some Fraud is Inherently Ambiguous

Upcoding and diagnosis mismatch exist on a spectrum. In production settings, low-confidence or borderline cases are better routed to human review than treated as simple deterministic decisions.

### 6. Document Integrity Adds a New Dimension — and a New False-Positive Surface

Agent 0 catches document-level fraud that would otherwise go undetected, such as forged clinic letterhead, altered dates, or modified procedure codes in the PDF itself. The tradeoff: it can flag legitimate-content PDFs generated by non-standard tooling (e.g. `reportlab`-produced test files) as suspicious purely on producer metadata, independent of whether the claim content itself is fraudulent. Test PDFs used to validate the clinical reasoning layers should be produced with standard tooling to avoid conflating "Agent 0 correctly flagged unusual PDF provenance" with "the clinical agents produced a false positive."

### 7. A Timezone Assumption in Agent 0's Date Check Is a Known Risk

`document_integrity_checker.py`'s future-creation-date check compares a UTC-stamped PDF timestamp against a timezone-naive `datetime.now()`. On a machine in a timezone behind UTC, a PDF stamped with the current UTC time can register as "future-dated" and get flagged CRITICAL. This hasn't caused a confirmed false positive on a real invoice yet, but it's a latent risk worth fixing before relying on this check in a timezone-diverse deployment.

---

## Current Limitations

| Limitation | Impact |
|------------|--------|
| **Inconsistent naming** between Attestia Vet and VetGuard | Confusion in documentation and code |
| **Model defaults not uniform** across entry points | Different behavior depending on how you run the system |
| **Result quality depends on valid Anthropic API key** | Many non-rule claims become `indeterminate` without a key |
| **No FastAPI implementation in this repo** | Flask is used here; a separate unified integration project wraps this repo's LangGraph engine behind FastAPI alongside Attestia Claims |
| **`upcoding_rules.json` not wired** | Reference data exists but is not used at runtime |
| **Generated dataset and results are gitignored** | Readers cannot verify static metrics from repo alone |
| **PDF field extraction (`extract_vet_fields.py`) has no OCR fallback and no accuracy benchmark** | Only text-layer PDFs work; extraction quality is unverified at scale |
| **`fraud_engine.py` and `fraud_engine_langgraph.py` both define a `run_batch` function with slightly different logic** | Minor code duplication; not currently causing incorrect behavior, but worth consolidating |
| **`finalize_node()` in `fraud_engine_langgraph.py` contains an unreachable duplicate branch** | Dead code from an earlier refactor; harmless but should be cleaned up |
| **Agent 0's future-date check assumes matching timezones** (see Engineering Findings above) | Latent false-positive risk on timezone-diverse deployments |

---

## Potential Next Steps

- [ ] **Standardize naming** across the repo (Attestia Vet vs VetGuard)
- [ ] **Add fail-fast API-key preflight** before long batch runs
- [ ] **Make model selection consistent** across all entry points
- [ ] **Add OCR fallback and an accuracy benchmark** for the PDF field-extraction path
- [ ] **Fix the timezone assumption** in Agent 0's future-date check
- [ ] **Consolidate the duplicate `run_batch` implementations** and remove dead code in `finalize_node()`
- [ ] **Wire up `upcoding_rules.json`** to the active runtime path
- [ ] **Pin `temperature=0`** on LLM calls for reproducible evaluation
- [ ] **Pydantic validation** for all claim objects

---

## Future Roadmap

| Phase | Feature | Timeline |
|-------|---------|----------|
| **Phase 1** | Enterprise AI Platform deployment | Q3 2026 |
| **Phase 2** | Multi-domain fraud detection (human healthcare, lending) | Q4 2026 |
| **Phase 3** | Human review dashboard with workflow integration | Q4 2026 |
| **Phase 4** | Cloud deployment (AWS/GCP) with auto-scaling | Q1 2027 |
| **Phase 5** | PostgreSQL backend for claims storage | Q1 2027 |
| **Phase 6** | Enterprise authentication (SSO, RBAC) | Q2 2027 |
| **Phase 7** | SAP ERP integration for financial reconciliation | Q2 2027 |
| **Phase 8** | Financial services risk detection (lending, insurance) | Q3 2027 |

Note: Phase 2 (multi-domain fraud detection) is already partially underway — a separate unified integration project combines this engine with Attestia Claims (human medical billing fraud detection, co-developed with team member S. Poman) behind a shared API, as a first step toward the domain-agnostic vision below.

### Phase 2 Detail: Multi-Domain Fraud Detection

The architectural patterns developed in Attestia Vet, multi-agent orchestration, explainable AI, and human-in-the-loop review, are domain-agnostic. The same pipeline can be adapted for:

- **Human healthcare claims** (with HIPAA compliance)
- **Consumer lending** (fraud detection in loan applications)
- **Insurance underwriting** (risk assessment)
- **Financial services** (transaction fraud, AML)

---

## What We Learned

This project demonstrates that **multi-agent architectures** with **explainable AI** can detect fraud with high accuracy while maintaining trust through transparency.

### Key Takeaways for Enterprise AI

1. **Deterministic rules should handle deterministic problems.** LLMs are powerful, but they are not calculators. Use rules for what is clear and LLMs for what requires reasoning.
2. **Independent validation catches errors.** A second LLM reviewer (Adversarial Validator) helps reduce false positives. This pattern is applicable to any high-stakes AI system.
3. **Evidence-based overrides build trust.** Requiring the adversarial agent to cite specific `knowledge_base` entries prevents arbitrary reversals.
4. **Explainability is not optional.** In regulated industries, you must be able to explain why the AI made a decision. Design for explainability from day one.
5. **Some decisions require human judgment.** Upcoding and diagnosis mismatch exist on a spectrum. The AI's role is to flag suspicious cases, not to replace domain experts in ambiguous cases.
6. **Multi-agent systems are more robust than single-agent systems.** By distributing responsibilities across specialized agents, the system is more resilient to individual agent failures.
7. **Document integrity matters — but bring its own false-positive surface.** Fraud does not always hide in the structured data; sometimes it is in the document itself. That said, document-provenance checks (like flagging unusual PDF-producer metadata) can conflate "unusual tooling" with "fraudulent content" — worth keeping conceptually separate when interpreting results.
8. **Read your own default configuration, not just your docstrings.** A documented, intentional precision/recall tradeoff (like the confidence-threshold toggle in Attestia Claims) is only useful if the default value and the surrounding documentation actually agree with each other.

The lessons learned here translate directly to:

- **Consumer lending** - fraud detection, affordability assessment
- **Insurance** - claims fraud, underwriting
- **Enterprise automation** - compliance checks, audit trails

---

## Author

**Hazel Sun**

Cornell University

Enterprise AI - Agentic AI - SAP - LLM Applications


## Acknowledgments

Developed as an independent research project exploring enterprise AI architectures. The system was inspired by real-world medical billing fraud detection challenges and designed to demonstrate how modern AI can be applied to regulated industries while maintaining explainability and auditability.

### References

- LangGraph documentation: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- Anthropic Claude API: [https://docs.anthropic.com/](https://docs.anthropic.com/)
- Model Context Protocol (MCP): [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)

---

## License

This project is for demonstration and educational purposes only. Contact the author for commercial licensing inquiries.
