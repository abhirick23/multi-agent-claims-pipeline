# Health Insurance Claims Processor

A multi-agent AI pipeline that automates health insurance claim decisions. Given a member's claim details and uploaded medical documents, the system verifies documents, extracts structured information using Gemini Vision, evaluates the claim against policy rules, checks for fraud signals, and produces a final decision — all with a full explainability trace showing every step.

**Live app:** [healthinuranceclaim.streamlit.app](https://healthinuranceclaim.streamlit.app)

---

## What it does

1. **Document verification** — checks that the right documents were uploaded for the claim type, that they are legible, and that they all belong to the same patient. Stops immediately with a specific, actionable message if anything is wrong.

2. **Extraction** — uses Gemini Vision to pull structured data from uploaded images: patient name, diagnosis, treatment, line items, amounts, doctor details.

3. **Policy evaluation** — runs the extracted data through a policy rules engine: waiting periods, exclusions, pre-authorization requirements, per-claim limits, network hospital discounts, copay calculations.

4. **Fraud detection** — checks for unusual same-day or monthly claim patterns and computes a fraud score.

5. **Decision** — produces `APPROVED`, `PARTIAL`, `REJECTED`, or `MANUAL_REVIEW` with an approved amount, rejection reasons, and a confidence score. Low confidence claims are flagged for advisory manual review.

Every step is recorded in a structured trace so the full reasoning behind any decision can be inspected.

---

## Try it live

Open [healthinuranceclaim.streamlit.app](https://healthinuranceclaim.streamlit.app) — no login required.

The app has three sections:

- **Home page** — replay any of 12 scripted test scenarios instantly (no API key needed, uses pre-loaded document content)
- **Submit claim** — upload real medical document images and run the full Gemini-powered pipeline
- **Eval suite** — run all 12 test cases at once and see the results

---

## Project structure

```
app/
  agents/         One class per pipeline stage
    document_verification_agent.py
    extraction_agent.py
    canonical_mapping.py        Diagnosis → policy term matching (keyword + LLM fallback)
    policy_evaluation_agent.py
    fraud_detection_agent.py
    decision_agent.py
  core/
    orchestrator.py             Wires the pipeline; never raises to caller
    confidence.py               Computes 0–1 confidence score from stage results
    exceptions.py
    logging_config.py           Rotating file logger, no stdout noise
  llm/
    gemini_client.py            Gemini Vision wrapper with retry/backoff and response_schema
    prompts.py
  models/                       Pydantic contracts for every stage input/output
  storage/
    policy_repository.py        Loads policy_terms.json into typed models
    claims_ledger.py            JSON-file ledger for claim history
  ui/
    streamlit_app.py            Home page + quick demo
    pages/
      1_submit_claim.py
      2_eval_suite.py
    components/
      trace_renderer.py

eval/
  run_eval.py                   Runs all 12 test cases, writes docs/EVAL_REPORT.md
  eval_helpers.py
  test_cases.json

data/
  policy_terms.json             Full policy config: coverage, limits, exclusions, members

docs/
  ARCHITECTURE.md               System design, decisions, 10x scale analysis
  COMPONENT_CONTRACTS.md        Input/output contracts for every component
  EVAL_REPORT.md                Per-case results for all 12 test scenarios

tests/
  agents/                       Unit tests per agent
  core/                         Orchestrator + confidence scoring tests
  conftest.py

mock_documents/
  generate_mocks.py             Generates sample medical document images for demo
  output/                       Pre-generated JPGs used in the live demo
```

---

## How to read the code

Start here, in this order:

1. **`app/models/`** — understand what data flows through the system before reading any logic
2. **`app/core/orchestrator.py`** — see how the five agents connect and how failures are handled
3. **`app/agents/`** — read each agent in pipeline order (verification → extraction → canonical_mapping → policy_evaluation → fraud → decision)
4. **`app/core/confidence.py`** — understand how the confidence score is calculated from stage results
5. **`app/llm/gemini_client.py`** — see how Gemini is called with structured output schemas

The architecture and design decisions are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Component input/output contracts are in [`docs/COMPONENT_CONTRACTS.md`](docs/COMPONENT_CONTRACTS.md).

---

## Run locally

**Requirements:** Python 3.10+, a [Gemini API key](https://aistudio.google.com/app/apikey) (free tier works for the live submission page; the home page quick demo and eval suite need no API key).

```bash
# 1. Clone and set up
git clone https://github.com/abhirick23/multi-agent-claims-pipeline.git
cd multi-agent-claims-pipeline
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

# 2. Add your API key
copy .env.example .env
# Open .env and set GEMINI_API_KEY=your-key-here

# 3. Run the app
streamlit run app/ui/streamlit_app.py

# 4. Run tests
pytest tests/ -q

# 5. Run the full eval suite (writes docs/EVAL_REPORT.md)
python eval/run_eval.py
```

---

## AI safety research

This project was used as an empirical testbed for four AI safety experiments, studying
pipeline-level safety properties that are separate from model-level alignment:

| Experiment | Safety property | Status |
|---|---|---|
| EXP1 — Schema enforcement | Does `response_schema` contain LLM hallucination vs prompt-only instructions? | Methodology complete — 30 API calls needed |
| EXP2 — Prompt injection | Can adversarial text in document images hijack pipeline extraction? | Methodology complete — 8 API calls needed |
| EXP3 — Human-in-the-loop robustness | Does confidence-based routing correctly escalate degraded/adversarial claims? | **5/5 scenarios passed** |
| EXP4 — Policy evasion | Does the two-tier exclusion mapping catch excluded conditions under adversarial paraphrasing? | **9/10 correct, FP = 0** |

**Key findings:**
- Confidence routing correctly preserved a policy rejection even when extraction confidence was 1.00 — high confidence does not override correct decisions
- The keyword tier caught 3 cases designed to require LLM fallback, via secondary synonyms in the combined text
- One false negative identified: "alcohol rehabilitation programme" evades both tiers due to a word-boundary regex gap — documented and root-caused in `eval/safety_results.json`

**Run the experiments:**

```bash
# EXP3 and EXP4 — no API key needed
python eval/safety_experiment.py --exp 3 4

# EXP1 and EXP2 — requires GEMINI_API_KEY (free tier, ~38 calls total)
python eval/safety_experiment.py --exp 1 2
```

Results are written to `eval/safety_results.json`. The full methodology is in `eval/safety_experiment.py`.

---

## Key design decisions

**Multi-agent over a single LLM call** — each agent has one responsibility, can fail independently, and can be tested in isolation. A failure in extraction does not stop policy evaluation.

**Keyword-first canonical mapping** — diagnosis text is matched to policy terms using a keyword/synonym dictionary before calling the LLM. This covers the common cases (diabetes, obesity, bariatric) deterministically at zero cost. The LLM is only invoked when keyword matching is inconclusive.

**Injection mode for testing** — `DocumentInput` accepts optional pre-loaded content fields that bypass Gemini entirely. All 12 test scenarios run deterministically with zero API calls using this mechanism.

**Confidence scoring** — the decision agent computes a 0–1 score based on extraction quality, stage failures, policy warnings, and LLM-assisted mapping. Below 0.75 triggers an advisory manual review recommendation; below 0.45 overrides the decision to `MANUAL_REVIEW` entirely.

**Structured LLM output** — all Gemini calls use `response_schema` to enforce exact Pydantic model shapes. No parsing, no regex, no prompt engineering tricks for output formatting.
