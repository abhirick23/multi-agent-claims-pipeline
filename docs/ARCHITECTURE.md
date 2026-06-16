# Architecture

## 1. Overview

This system processes a health insurance claim through five specialist agents, orchestrated
sequentially:

```
ClaimInput + DocumentInput[]
   |
   v
[1] Document Verification Agent  ──► VerificationResult
   |  (BLOCKING issue? -> STOP, return member_message, no decision)
   v
[2] Extraction Agent (per document) ──► ExtractionResult[]
   |
   v
[3] Policy Evaluation Agent (+ Canonical Mapping) ──► PolicyEvaluationResult
   |
   v
[4] Fraud Detection Agent ──► FraudCheckResult
   |
   v
[5] Decision Agent (+ confidence scoring) ──► ClaimDecision
   |
   v
ClaimResult { verification, extractions, policy_evaluation, fraud_check, decision, trace }
```

Every stage appends structured entries to a shared `ClaimTrace` — the trace **is** the
explainability layer. The Streamlit UI renders it as a timeline of expandable steps, each showing
what was checked, what was found, and why a particular branch was taken.

## 2. Design philosophy: a custom lightweight multi-agent pipeline

We deliberately did **not** reach for LangGraph/CrewAI/AutoGen. The "agents" here are plain Python
classes (`app/agents/*.py`), each:

- subclassing a tiny `BaseAgent` (`app/agents/base.py`) that lazily provides a `GeminiClient`,
- exposing one `run(input, trace) -> output` method,
- with a typed Pydantic **input** and **output** contract (`app/models/*.py`).

The `Orchestrator` (`app/core/orchestrator.py`) is the only thing that knows the call order. This
keeps the system:

- **Simple** — five files, five contracts, one orchestrator. No framework-specific concepts
  (graphs, tools, memory stores) to learn or debug.
- **Elegant for this domain** — each agent maps 1:1 onto a real claims-processing role (document
  intake, transcription, policy adjudication, fraud review, final decision), which is exactly how
  a human claims team is organized. The trace reads like an adjudicator's case notes.
- **Testable** — every agent can be unit-tested with plain Pydantic objects, no mocked framework
  runtime.
- **"Multi-agent" in the sense that matters**: each stage has a distinct responsibility,
  distinct authority (e.g. only Verification can stop the pipeline before a decision; only Fraud
  Detection can force `MANUAL_REVIEW` regardless of policy outcome), and its own auditable
  reasoning trace — not just "five function calls."

## 3. The five agents

### Stage 1 — Document Verification Agent (`document_verification_agent.py`)

Confirms the *right* documents were uploaded, are *legible*, and belong to the *same patient* —
before any policy logic runs. Three checks, each producing `VerificationIssue`s:

1. **Quality** — any document classified `UNREADABLE` is a BLOCKING issue naming that specific
   file and asking for a re-upload (TC002).
2. **Document requirements** — uploaded document types are compared against
   `policy.document_requirements[claim_category]`. If a required type is missing, the message
   names both what was uploaded and what's required (TC001).
3. **Patient name consistency** — if documents carry different `patient_name_on_doc` values, this
   is BLOCKING and the message names both people (TC003).

If any issue is BLOCKING, `VerificationResult.passed = False` and the Orchestrator **stops the
pipeline immediately** — `ClaimResult.stopped_early = True`, `decision = None`,
`member_message` is the concatenation of the blocking messages. No claim decision is ever made on
a claim with an unresolved document problem.

Each document is classified either from `DocumentInput.actual_type/quality/patient_name_on_doc`
(injection mode, `source="INJECTED"`) or via `GeminiClient.classify_document()` on
`DocumentInput.file_path` (live mode, `source="VISION_LLM"`). If the Gemini call fails, the
document degrades to `UNKNOWN`/`POOR` rather than crashing — this can itself surface as a
verification issue, but never as an unhandled exception.

### Stage 2 — Extraction Agent (`extraction_agent.py`)

Runs once per document, transcribing it into the single flat `ExtractedContent` schema (a
superset of fields across all document types — prescriptions, bills, lab reports, etc.). This
agent is deliberately **policy-agnostic**: it transcribes what's on the page, nothing more.

- **Injection mode**: `DocumentInput.content` is validated directly into `ExtractedContent`,
  `extraction_status="SUCCESS"`, `overall_confidence=HIGH`, `source="INJECTED"`.
- **Live mode**: `GeminiClient.extract_content()` with `response_schema=ExtractedContent`. Field-
  level confidence is then derived by checking whether the document-type's "key fields"
  (`KEY_FIELDS_BY_TYPE`) were populated — all present → `HIGH`, some missing → `MEDIUM`, all
  missing → `LOW`.
- **Failure modes**: no `file_path` and no `content` → `extraction_status="FAILED"`. A Gemini
  call that exhausts retries (`GeminiAPIError`) → `extraction_status="FAILED"`,
  `overall_confidence=LOW`, `error` set — but the **other** documents still get extracted, and
  the pipeline continues.
- **`simulate_failure`** (TC011): the Orchestrator sets this on the *last* document when
  `ClaimInput.simulate_component_failure=True`, deterministically exercising the FAILED path
  without a real outage.

### Stage 3 — Policy Evaluation Agent (`policy_evaluation_agent.py`)

The deterministic rules engine. Every numeric threshold and rule it applies is read from
`policy_terms.json` via `PolicyRepository` / `PolicyTerms` — **nothing is hardcoded**. Checks run
in a fixed order; the first hard rule that FAILs short-circuits to `REJECTED` with a specific,
member-facing message and `policy_reference`:

| # | Check | Code | On FAIL |
|---|-------|------|---------|
| 1 | Canonical mapping (see §4) | — | (never fails the claim itself) |
| 2 | Minimum claim amount | `MIN_CLAIM_AMOUNT` | REJECTED |
| 3 | Submission deadline | `SUBMISSION_DEADLINE_EXCEEDED` | REJECTED |
| 4 | Initial + condition-specific waiting periods | `WAITING_PERIOD` | REJECTED, states the exact eligibility date (TC005) |
| 5 | Whole-claim exclusions | `EXCLUDED_CONDITION` | REJECTED (TC012) |
| 6 | Pre-authorization for high-value tests | `PRE_AUTH_MISSING` | REJECTED, tells member how to resubmit (TC007) |
| 7 | Per-claim limit (DENTAL exempt — see §6) | `PER_CLAIM_EXCEEDED` | REJECTED, states both limit and claimed amount (TC008) |
| 8 | Annual OPD limit | `ANNUAL_OPD_LIMIT_EXCEEDED` | WARN only (doesn't block) |
| 9 | Category-specific evaluation (line items for DENTAL, sub-limit for others) | — | may produce `PARTIAL`/`REJECTED` line items (TC006) |
| 10 | Financial calculation: network discount **before** co-pay | `FINANCIAL_CALCULATION` (INFO) | — |

Step 10 always computes, in order: `eligible_base → amount_after_discount (network discount) →
approved_amount (co-pay applied on the discounted amount)`, recorded in full as
`FinancialBreakdown` (TC010: discount before co-pay is load-bearing for the test).

### Stage 4 — Fraud Detection Agent (`fraud_detection_agent.py`)

Pure computation over `claim.claims_history` (merged with the persisted `ClaimsLedger`) and
`policy.fraud_thresholds` — no LLM calls, nothing that can fail at runtime. Produces
`FraudSignal`s:

- `SAME_DAY_CLAIMS_EXCEEDED` (HIGH) — more same-day claims than `same_day_claims_limit` (TC009).
- `MONTHLY_CLAIMS_EXCEEDED` (MEDIUM) — more this-month claims than `monthly_claims_limit`.
- `HIGH_VALUE_CLAIM` (MEDIUM) — `claimed_amount > high_value_claim_threshold`.

A weighted `fraud_score` is computed from severities (see §6 for the weight values) and
`requires_manual_review` is set if the score crosses `fraud_score_manual_review_threshold`,
`claimed_amount > auto_manual_review_above`, or **any** HIGH-severity signal fired.

### Stage 5 — Decision Agent (`decision_agent.py`)

The final synthesis step:

1. Compute `confidence_score` via `app/core/confidence.py` (see §5).
2. If `fraud.requires_manual_review` → `decision = MANUAL_REVIEW`, overriding the policy hint;
   every fraud signal's message is copied into both `notes` and `manual_review_reasons` (TC009).
3. Otherwise `decision = policy_eval.decision_hint`.
4. If `confidence_score < CONFIDENCE_FORCE_MANUAL_REVIEW (0.45)` → **force** `MANUAL_REVIEW`
   regardless of (2)/(3) — too much of the pipeline degraded to trust the result.
5. Else if `confidence_score < CONFIDENCE_ADVISORY_REVIEW (0.75)` → keep the decision but set
   `manual_review_recommended=True` with a reason (TC011: one failed extraction, claim stays
   `APPROVED`, confidence=0.65, flagged for a human to double-check).
6. `approved_amount` / `financial_breakdown` are only populated for `APPROVED`/`PARTIAL`.
   `rejection_reasons` are built by matching `policy_eval.rejection_reasons` codes back to the
   matching FAILed `PolicyCheckResult` to recover the full message + `policy_reference`.

## 4. Canonical mapping — the "interpretation" sub-step

Free-text diagnoses ("Morbid Obesity — BMI 37", "Type 2 Diabetes Mellitus") need to be mapped onto
the policy's own vocabulary (`waiting_periods.specific_conditions` keys,
`exclusions.conditions` terms) before the deterministic rules in §3 can apply. This mapping lives
**inside the Policy Evaluation Agent** (`app/agents/canonical_mapping.py`), not the Extraction
Agent — extraction stays a faithful, policy-agnostic transcription, while mapping is the
auditable "interpretation" step that ops needs to inspect separately. Every mapping result is its
own trace entry: raw diagnosis/treatment text in, matched policy terms + method out.

- **Tier 1 (always runs, no LLM)** — keyword/synonym dictionaries
  (`WAITING_PERIOD_KEYWORDS`, `EXCLUSION_KEYWORDS`) match free text to policy terms via
  whole-word regex. Every candidate term is filtered against what `policy_terms.json` actually
  defines, so this tier can never invent a term the policy doesn't have. Exclusions are checked
  **before** waiting periods — a permanently excluded condition makes any waiting period moot
  (TC012's "Bariatric Consultation" hits the exclusion keyword list directly).
- **Tier 2 (LLM fallback)** — only runs if Tier 1 found nothing *and* there's diagnosis/treatment
  text to reason about. Calls `GeminiClient.classify_canonical_category()` with
  `response_schema=GeminiCanonicalMappingResponse`, constrained to the policy's actual candidate
  terms. On any Gemini failure, Tier 1's (empty) result is kept, the attempt is logged as
  `DEGRADED` in the trace, and the pipeline continues — this sub-step can never fail the claim.
- **Dental line items** — always a deterministic exact/substring match against
  `opd_categories.dental.covered_procedures` / `excluded_procedures` (TC006: "Root Canal
  Treatment" → COVERED, "Teeth Whitening" → EXCLUDED/cosmetic).

## 5. Confidence scoring (`app/core/confidence.py`)

A single 0–1 number summarizing how much the pipeline had to compromise before reaching
`decision_hint`. Starts at `BASE_CONFIDENCE = 1.0` and subtracts:

| Source | Penalty |
|---|---|
| Extraction `FAILED` | −0.25 |
| Extraction `PARTIAL` | −0.10 |
| Extraction `overall_confidence == LOW` | −0.10 |
| Extraction `overall_confidence == MEDIUM` | −0.05 |
| Each LOW-confidence field (capped) | −0.05 each, max −0.10 |
| Each `DegradedContext.degraded_stages` entry | −0.15 |
| Each `DegradedContext.failed_stages` entry | −0.30 |
| Each `WARN`-status policy check | −0.05 |
| Canonical mapping `method == LLM_ASSISTED` | −0.05 |
| Any exclusion match with `confidence < 0.75` | −0.05 |

Result is clamped to `[0, 1]`. Two thresholds (also tunable constants, used by the Decision
Agent — §3.5):

- `CONFIDENCE_FORCE_MANUAL_REVIEW = 0.45` — hard floor, overrides the decision itself.
- `CONFIDENCE_ADVISORY_REVIEW = 0.75` — below this, keep the decision but recommend review.

**TC011 confidence breakdown (verified against actual output):**

TC011 has `simulate_component_failure=True`. The `ExtractionAgent.run()` handles this internally
and returns `ExtractionResult(extraction_status="FAILED", overall_confidence=LOW)` without
raising. Because no exception propagates to the Orchestrator's `try/except`, **no
`DegradedContext.failed_stages` entry is created** — so `FAILED_STAGE_PENALTY (−0.30)` does
**not** apply. What applies is:

| Penalty | Amount |
|---|---|
| F022 `extraction_status == FAILED` | −0.25 |
| F022 `overall_confidence == LOW` | −0.10 |
| **Total** | **−0.35** |

Final score: `1.0 − 0.35 = **0.65**` — between the two thresholds (0.45 and 0.75), which is
exactly the "keep APPROVED but flag for advisory review" outcome TC011 requires.

A clean injected run (TC004, TC012) stays at `1.0`.

## 6. Explicit assumptions & tunable constants

These values are not given numerically by `policy_terms.json` / the assignment, so they're called
out here as documented design decisions:

- **`submission_date` defaults to `treatment_date`** (`ClaimInput._default_submission_date`).
  `submission_rules.deadline_days_from_treatment` (30 days) would reject every 2024-dated test
  case if compared against today's real date. Defaulting to 0 days elapsed still *exercises* the
  deadline check (a caller can pass an explicit `submission_date` to test it) without spuriously
  failing the provided historical test data.
- **Fraud severity weights** (`fraud_detection_agent.SEVERITY_WEIGHTS`): HIGH=0.85, MEDIUM=0.40,
  LOW=0.15, summed and capped at 1.0. Chosen so a single HIGH signal
  (`SAME_DAY_CLAIMS_EXCEEDED`) alone crosses the policy's `fraud_score_manual_review_threshold`
  of 0.80 (TC009).
- **Confidence penalties and thresholds** (`app/core/confidence.py`, §5) — chosen so TC011's
  single failed extraction lands strictly between the advisory and force-review thresholds.
- **DENTAL is exempt from the global per-claim limit** (`policy_evaluation_agent.py` Step 7).
  TC006 is a DENTAL claim for ₹12,000 — above the global `coverage.per_claim_limit` of ₹5,000 —
  but the policy gives DENTAL its own higher `sub_limit` (₹10,000) evaluated at the line-item
  level in Step 9. Applying the global per-claim limit first would REJECT TC006 before dental
  line-item logic ever ran, contradicting its expected `PARTIAL/₹8,000` outcome. TC008
  (CONSULTATION, ₹7,500 > ₹5,000) is unaffected and still correctly REJECTED with
  `PER_CLAIM_EXCEEDED`.

## 7. Resilience contract

**`Orchestrator.process_claim()` never raises.** Each of the five stages runs inside its own
`try/except`:

- `MemberNotFoundError` (member lookup, before Stage 1) → `stopped_early=True` with a message
  asking the member to verify their member ID.
- `PolicyConfigError` (Stages 1 & 3 — e.g. an unknown `claim_category`) → synthetic
  `MANUAL_REVIEW` decision via `_config_error_result()`, with the config error quoted in
  `manual_review_reasons`.
- Any other `Exception` in Stages 1–4 → `DegradedContext.mark_failed(stage, note)`, a best-effort
  default result for that stage (e.g. `VerificationResult(passed=True, ...)`,
  `FraudCheckResult(fraud_score=1.0, requires_manual_review=True)`), and the pipeline
  **continues**. These failures feed directly into the confidence penalty table in §5.
- Stage 2 (Extraction) catches per-document, so one bad document never blocks extraction of the
  others.
- Stage 5 (Decision Agent) is the **final fallback** — if it itself raises, the Orchestrator
  builds a synthetic `MANUAL_REVIEW` `ClaimDecision` inline, with the exception message in
  `notes` and `manual_review_reasons`.

This is exercised end-to-end by TC011 (`simulate_component_failure=True` — Extraction fails for
one document, pipeline still reaches `APPROVED` with reduced confidence and
`manual_review_recommended=True`) and by `tests/core/test_orchestrator.py`'s monkeypatched-
exception tests.

## 8. Injection-mode test harness

`DocumentInput` carries optional fields — `actual_type`, `quality`, `patient_name_on_doc`,
`content` — that, when present, let Stage 1 and Stage 2 skip their Gemini calls entirely and use
the supplied data as ground truth (`source="INJECTED"`). All 12 `test_cases.json` scenarios are
expressed this way, so `eval/run_eval.py` runs the **real** Orchestrator and the **real**
deterministic logic for every stage, end-to-end, with **zero API calls** — fully reproducible,
and exercises exactly the same code paths as the live Gemini-backed demo (the only thing that
changes is where `ExtractedContent`/`DocumentClassification` come from).

## 9. Explainability & UI

- `ClaimTrace.entries: list[TraceEntry]` — every check, classification, mapping decision, and
  final synthesis step is one entry (`stage`, `step`, `status`, human-readable `summary`,
  structured `detail`).
- `ClaimResult` is the top-level object: `stopped_early` + `member_message`, or the full
  `verification` / `extractions` / `policy_evaluation` / `fraud_check` / `decision` + `trace`.
- The Streamlit app (`app/ui/`) has three pages:
  - **Home** — replay any of the 12 `test_cases.json` scenarios (zero API calls), see the
    decision and the full trace.
  - **Submit Claim** — a live form (member/category/amount/documents), supporting both image
    uploads (real Gemini calls) and the same injection-mode fields for a no-API-key demo.
  - **Eval Suite** — runs all 12 cases in the browser, shows the same pass/fail checks as
    `docs/EVAL_REPORT.md`, with a button to regenerate that file.

## 10. Scale analysis — current design vs 10x load

**Current load:** ~75,000 claims/year ≈ 206/day ≈ 9/hour peak. The synchronous sequential
pipeline handles this comfortably — each claim makes 2–9 Gemini API calls (classify + extract
per document, optionally one canonical mapping LLM call), all within Gemini's free-tier limits.

**At 10x (750,000 claims/year ≈ 2,060/day ≈ 86/hour)** the design has three failure points:

### What breaks and how to fix it

**1. Synchronous per-document Gemini calls (biggest bottleneck)**

Today, documents inside a claim are extracted sequentially:
```
doc1 → Gemini classify → Gemini extract
doc2 → Gemini classify → Gemini extract   ← waits for doc1 to finish
```
A 3-document claim with 1-second Gemini latency takes ~6 seconds. At 86 claims/hour with bursts,
this saturates a single process.

Fix: replace `ExtractionAgent.run()` with async calls using the `google-genai` async client, and
run all documents in a claim concurrently:
```python
results = await asyncio.gather(*[extract_doc(doc) for doc in claim.documents])
```
This collapses 6 sequential seconds to ~1 second (slowest doc). The Orchestrator becomes an
`async def process_claim(...)`.

**2. Synchronous HTTP response blocks the caller**

Today, `POST /claim` waits for the full pipeline (seconds) before returning. At 86/hour with
bursty arrivals, slow claims back up the queue.

Fix: decouple submission from processing via a task queue (Celery + Redis, or Google Cloud Tasks).
`POST /claim` returns `{claim_ref, status: "PROCESSING"}` immediately. A worker pool runs
`process_claim()` async, writes the result to persistent storage, and optionally pushes a webhook
or SSE update. The UI polls `/claim/{ref}/result`.

**3. JSON-file `ClaimsLedger` and `PolicyRepository`**

`ClaimsLedger` holds claims history in a single JSON file with no locking — concurrent writers
corrupt it. `PolicyRepository` re-parses the JSON on every cold start.

Fix: replace `ClaimsLedger` with a proper database (PostgreSQL + asyncpg, or Firestore). Cache
`PolicyTerms` once at worker startup; invalidate on policy file change via a config-reload signal.

### What does NOT need to change

- **All five agent classes**: their logic is pure computation over typed models — they scale
  horizontally with no changes.
- **Pydantic contracts**: the input/output schemas remain the contract boundary between agent
  versions; each can be deployed and scaled independently.
- **Confidence scoring and policy evaluation**: fully deterministic, sub-millisecond, no I/O.
- **The injection-mode eval harness**: still zero API calls, still fully deterministic.

### Summary

| Change | Impact | Effort |
|---|---|---|
| Async Gemini calls within a claim | ~6× throughput per worker | Medium |
| Queue-backed async claim processing | Decouples ingestion from processing, handles bursts | Medium |
| PostgreSQL for ClaimsLedger | Concurrent-safe, queryable fraud history | Medium |
| Horizontal worker scaling | Linear throughput with instance count | Low (config only) |

## 12. File layout

```
app/
  models/    common, verification, extraction, policy_eval, fraud, decision, trace, policy, result
  agents/    base, document_verification_agent, extraction_agent, policy_evaluation_agent,
             canonical_mapping, fraud_detection_agent, decision_agent
  core/      orchestrator, confidence, exceptions
  llm/       gemini_client, prompts, schemas
  storage/   policy_repository, claims_ledger
  ui/        streamlit_app, pages/{1_submit_claim, 2_eval_suite}, components/trace_renderer
tests/       agents/, core/, conftest.py
eval/        run_eval.py, eval_helpers.py
mock_documents/ generate_mocks.py, output/*.jpg
docs/        ARCHITECTURE.md, COMPONENT_CONTRACTS.md, EVAL_REPORT.md
```
