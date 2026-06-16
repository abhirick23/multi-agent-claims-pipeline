# Component Contracts

Every agent is a plain class with one `run(input, trace) -> output` method. All inputs/outputs
are Pydantic v2 models (`app/models/`), so contracts are enforced by validation, not convention.
`trace: ClaimTrace` is threaded through every call and mutated in place (`trace.add(...)`); it is
not part of any agent's return value.

This document lists every contract field, grouped by pipeline stage. Enum values are shown in
`UPPER_CASE`.

---

## Shared types (`app/models/common.py`)

### `DocumentInput`
| Field | Type | Notes |
|---|---|---|
| `file_id` | `str` | required |
| `file_name` | `str \| None` | |
| `file_path` | `str \| None` | live mode: path to image for Gemini vision |
| `actual_type` | `DocumentType \| None` | injection mode: skips classification |
| `quality` | `DocumentQuality \| None` | injection mode |
| `patient_name_on_doc` | `str \| None` | injection mode |
| `content` | `dict \| None` | injection mode: skips extraction, validated as `ExtractedContent` |

`DocumentType`: `PRESCRIPTION, HOSPITAL_BILL, LAB_REPORT, PHARMACY_BILL, DISCHARGE_SUMMARY,
DENTAL_REPORT, DIAGNOSTIC_REPORT, UNKNOWN`
`DocumentQuality`: `GOOD, POOR, UNREADABLE`
`ClaimCategory`: `CONSULTATION, DIAGNOSTIC, PHARMACY, DENTAL, VISION, ALTERNATIVE_MEDICINE`

### `ClaimHistoryEntry`
`claim_id: str`, `date: date`, `amount: float`, `provider: str | None`

### `ClaimInput` — the pipeline's entry point
| Field | Type | Default |
|---|---|---|
| `member_id` | `str` | required |
| `policy_id` | `str` | required |
| `claim_category` | `ClaimCategory` | required |
| `treatment_date` | `date` | required |
| `submission_date` | `date \| None` | defaults to `treatment_date` (see ARCHITECTURE §6) |
| `claimed_amount` | `float` | required |
| `hospital_name` | `str \| None` | `None` |
| `pre_auth_obtained` | `bool` | `False` |
| `ytd_claims_amount` | `float` | `0` |
| `claims_history` | `list[ClaimHistoryEntry]` | `[]` |
| `simulate_component_failure` | `bool` | `False` |
| `documents` | `list[DocumentInput]` | `[]` |

---

## Stage 1 — Document Verification Agent

`app/agents/document_verification_agent.py` · `DocumentVerificationAgent.run`

**Input — `VerificationInput`** (`app/models/verification.py`)
| Field | Type |
|---|---|
| `claim` | `ClaimInput` |
| `policy` | `PolicyTerms` |

**Output — `VerificationResult`**
| Field | Type | Notes |
|---|---|---|
| `passed` | `bool` | `False` if any `BLOCKING` issue |
| `issues` | `list[VerificationIssue]` | |
| `classified_documents` | `list[DocumentClassification]` | one per uploaded document |

**`VerificationIssue`**: `code` (`WRONG_DOCUMENT_TYPE \| MISSING_REQUIRED_DOCUMENT \|
UNREADABLE_DOCUMENT \| PATIENT_NAME_MISMATCH`), `severity` (`BLOCKING \| WARNING`), `file_id:
str | None`, `message: str` (always member-facing and specific — names document types / patient
names involved), `detail: dict`.

**`DocumentClassification`**: `file_id`, `document_type: DocumentType`, `quality:
DocumentQuality`, `patient_name_on_doc: str | None`, `source` (`INJECTED \| VISION_LLM`).

**Orchestrator contract**: if `not passed`, the Orchestrator returns immediately with
`ClaimResult(stopped_early=True, decision=None, member_message=<joined BLOCKING messages>)` —
Stages 2–5 never run.

---

## Stage 2 — Extraction Agent

`app/agents/extraction_agent.py` · `ExtractionAgent.run` — called once per document.

**Input — `ExtractionInput`** (`app/models/extraction.py`)
| Field | Type | Notes |
|---|---|---|
| `document` | `DocumentInput` | |
| `document_type` | `DocumentType` | from Stage 1's classification |
| `claim_category` | `str` | |
| `simulate_failure` | `bool` | set by Orchestrator for TC011 |

**Output — `ExtractionResult`**
| Field | Type | Notes |
|---|---|---|
| `file_id` | `str` | |
| `document_type` | `DocumentType` | |
| `content` | `ExtractedContent` | see below |
| `field_confidences` | `list[FieldConfidence]` | live mode only |
| `overall_confidence` | `ConfidenceLevel` (`HIGH \| MEDIUM \| LOW`) | |
| `extraction_status` | `"SUCCESS" \| "PARTIAL" \| "FAILED"` | |
| `source` | `"INJECTED" \| "VISION_LLM"` | |
| `error` | `str \| None` | set when `FAILED` |

**`ExtractedContent`** — single flat schema covering all document types:
`doctor_name`, `doctor_registration`, `patient_name`, `date`, `diagnosis`, `treatment`,
`medicines: list[str]`, `tests_ordered: list[str]`, `hospital_name`, `line_items:
list[LineItem]` (`description: str, amount: float`), `total: float | None`, `lab_name`,
`test_results: list[dict]` — all optional except the list fields (default `[]`).

**`FieldConfidence`**: `field_name: str`, `confidence: ConfidenceLevel`, `reason: str | None`.

---

## Stage 3 — Policy Evaluation Agent (+ Canonical Mapping)

`app/agents/policy_evaluation_agent.py` · `PolicyEvaluationAgent.run`
Sub-step: `app/agents/canonical_mapping.py` · `CanonicalMapper.map` (called internally; not
invoked directly by the Orchestrator).

**Input — `PolicyEvaluationInput`** (`app/models/policy_eval.py`)
| Field | Type | Notes |
|---|---|---|
| `claim` | `ClaimInput` | |
| `member` | `MemberRecord` | from `PolicyRepository.get_member()` |
| `extractions` | `list[ExtractionResult]` | all Stage 2 outputs |
| `policy` | `PolicyTerms` | |
| `degraded_context` | `DegradedContext` | accumulated so far |

**Output — `PolicyEvaluationResult`**
| Field | Type | Notes |
|---|---|---|
| `decision_hint` | `"APPROVED" \| "PARTIAL" \| "REJECTED" \| "MANUAL_REVIEW"` | input to Stage 5 |
| `checks` | `list[PolicyCheckResult]` | every rule evaluated, in order |
| `rejection_reasons` | `list[str]` | check `code`s that caused `REJECTED` |
| `approved_amount` | `float \| None` | pre-fraud/confidence; `None` if rejected |
| `financial_breakdown` | `FinancialBreakdown \| None` | |
| `line_item_results` | `list[LineItemResult] \| None` | DENTAL only |
| `canonical_mapping` | `CanonicalMapping` | always populated |

**`PolicyCheckResult`**: `code: str`, `status` (`PASS \| FAIL \| WARN \| INFO`), `message: str`,
`policy_reference: str | None` (dotted path into `policy_terms.json`), `detail: dict`.

**`FinancialBreakdown`**: `claimed_amount`, `eligible_base`, `network_hospital: bool`,
`network_discount_percent`, `amount_after_discount`, `copay_percent`, `copay_amount`,
`approved_amount` — all `float` except `network_hospital`. Discount is applied to
`eligible_base` to get `amount_after_discount`; co-pay is applied to `amount_after_discount` to
get `approved_amount` (discount **before** co-pay — TC010).

**`LineItemResult`**: `description: str`, `amount: float`, `status` (`APPROVED \| REJECTED`),
`reason: str | None` (always set when `REJECTED`).

**`CanonicalMapping`** — the diagnosis/treatment → policy-vocabulary mapping, always recorded:
| Field | Type |
|---|---|
| `waiting_period_key` | `str \| None` — key into `policy.waiting_periods.specific_conditions` |
| `exclusion_matches` | `list[ExclusionMatch]` |
| `dental_procedures` | `list[DentalProcedureClassification]` |
| `tests_ordered` | `list[str]` |
| `method` | `"KEYWORD_MATCH" \| "LLM_ASSISTED" \| "NONE"` |
| `raw_diagnosis_text` / `raw_treatment_text` | `str \| None` |
| `rationale` | `str \| None` — set when `method == "LLM_ASSISTED"` |

**`ExclusionMatch`**: `policy_term: str`, `matched_via: str`, `scope` (`WHOLE_CLAIM \|
LINE_ITEM`), `confidence: float`, `line_item_ref: str | None`.

**`DentalProcedureClassification`**: `description: str`, `amount: float`, `status` (`COVERED \|
EXCLUDED \| UNKNOWN`), `matched_via: str | None`.

**Errors**: an unknown `claim_category` (not in `policy.opd_categories` /
`policy.document_requirements`) raises `PolicyConfigError` — caught by the Orchestrator and
turned into a `MANUAL_REVIEW` decision for the whole claim (see ARCHITECTURE §7).

---

## Stage 4 — Fraud Detection Agent

`app/agents/fraud_detection_agent.py` · `FraudDetectionAgent.run`

**Input — `FraudCheckInput`** (`app/models/fraud.py`)
| Field | Type | Notes |
|---|---|---|
| `claim` | `ClaimInput` | `claims_history` merged with `ClaimsLedger` by the Orchestrator |
| `member` | `MemberRecord` | |
| `policy_eval` | `PolicyEvaluationResult` | not currently used for scoring, available for future signals |
| `policy` | `PolicyTerms` | |

**Output — `FraudCheckResult`**
| Field | Type | Notes |
|---|---|---|
| `fraud_score` | `float` | weighted sum of signal severities, capped at 1.0 |
| `signals` | `list[FraudSignal]` | |
| `requires_manual_review` | `bool` | see formula below |

**`FraudSignal`**: `code` (`SAME_DAY_CLAIMS_EXCEEDED \| MONTHLY_CLAIMS_EXCEEDED \|
HIGH_VALUE_CLAIM \| DOCUMENT_ALTERATION`), `severity` (`LOW \| MEDIUM \| HIGH`), `message: str`
(member/ops-facing, includes the actual counts/amounts and thresholds), `detail: dict`.

**Formula**:
```
fraud_score = min(1.0, sum(SEVERITY_WEIGHT[s.severity] for s in signals))
  where SEVERITY_WEIGHT = {HIGH: 0.85, MEDIUM: 0.40, LOW: 0.15}

requires_manual_review = (
    fraud_score >= policy.fraud_thresholds.fraud_score_manual_review_threshold
    or claim.claimed_amount > policy.fraud_thresholds.auto_manual_review_above
    or any(signal.severity == HIGH for signal in signals)
)
```

**Errors**: pure computation, no LLM calls — no expected runtime failures. If this stage *does*
raise unexpectedly, the Orchestrator substitutes
`FraudCheckResult(fraud_score=1.0, signals=[], requires_manual_review=True)` (fail safe toward
manual review).

---

## Stage 5 — Decision Agent (+ Confidence Scoring)

`app/agents/decision_agent.py` · `DecisionAgent.run`
Sub-step: `app/core/confidence.py` · `compute_confidence()` (pure function, not an agent).

**Input — `DecisionInput`** (`app/models/decision.py`)
| Field | Type |
|---|---|
| `claim` | `ClaimInput` |
| `verification` | `VerificationResult` |
| `extractions` | `list[ExtractionResult]` |
| `policy_eval` | `PolicyEvaluationResult` |
| `fraud` | `FraudCheckResult` |
| `degraded_context` | `DegradedContext` |

**Output — `ClaimDecision`**
| Field | Type | Notes |
|---|---|---|
| `decision` | `"APPROVED" \| "PARTIAL" \| "REJECTED" \| "MANUAL_REVIEW"` | |
| `approved_amount` | `float` | `0` unless `APPROVED`/`PARTIAL` |
| `rejection_reasons` | `list[RejectionReason]` | `code`, `message`, `policy_reference` |
| `confidence_score` | `float` | `0.0`–`1.0`, see `compute_confidence` |
| `notes` | `list[str]` | human-readable explanations of every confidence deduction + fraud signals |
| `line_item_breakdown` | `list[LineItemResult] \| None` | DENTAL only |
| `financial_breakdown` | `FinancialBreakdown \| None` | `None` unless `APPROVED`/`PARTIAL` |
| `manual_review_recommended` | `bool` | advisory flag, decision unchanged |
| `manual_review_reasons` | `list[str]` | why fraud/confidence triggered review |

**`compute_confidence(extractions, policy_eval, degraded_context) -> (score, notes)`** — see
ARCHITECTURE §5 for the full penalty table; returns a value clamped to `[0, 1]`.

**Decision precedence** (highest wins):
1. `fraud.requires_manual_review` → `MANUAL_REVIEW` (fraud signal messages copied into `notes`
   and `manual_review_reasons`).
2. Else `policy_eval.decision_hint`.
3. `confidence_score < CONFIDENCE_FORCE_MANUAL_REVIEW (0.45)` → forces `MANUAL_REVIEW`
   regardless of 1/2.
4. `confidence_score < CONFIDENCE_ADVISORY_REVIEW (0.75)` (and not already `MANUAL_REVIEW`) →
   decision unchanged, `manual_review_recommended=True`.

**Errors**: this is the pipeline's last-resort fallback (ARCHITECTURE §7). If `run()` itself
raises, the Orchestrator builds a synthetic `ClaimDecision(decision="MANUAL_REVIEW",
confidence_score=0.0, notes=[f"Decision agent failed: {exc}"], manual_review_recommended=True,
manual_review_reasons=[f"Decision agent failed: {exc}"])` directly — `process_claim()` still
returns normally.

---

## Trace & top-level result (`app/models/trace.py`, `app/models/result.py`)

**`TraceEntry`**: `stage` (`VERIFICATION \| EXTRACTION \| POLICY_EVALUATION \| FRAUD_DETECTION \|
DECISION \| ORCHESTRATOR`), `step: str`, `status` (`SUCCESS \| DEGRADED \| FAILED \| INFO`),
`summary: str`, `detail: dict`, `timestamp: datetime`.

**`ClaimTrace`**: `claim_ref: str`, `entries: list[TraceEntry]`, with `.add(...)` appending and
returning a new entry.

**`DegradedContext`**: `failed_stages: list[str]`, `degraded_stages: list[str]`, `notes:
list[str]` — populated by the Orchestrator's exception handlers, consumed by
`compute_confidence`.

**`ClaimResult`** — returned by `Orchestrator.process_claim()`:
| Field | Type | Notes |
|---|---|---|
| `claim_ref` | `str` | |
| `stopped_early` | `bool` | `True` only for Stage-1 BLOCKING issues or unknown `member_id` |
| `verification` | `VerificationResult \| None` | |
| `extractions` | `list[ExtractionResult] \| None` | |
| `policy_evaluation` | `PolicyEvaluationResult \| None` | |
| `fraud_check` | `FraudCheckResult \| None` | |
| `decision` | `ClaimDecision \| None` | `None` iff `stopped_early` |
| `trace` | `ClaimTrace` | always populated |
| `member_message` | `str \| None` | populated iff `stopped_early` |

---

## Orchestrator (`app/core/orchestrator.py`)

```python
Orchestrator(
    policy_repository: PolicyRepository | None = None,
    claims_ledger: ClaimsLedger | None = None,
    gemini_client: GeminiClient | None = None,
)

.process_claim(
    claim: ClaimInput,
    claim_ref: str | None = None,        # defaults to "{member_id}_{treatment_date}_{category}"
    record_in_ledger: bool = True,        # append this claim to ClaimsLedger after a decision
) -> ClaimResult
```

`process_claim()` never raises — see ARCHITECTURE §7 for the full per-stage fallback table.

## LLM layer (`app/llm/`)

- **`GeminiClient`** (`gemini_client.py`) — `classify_document(image_path)`,
  `extract_content(image_path, document_type)`, `classify_canonical_category(diagnosis_text,
  treatment_text, waiting_period_keys, exclusion_terms)`. All three use
  `response_schema=<Pydantic model>` so responses validate directly; retries with exponential
  backoff (`DEFAULT_MAX_RETRIES=3`, `DEFAULT_BACKOFF_SECONDS=2.0`) before raising
  `GeminiAPIError`.
- **`schemas.py`** — `GeminiDocumentClassification` (`document_type`, `quality`,
  `patient_name_on_doc`), `GeminiCanonicalMappingResponse` (`waiting_period_key`,
  `matched_exclusion_terms`, `confidence`, `rationale`). Deliberately narrower than the full
  pipeline models — Gemini only fills fields it can observe/reason about; bookkeeping fields
  (`file_id`, `source`, etc.) are filled in by the calling agent.
- **`prompts.py`** — prompt templates for the three calls above.

## Storage (`app/storage/`)

- **`PolicyRepository`** (`policy_repository.py`) — loads `policy_terms.json` into typed
  `PolicyTerms`; `get_member(member_id)` (raises `MemberNotFoundError`),
  `get_effective_join_date(member)` (dependents inherit the primary member's `join_date`),
  `get_category_rules(claim_category)`, `get_document_requirements(claim_category)` (both raise
  `PolicyConfigError` if the category is unknown).
- **`ClaimsLedger`** (`claims_ledger.py`) — JSON-file store of `{member_id: [ClaimHistoryEntry,
  ...]}`. `get_history(member_id)`, `append(member_id, entry)`,
  `merged_history(member_id, provided)` (de-dupes by `claim_id`, combines ledger history with
  whatever `ClaimInput.claims_history` already supplied — used so eval-injected history and live
  ledger history compose correctly).
