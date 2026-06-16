"""Contracts for the Policy Evaluation Agent (Stage 3) and its canonical-mapping sub-step."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.models.common import ClaimInput
from app.models.extraction import ExtractionResult
from app.models.policy import MemberRecord, PolicyTerms
from app.models.trace import DegradedContext


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


class PolicyCheckResult(BaseModel):
    code: str
    status: CheckStatus
    message: str
    policy_reference: Optional[str] = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ExclusionMatch(BaseModel):
    policy_term: str
    matched_via: str
    scope: Literal["WHOLE_CLAIM", "LINE_ITEM"]
    confidence: float
    line_item_ref: Optional[str] = None


class DentalProcedureClassification(BaseModel):
    description: str
    amount: float
    status: Literal["COVERED", "EXCLUDED", "UNKNOWN"]
    matched_via: Optional[str] = None


class CanonicalMapping(BaseModel):
    """Output of the diagnosis/treatment -> policy-vocabulary mapping sub-step. Always recorded
    in the trace so ops can audit *why* a given diagnosis did or didn't trigger a waiting period
    or exclusion."""

    waiting_period_key: Optional[str] = None
    exclusion_matches: list[ExclusionMatch] = Field(default_factory=list)
    dental_procedures: list[DentalProcedureClassification] = Field(default_factory=list)
    tests_ordered: list[str] = Field(default_factory=list)
    method: Literal["KEYWORD_MATCH", "LLM_ASSISTED", "NONE"] = "NONE"
    raw_diagnosis_text: Optional[str] = None
    raw_treatment_text: Optional[str] = None
    rationale: Optional[str] = None


class FinancialBreakdown(BaseModel):
    claimed_amount: float
    eligible_base: float
    network_hospital: bool
    network_discount_percent: float
    amount_after_discount: float
    copay_percent: float
    copay_amount: float
    approved_amount: float


class LineItemResult(BaseModel):
    description: str
    amount: float
    status: Literal["APPROVED", "REJECTED"]
    reason: Optional[str] = None


class PolicyEvaluationInput(BaseModel):
    claim: ClaimInput
    member: MemberRecord
    extractions: list[ExtractionResult] = Field(default_factory=list)
    policy: PolicyTerms
    degraded_context: DegradedContext = Field(default_factory=DegradedContext)


class PolicyEvaluationResult(BaseModel):
    decision_hint: Literal["APPROVED", "PARTIAL", "REJECTED", "MANUAL_REVIEW"]
    checks: list[PolicyCheckResult] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    approved_amount: Optional[float] = None
    financial_breakdown: Optional[FinancialBreakdown] = None
    line_item_results: Optional[list[LineItemResult]] = None
    canonical_mapping: CanonicalMapping = Field(default_factory=CanonicalMapping)
