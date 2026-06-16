"""Contracts for the Decision Agent (Stage 5) -- the final synthesis step."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.common import ClaimInput
from app.models.extraction import ExtractionResult
from app.models.fraud import FraudCheckResult
from app.models.policy_eval import FinancialBreakdown, LineItemResult, PolicyEvaluationResult
from app.models.trace import DegradedContext
from app.models.verification import VerificationResult


class RejectionReason(BaseModel):
    code: str
    message: str
    policy_reference: Optional[str] = None


class DecisionInput(BaseModel):
    claim: ClaimInput
    verification: VerificationResult
    extractions: list[ExtractionResult] = Field(default_factory=list)
    policy_eval: PolicyEvaluationResult
    fraud: FraudCheckResult
    degraded_context: DegradedContext = Field(default_factory=DegradedContext)


class ClaimDecision(BaseModel):
    decision: Literal["APPROVED", "PARTIAL", "REJECTED", "MANUAL_REVIEW"]
    approved_amount: float = 0
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)
    confidence_score: float
    notes: list[str] = Field(default_factory=list)
    line_item_breakdown: Optional[list[LineItemResult]] = None
    financial_breakdown: Optional[FinancialBreakdown] = None
    manual_review_recommended: bool = False
    manual_review_reasons: list[str] = Field(default_factory=list)
