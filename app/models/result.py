"""The top-level object returned by the Orchestrator to the UI / eval harness."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.decision import ClaimDecision
from app.models.extraction import ExtractionResult
from app.models.fraud import FraudCheckResult
from app.models.policy_eval import PolicyEvaluationResult
from app.models.trace import ClaimTrace
from app.models.verification import VerificationResult


class ClaimResult(BaseModel):
    claim_ref: str
    stopped_early: bool = False
    verification: Optional[VerificationResult] = None
    extractions: Optional[list[ExtractionResult]] = None
    policy_evaluation: Optional[PolicyEvaluationResult] = None
    fraud_check: Optional[FraudCheckResult] = None
    decision: Optional[ClaimDecision] = None
    trace: ClaimTrace
    member_message: Optional[str] = Field(
        default=None,
        description="Specific, actionable message shown to the member when stopped_early=True.",
    )
