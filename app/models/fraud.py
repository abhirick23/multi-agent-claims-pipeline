"""Contracts for the Fraud Detection Agent (Stage 4)."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.common import ClaimInput
from app.models.policy import MemberRecord, PolicyTerms
from app.models.policy_eval import PolicyEvaluationResult


class FraudSignalCode(str, Enum):
    SAME_DAY_CLAIMS_EXCEEDED = "SAME_DAY_CLAIMS_EXCEEDED"
    MONTHLY_CLAIMS_EXCEEDED = "MONTHLY_CLAIMS_EXCEEDED"
    HIGH_VALUE_CLAIM = "HIGH_VALUE_CLAIM"
    DOCUMENT_ALTERATION = "DOCUMENT_ALTERATION"


class FraudSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FraudSignal(BaseModel):
    code: FraudSignalCode
    severity: FraudSeverity
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class FraudCheckInput(BaseModel):
    claim: ClaimInput
    member: MemberRecord
    policy_eval: PolicyEvaluationResult
    policy: PolicyTerms


class FraudCheckResult(BaseModel):
    fraud_score: float
    signals: list[FraudSignal] = Field(default_factory=list)
    requires_manual_review: bool
