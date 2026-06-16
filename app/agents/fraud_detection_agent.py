"""Stage 4: Fraud Detection Agent.

Pure computation over ``claim.claims_history`` and ``policy.fraud_thresholds`` -- no LLM calls,
nothing that can fail at runtime. Produces a list of ``FraudSignal``s, a weighted
``fraud_score``, and a ``requires_manual_review`` flag consumed by the Decision Agent.

**Severity weights** (tunable constants, not given numerically by ``policy_terms.json``):
HIGH=0.85, MEDIUM=0.40, LOW=0.15, summed and capped at 1.0. A single HIGH signal (e.g. one
``SAME_DAY_CLAIMS_EXCEEDED``) is enough to cross the default
``fraud_score_manual_review_threshold`` of 0.80 on its own.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.models.fraud import FraudCheckInput, FraudCheckResult, FraudSignal, FraudSignalCode, FraudSeverity
from app.models.trace import ClaimTrace, TraceStage, TraceStatus

SEVERITY_WEIGHTS: dict[FraudSeverity, float] = {
    FraudSeverity.HIGH: 0.85,
    FraudSeverity.MEDIUM: 0.40,
    FraudSeverity.LOW: 0.15,
}


def _fmt(amount: float) -> str:
    return f"₹{amount:,.2f}"


class FraudDetectionAgent(BaseAgent):
    def run(self, input: FraudCheckInput, trace: ClaimTrace) -> FraudCheckResult:
        claim, thresholds = input.claim, input.policy.fraud_thresholds
        signals: list[FraudSignal] = []

        same_day_count = 1 + sum(1 for h in claim.claims_history if h.date == claim.treatment_date)
        if same_day_count > thresholds.same_day_claims_limit:
            signals.append(FraudSignal(
                code=FraudSignalCode.SAME_DAY_CLAIMS_EXCEEDED,
                severity=FraudSeverity.HIGH,
                message=(
                    f"This is claim #{same_day_count} for {claim.treatment_date.isoformat()} from this member, "
                    f"exceeding the same-day limit of {thresholds.same_day_claims_limit}."
                ),
                detail={"same_day_claim_count": same_day_count, "limit": thresholds.same_day_claims_limit},
            ))

        treatment_month = (claim.treatment_date.year, claim.treatment_date.month)
        monthly_count = 1 + sum(1 for h in claim.claims_history if (h.date.year, h.date.month) == treatment_month)
        if monthly_count > thresholds.monthly_claims_limit:
            signals.append(FraudSignal(
                code=FraudSignalCode.MONTHLY_CLAIMS_EXCEEDED,
                severity=FraudSeverity.MEDIUM,
                message=(
                    f"This is claim #{monthly_count} for {claim.treatment_date.strftime('%B %Y')} from this member, "
                    f"exceeding the monthly limit of {thresholds.monthly_claims_limit}."
                ),
                detail={"monthly_claim_count": monthly_count, "limit": thresholds.monthly_claims_limit},
            ))

        if claim.claimed_amount > thresholds.high_value_claim_threshold:
            signals.append(FraudSignal(
                code=FraudSignalCode.HIGH_VALUE_CLAIM,
                severity=FraudSeverity.MEDIUM,
                message=(
                    f"The claimed amount ({_fmt(claim.claimed_amount)}) exceeds the high-value claim "
                    f"threshold of {_fmt(thresholds.high_value_claim_threshold)}."
                ),
                detail={"claimed_amount": claim.claimed_amount, "threshold": thresholds.high_value_claim_threshold},
            ))

        fraud_score = min(1.0, round(sum(SEVERITY_WEIGHTS[s.severity] for s in signals), 2))
        requires_manual_review = (
            fraud_score >= thresholds.fraud_score_manual_review_threshold
            or claim.claimed_amount > thresholds.auto_manual_review_above
            or any(s.severity == FraudSeverity.HIGH for s in signals)
        )

        trace.add(
            TraceStage.FRAUD_DETECTION, "fraud_check",
            TraceStatus.SUCCESS if not requires_manual_review else TraceStatus.INFO,
            f"Fraud score {fraud_score:.2f} from {len(signals)} signal(s); "
            f"requires_manual_review={requires_manual_review}.",
            detail={
                "signals": [s.model_dump() for s in signals],
                "fraud_score": fraud_score,
                "requires_manual_review": requires_manual_review,
            },
        )

        return FraudCheckResult(fraud_score=fraud_score, signals=signals, requires_manual_review=requires_manual_review)
