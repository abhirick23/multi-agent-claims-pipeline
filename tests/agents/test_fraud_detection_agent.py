from __future__ import annotations

from datetime import date

from app.agents.extraction_agent import ExtractionAgent
from app.agents.fraud_detection_agent import FraudDetectionAgent
from app.agents.policy_evaluation_agent import PolicyEvaluationAgent
from app.models.common import ClaimHistoryEntry, ClaimInput
from app.models.extraction import ExtractionInput
from app.models.fraud import FraudCheckInput
from app.models.policy_eval import PolicyEvaluationInput
from app.models.trace import ClaimTrace
from tests.conftest import get_case


def _evaluate_policy(policy, case: dict):
    claim = ClaimInput.model_validate(case["input"])
    member = next(m for m in policy.members if m.member_id == claim.member_id)
    trace = ClaimTrace(claim_ref=case["case_id"])

    extraction_agent = ExtractionAgent()
    extractions = [
        extraction_agent.run(ExtractionInput(document=doc, document_type=doc.actual_type, claim_category=claim.claim_category.value), trace)
        for doc in claim.documents
    ]

    policy_eval = PolicyEvaluationAgent().run(PolicyEvaluationInput(claim=claim, member=member, extractions=extractions, policy=policy), trace)
    return claim, member, policy_eval, trace


def test_tc004_clean_claim_no_signals(policy, test_cases):
    case = get_case(test_cases, "TC004")
    claim, member, policy_eval, trace = _evaluate_policy(policy, case)

    result = FraudDetectionAgent().run(FraudCheckInput(claim=claim, member=member, policy_eval=policy_eval, policy=policy), trace)

    assert result.signals == []
    assert result.fraud_score == 0
    assert result.requires_manual_review is False


def test_tc009_same_day_claims_exceeded(policy, test_cases):
    case = get_case(test_cases, "TC009")
    claim, member, policy_eval, trace = _evaluate_policy(policy, case)

    result = FraudDetectionAgent().run(FraudCheckInput(claim=claim, member=member, policy_eval=policy_eval, policy=policy), trace)

    assert result.requires_manual_review is True
    assert result.fraud_score >= policy.fraud_thresholds.fraud_score_manual_review_threshold
    signal_codes = [s.code for s in result.signals]
    assert "SAME_DAY_CLAIMS_EXCEEDED" in signal_codes
    same_day = next(s for s in result.signals if s.code == "SAME_DAY_CLAIMS_EXCEEDED")
    assert same_day.severity == "HIGH"
    assert "4" in same_day.message


def test_monthly_claims_exceeded(policy, test_cases):
    case = get_case(test_cases, "TC004")
    claim, member, policy_eval, trace = _evaluate_policy(policy, case)

    history = [
        ClaimHistoryEntry(claim_id=f"CLM_{i}", date=date(claim.treatment_date.year, claim.treatment_date.month, 1), amount=500)
        for i in range(6)
    ]
    claim_with_history = claim.model_copy(update={"claims_history": history})

    result = FraudDetectionAgent().run(FraudCheckInput(claim=claim_with_history, member=member, policy_eval=policy_eval, policy=policy), trace)

    signal_codes = [s.code for s in result.signals]
    assert "MONTHLY_CLAIMS_EXCEEDED" in signal_codes
    monthly = next(s for s in result.signals if s.code == "MONTHLY_CLAIMS_EXCEEDED")
    assert monthly.severity == "MEDIUM"


def test_high_value_claim_signal(policy, test_cases):
    case = get_case(test_cases, "TC004")
    claim, member, policy_eval, trace = _evaluate_policy(policy, case)

    high_value_claim = claim.model_copy(update={"claimed_amount": policy.fraud_thresholds.high_value_claim_threshold + 1})

    result = FraudDetectionAgent().run(FraudCheckInput(claim=high_value_claim, member=member, policy_eval=policy_eval, policy=policy), trace)

    signal_codes = [s.code for s in result.signals]
    assert "HIGH_VALUE_CLAIM" in signal_codes
    assert result.requires_manual_review is True  # also exceeds auto_manual_review_above
