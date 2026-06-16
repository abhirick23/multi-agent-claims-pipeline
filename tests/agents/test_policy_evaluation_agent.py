from __future__ import annotations

from app.agents.extraction_agent import ExtractionAgent
from app.agents.policy_evaluation_agent import PolicyEvaluationAgent
from app.models.common import ClaimInput
from app.models.extraction import ExtractionInput
from app.models.policy_eval import PolicyEvaluationInput
from app.models.trace import ClaimTrace
from tests.conftest import get_case


def _evaluate(policy, case: dict):
    claim = ClaimInput.model_validate(case["input"])
    member = next(m for m in policy.members if m.member_id == claim.member_id)
    trace = ClaimTrace(claim_ref=case["case_id"])

    extraction_agent = ExtractionAgent()
    extractions = [
        extraction_agent.run(ExtractionInput(document=doc, document_type=doc.actual_type, claim_category=claim.claim_category.value), trace)
        for doc in claim.documents
    ]

    agent = PolicyEvaluationAgent()
    result = agent.run(PolicyEvaluationInput(claim=claim, member=member, extractions=extractions, policy=policy), trace)
    return result, trace


def test_tc004_approved_consultation(policy, test_cases):
    case = get_case(test_cases, "TC004")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "APPROVED"
    assert result.approved_amount == 1350
    assert result.financial_breakdown.network_hospital is False


def test_tc005_waiting_period_diabetes(policy, test_cases):
    case = get_case(test_cases, "TC005")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "REJECTED"
    assert result.rejection_reasons == ["WAITING_PERIOD"]
    fail_check = next(c for c in result.checks if c.code == "WAITING_PERIOD" and c.status == "FAIL")
    assert "2024-11-30" in fail_check.message
    assert result.canonical_mapping.waiting_period_key == "diabetes"


def test_tc006_dental_partial_cosmetic_exclusion(policy, test_cases):
    case = get_case(test_cases, "TC006")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "PARTIAL"
    assert result.approved_amount == 8000
    by_desc = {li.description: li for li in result.line_item_results}
    assert by_desc["Root Canal Treatment"].status == "APPROVED"
    assert by_desc["Teeth Whitening"].status == "REJECTED"
    assert by_desc["Teeth Whitening"].reason is not None


def test_tc007_pre_auth_missing(policy, test_cases):
    case = get_case(test_cases, "TC007")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "REJECTED"
    assert result.rejection_reasons == ["PRE_AUTH_MISSING"]
    fail_check = next(c for c in result.checks if c.code == "PRE_AUTH_MISSING")
    assert "pre-authorization" in fail_check.message.lower()
    assert "resubmit" in fail_check.message.lower()


def test_tc008_per_claim_limit_exceeded(policy, test_cases):
    case = get_case(test_cases, "TC008")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "REJECTED"
    assert result.rejection_reasons == ["PER_CLAIM_EXCEEDED"]
    fail_check = next(c for c in result.checks if c.code == "PER_CLAIM_EXCEEDED")
    assert "7,500" in fail_check.message
    assert "5,000" in fail_check.message


def test_tc010_network_discount_before_copay(policy, test_cases):
    case = get_case(test_cases, "TC010")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "APPROVED"
    assert result.approved_amount == 3240
    fb = result.financial_breakdown
    assert fb.network_hospital is True
    assert fb.network_discount_percent == 20
    assert fb.amount_after_discount == 3600
    assert fb.copay_percent == 10
    assert fb.approved_amount == 3240


def test_tc011_alternative_medicine_within_sublimit(policy, test_cases):
    case = get_case(test_cases, "TC011")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "APPROVED"
    assert result.approved_amount == 4000


def test_tc012_excluded_obesity_treatment(policy, test_cases):
    case = get_case(test_cases, "TC012")
    result, _trace = _evaluate(policy, case)

    assert result.decision_hint == "REJECTED"
    assert result.rejection_reasons == ["EXCLUDED_CONDITION"]
    assert result.canonical_mapping.method == "KEYWORD_MATCH"
    matched_terms = [m.policy_term for m in result.canonical_mapping.exclusion_matches]
    assert "Obesity and weight loss programs" in matched_terms
    # Waiting period must not have fired instead -- exclusion takes precedence.
    assert result.canonical_mapping.waiting_period_key is None
