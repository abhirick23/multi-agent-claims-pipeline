from __future__ import annotations

from app.agents.decision_agent import DecisionAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.fraud_detection_agent import FraudDetectionAgent
from app.agents.policy_evaluation_agent import PolicyEvaluationAgent
from app.core.confidence import CONFIDENCE_ADVISORY_REVIEW
from app.models.common import ClaimInput
from app.models.decision import DecisionInput
from app.models.extraction import ExtractionInput
from app.models.fraud import FraudCheckInput
from app.models.policy_eval import PolicyEvaluationInput
from app.models.trace import ClaimTrace
from app.models.verification import VerificationResult
from tests.conftest import get_case


def _run_pipeline(policy, case: dict, simulate_failure_file_ids: set[str] = frozenset()):
    claim = ClaimInput.model_validate(case["input"])
    member = next(m for m in policy.members if m.member_id == claim.member_id)
    trace = ClaimTrace(claim_ref=case["case_id"])

    extraction_agent = ExtractionAgent()
    extractions = [
        extraction_agent.run(
            ExtractionInput(
                document=doc,
                document_type=doc.actual_type,
                claim_category=claim.claim_category.value,
                simulate_failure=doc.file_id in simulate_failure_file_ids,
            ),
            trace,
        )
        for doc in claim.documents
    ]

    policy_eval = PolicyEvaluationAgent().run(PolicyEvaluationInput(claim=claim, member=member, extractions=extractions, policy=policy), trace)
    fraud = FraudDetectionAgent().run(FraudCheckInput(claim=claim, member=member, policy_eval=policy_eval, policy=policy), trace)
    decision = DecisionAgent().run(
        DecisionInput(
            claim=claim,
            verification=VerificationResult(passed=True, issues=[], classified_documents=[]),
            extractions=extractions,
            policy_eval=policy_eval,
            fraud=fraud,
        ),
        trace,
    )
    return decision, trace


def test_tc004_full_confidence_approval(policy, test_cases):
    case = get_case(test_cases, "TC004")
    decision, _trace = _run_pipeline(policy, case)

    assert decision.decision == "APPROVED"
    assert decision.approved_amount == 1350
    assert decision.confidence_score == 1.0
    assert decision.manual_review_recommended is False
    assert decision.rejection_reasons == []


def test_tc005_rejection_reasons_populated(policy, test_cases):
    case = get_case(test_cases, "TC005")
    decision, _trace = _run_pipeline(policy, case)

    assert decision.decision == "REJECTED"
    assert decision.approved_amount == 0
    assert len(decision.rejection_reasons) == 1
    reason = decision.rejection_reasons[0]
    assert reason.code == "WAITING_PERIOD"
    assert "2024-11-30" in reason.message


def test_tc006_partial_line_item_breakdown(policy, test_cases):
    case = get_case(test_cases, "TC006")
    decision, _trace = _run_pipeline(policy, case)

    assert decision.decision == "PARTIAL"
    assert decision.approved_amount == 8000
    assert decision.line_item_breakdown is not None
    by_desc = {li.description: li for li in decision.line_item_breakdown}
    assert by_desc["Root Canal Treatment"].status == "APPROVED"
    assert by_desc["Teeth Whitening"].status == "REJECTED"
    assert by_desc["Teeth Whitening"].reason is not None


def test_tc009_fraud_triggers_manual_review(policy, test_cases):
    case = get_case(test_cases, "TC009")
    decision, _trace = _run_pipeline(policy, case)

    assert decision.decision == "MANUAL_REVIEW"
    assert len(decision.manual_review_reasons) >= 1
    assert any("same-day" in reason.lower() or "claim #" in reason.lower() for reason in decision.manual_review_reasons)


def test_tc011_degraded_extraction_recommends_review(policy, test_cases):
    case = get_case(test_cases, "TC011")
    decision, _trace = _run_pipeline(policy, case, simulate_failure_file_ids={"F022"})

    assert decision.decision == "APPROVED"
    assert decision.approved_amount == 4000
    assert decision.confidence_score < CONFIDENCE_ADVISORY_REVIEW
    assert decision.manual_review_recommended is True
    assert any("extraction failed" in note.lower() for note in decision.notes)
    assert any("manual review" in reason.lower() for reason in decision.manual_review_reasons)


def test_tc012_excluded_condition_rejection(policy, test_cases):
    case = get_case(test_cases, "TC012")
    decision, _trace = _run_pipeline(policy, case)

    assert decision.decision == "REJECTED"
    assert decision.approved_amount == 0
    assert decision.rejection_reasons[0].code == "EXCLUDED_CONDITION"
    assert decision.confidence_score > 0.9
