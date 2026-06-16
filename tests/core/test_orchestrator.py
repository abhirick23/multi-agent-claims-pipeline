from __future__ import annotations

import pytest

from app.core.orchestrator import Orchestrator
from app.models.common import ClaimInput
from app.storage.claims_ledger import ClaimsLedger
from app.storage.policy_repository import PolicyRepository
from tests.conftest import get_case


@pytest.fixture
def orchestrator(tmp_path):
    return Orchestrator(policy_repository=PolicyRepository(), claims_ledger=ClaimsLedger(tmp_path / "claims_ledger.json"))


def test_tc001_wrong_document_stops_early(orchestrator, test_cases):
    case = get_case(test_cases, "TC001")
    claim = ClaimInput.model_validate(case["input"])

    result = orchestrator.process_claim(claim, claim_ref="TC001")

    assert result.stopped_early is True
    assert result.decision is None
    assert "PRESCRIPTION" in result.member_message
    assert "HOSPITAL_BILL" in result.member_message


def test_tc002_unreadable_document_stops_early(orchestrator, test_cases):
    case = get_case(test_cases, "TC002")
    claim = ClaimInput.model_validate(case["input"])

    result = orchestrator.process_claim(claim, claim_ref="TC002")

    assert result.stopped_early is True
    assert "re-upload" in result.member_message.lower()


def test_tc003_patient_name_mismatch_stops_early(orchestrator, test_cases):
    case = get_case(test_cases, "TC003")
    claim = ClaimInput.model_validate(case["input"])

    result = orchestrator.process_claim(claim, claim_ref="TC003")

    assert result.stopped_early is True
    assert "Rajesh Kumar" in result.member_message
    assert "Arjun Mehta" in result.member_message


def test_tc004_happy_path_full_pipeline(orchestrator, test_cases):
    case = get_case(test_cases, "TC004")
    claim = ClaimInput.model_validate(case["input"])

    result = orchestrator.process_claim(claim, claim_ref="TC004")

    assert result.stopped_early is False
    assert result.decision.decision == "APPROVED"
    assert result.decision.approved_amount == 1350
    assert result.decision.confidence_score == 1.0


def test_tc009_fraud_signal_routes_to_manual_review(orchestrator, test_cases):
    case = get_case(test_cases, "TC009")
    claim = ClaimInput.model_validate(case["input"])

    result = orchestrator.process_claim(claim, claim_ref="TC009")

    assert result.stopped_early is False
    assert result.decision.decision == "MANUAL_REVIEW"
    assert result.fraud_check.requires_manual_review is True
    assert any("SAME_DAY_CLAIMS_EXCEEDED" == s.code for s in result.fraud_check.signals)


def test_tc011_component_failure_degrades_gracefully(orchestrator, test_cases):
    case = get_case(test_cases, "TC011")
    claim = ClaimInput.model_validate(case["input"])
    assert claim.simulate_component_failure is True

    result = orchestrator.process_claim(claim, claim_ref="TC011")

    assert result.stopped_early is False
    assert result.decision.decision == "APPROVED"
    assert result.decision.approved_amount == 4000
    assert any(e.extraction_status == "FAILED" for e in result.extractions)
    assert result.decision.confidence_score < 1.0
    assert result.decision.manual_review_recommended is True
    assert any("extraction failed" in note.lower() for note in result.decision.notes)


def test_unknown_member_returns_stopped_early_with_message(orchestrator, test_cases):
    case = get_case(test_cases, "TC004")
    claim = ClaimInput.model_validate(case["input"]).model_copy(update={"member_id": "EMP999"})

    result = orchestrator.process_claim(claim, claim_ref="TC_UNKNOWN_MEMBER")

    assert result.stopped_early is True
    assert "EMP999" in result.member_message


def test_unexpected_exception_in_a_stage_does_not_crash(orchestrator, test_cases, monkeypatch):
    case = get_case(test_cases, "TC004")
    claim = ClaimInput.model_validate(case["input"])

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator._policy_agent, "run", boom)

    result = orchestrator.process_claim(claim, claim_ref="TC_BOOM")

    assert result.stopped_early is False
    assert result.decision.decision == "MANUAL_REVIEW"
