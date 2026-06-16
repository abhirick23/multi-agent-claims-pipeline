from __future__ import annotations

from app.agents.document_verification_agent import DocumentVerificationAgent
from app.models.common import ClaimInput
from app.models.trace import ClaimTrace, TraceStatus
from app.models.verification import VerificationInput
from tests.conftest import get_case


def _run(policy, case: dict):
    claim = ClaimInput.model_validate(case["input"])
    trace = ClaimTrace(claim_ref=case["case_id"])
    agent = DocumentVerificationAgent()
    result = agent.run(VerificationInput(claim=claim, policy=policy), trace)
    return result, trace


def test_tc001_wrong_document_uploaded(policy, test_cases):
    case = get_case(test_cases, "TC001")
    result, _trace = _run(policy, case)

    assert result.passed is False
    blocking = [i for i in result.issues if i.severity == "BLOCKING"]
    assert len(blocking) == 1
    message = blocking[0].message
    # Must name the uploaded type and the required/missing type.
    assert "PRESCRIPTION" in message
    assert "HOSPITAL_BILL" in message


def test_tc002_unreadable_document(policy, test_cases):
    case = get_case(test_cases, "TC002")
    result, _trace = _run(policy, case)

    assert result.passed is False
    blocking = [i for i in result.issues if i.severity == "BLOCKING"]
    assert len(blocking) == 1
    issue = blocking[0]
    assert issue.code.value == "UNREADABLE_DOCUMENT"
    assert issue.file_id == "F004"
    # Must identify the specific document and ask for re-upload.
    assert "blurry_bill.jpg" in issue.message
    assert "re-upload" in issue.message.lower()


def test_tc003_patient_name_mismatch(policy, test_cases):
    case = get_case(test_cases, "TC003")
    result, _trace = _run(policy, case)

    assert result.passed is False
    blocking = [i for i in result.issues if i.severity == "BLOCKING"]
    assert len(blocking) == 1
    message = blocking[0].message
    assert "Rajesh Kumar" in message
    assert "Arjun Mehta" in message


def test_tc004_happy_path_passes(policy, test_cases):
    case = get_case(test_cases, "TC004")
    result, trace = _run(policy, case)

    assert result.passed is True
    assert result.issues == []
    assert len(result.classified_documents) == 2
    assert any(e.step == "result" and e.status == TraceStatus.SUCCESS for e in trace.entries)
