from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agents.extraction_agent import ExtractionAgent
from app.core.exceptions import GeminiAPIError
from app.models.common import DocumentInput, DocumentType
from app.models.extraction import ConfidenceLevel, ExtractedContent, ExtractionInput
from app.models.trace import ClaimTrace, TraceStatus


def test_injection_mode_uses_content_as_ground_truth():
    doc = DocumentInput(
        file_id="F021",
        actual_type=DocumentType.PRESCRIPTION,
        content={"doctor_name": "Vaidya T. Krishnan", "diagnosis": "Chronic Joint Pain"},
    )
    trace = ClaimTrace(claim_ref="TEST")
    agent = ExtractionAgent()

    result = agent.run(
        ExtractionInput(document=doc, document_type=DocumentType.PRESCRIPTION, claim_category="ALTERNATIVE_MEDICINE"),
        trace,
    )

    assert result.extraction_status == "SUCCESS"
    assert result.source == "INJECTED"
    assert result.overall_confidence == ConfidenceLevel.HIGH
    assert result.content.doctor_name == "Vaidya T. Krishnan"
    assert any(e.status == TraceStatus.SUCCESS for e in trace.entries)


def test_simulate_failure_returns_failed_status_and_continues():
    doc = DocumentInput(
        file_id="F021",
        actual_type=DocumentType.PRESCRIPTION,
        content={"doctor_name": "Vaidya T. Krishnan"},
    )
    trace = ClaimTrace(claim_ref="TC011")
    agent = ExtractionAgent()

    result = agent.run(
        ExtractionInput(
            document=doc,
            document_type=DocumentType.PRESCRIPTION,
            claim_category="ALTERNATIVE_MEDICINE",
            simulate_failure=True,
        ),
        trace,
    )

    assert result.extraction_status == "FAILED"
    assert result.overall_confidence == ConfidenceLevel.LOW
    assert result.error is not None
    assert any(e.status == TraceStatus.FAILED for e in trace.entries)


def test_no_content_or_path_fails_gracefully():
    doc = DocumentInput(file_id="F099", actual_type=DocumentType.PRESCRIPTION)
    trace = ClaimTrace(claim_ref="TEST")
    agent = ExtractionAgent()

    result = agent.run(
        ExtractionInput(document=doc, document_type=DocumentType.PRESCRIPTION, claim_category="CONSULTATION"),
        trace,
    )

    assert result.extraction_status == "FAILED"
    assert result.error is not None


def test_live_mode_calls_gemini_and_scores_fields():
    doc = DocumentInput(file_id="F100", file_path="some/path.jpg", actual_type=DocumentType.PRESCRIPTION)
    trace = ClaimTrace(claim_ref="TEST")

    mock_gemini = MagicMock()
    mock_gemini.extract_content.return_value = ExtractedContent(
        doctor_name="Dr. Sharma", patient_name="Rajesh Kumar", diagnosis="Viral Fever", medicines=["Paracetamol"]
    )
    agent = ExtractionAgent(gemini_client=mock_gemini)

    result = agent.run(
        ExtractionInput(document=doc, document_type=DocumentType.PRESCRIPTION, claim_category="CONSULTATION"),
        trace,
    )

    assert result.extraction_status == "SUCCESS"
    assert result.source == "VISION_LLM"
    assert result.overall_confidence == ConfidenceLevel.HIGH
    assert all(fc.confidence == ConfidenceLevel.HIGH for fc in result.field_confidences)


def test_live_mode_gemini_failure_returns_failed():
    doc = DocumentInput(file_id="F101", file_path="some/path.jpg", actual_type=DocumentType.PHARMACY_BILL)
    trace = ClaimTrace(claim_ref="TEST")

    mock_gemini = MagicMock()
    mock_gemini.extract_content.side_effect = GeminiAPIError("rate limited")
    agent = ExtractionAgent(gemini_client=mock_gemini)

    result = agent.run(
        ExtractionInput(document=doc, document_type=DocumentType.PHARMACY_BILL, claim_category="PHARMACY"),
        trace,
    )

    assert result.extraction_status == "FAILED"
    assert result.overall_confidence == ConfidenceLevel.LOW
    assert "rate limited" in result.error
    assert any(e.status == TraceStatus.FAILED for e in trace.entries)
