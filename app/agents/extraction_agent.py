"""Stage 2: Extraction Agent.

Runs once per uploaded document. In injection mode (``document.content`` supplied by the eval
harness) the content is taken as ground truth. In live mode, Gemini vision extracts structured
content using ``ExtractedContent`` as the ``response_schema``.

``simulate_failure`` (set by the Orchestrator on one document when
``claim.simulate_component_failure`` is true) deterministically exercises the FAILED path without
needing a real Gemini outage -- the pipeline must continue with the remaining documents.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.exceptions import GeminiAPIError
from app.models.common import DocumentType
from app.models.extraction import (
    ConfidenceLevel,
    ExtractedContent,
    ExtractionInput,
    ExtractionResult,
    FieldConfidence,
)
from app.models.trace import ClaimTrace, TraceStage, TraceStatus

# Fields whose presence most strongly indicates a high-quality extraction, per document type.
KEY_FIELDS_BY_TYPE: dict[DocumentType, list[str]] = {
    DocumentType.PRESCRIPTION: ["doctor_name", "patient_name", "diagnosis", "medicines"],
    DocumentType.HOSPITAL_BILL: ["hospital_name", "patient_name", "line_items", "total"],
    DocumentType.PHARMACY_BILL: ["line_items", "total"],
    DocumentType.LAB_REPORT: ["lab_name", "tests_ordered", "test_results"],
    DocumentType.DIAGNOSTIC_REPORT: ["lab_name", "tests_ordered", "test_results"],
    DocumentType.DISCHARGE_SUMMARY: ["hospital_name", "patient_name", "diagnosis", "treatment"],
    DocumentType.DENTAL_REPORT: ["diagnosis", "treatment", "line_items"],
}


class ExtractionAgent(BaseAgent):
    def run(self, input: ExtractionInput, trace: ClaimTrace) -> ExtractionResult:
        doc = input.document

        if input.simulate_failure:
            trace.add(
                TraceStage.EXTRACTION,
                "extract_content",
                TraceStatus.FAILED,
                f"{doc.file_id}: extraction failed (simulated component failure).",
                detail={"file_id": doc.file_id, "simulated": True},
            )
            return ExtractionResult(
                file_id=doc.file_id,
                document_type=input.document_type,
                content=ExtractedContent(),
                field_confidences=[],
                overall_confidence=ConfidenceLevel.LOW,
                extraction_status="FAILED",
                source="INJECTED" if doc.content is not None else "VISION_LLM",
                error="Simulated component failure: extraction could not be completed for this document.",
            )

        if doc.content is not None:
            content = ExtractedContent.model_validate(doc.content)
            trace.add(
                TraceStage.EXTRACTION,
                "extract_content",
                TraceStatus.SUCCESS,
                f"{doc.file_id}: content provided directly (INJECTED), treated as ground truth.",
                detail={"file_id": doc.file_id, "document_type": input.document_type.value},
            )
            return ExtractionResult(
                file_id=doc.file_id,
                document_type=input.document_type,
                content=content,
                field_confidences=[],
                overall_confidence=ConfidenceLevel.HIGH,
                extraction_status="SUCCESS",
                source="INJECTED",
            )

        if not doc.file_path:
            trace.add(
                TraceStage.EXTRACTION,
                "extract_content",
                TraceStatus.FAILED,
                f"{doc.file_id}: no file content or path available to extract from.",
                detail={"file_id": doc.file_id},
            )
            return ExtractionResult(
                file_id=doc.file_id,
                document_type=input.document_type,
                content=ExtractedContent(),
                field_confidences=[],
                overall_confidence=ConfidenceLevel.LOW,
                extraction_status="FAILED",
                source="VISION_LLM",
                error="No document content or file path provided.",
            )

        try:
            content = self.gemini.extract_content(doc.file_path, input.document_type)
        except GeminiAPIError as exc:
            short = str(exc).split("\n")[0][:120]
            trace.add(
                TraceStage.EXTRACTION,
                "extract_content",
                TraceStatus.FAILED,
                f"{doc.file_id}: Gemini extraction failed.",
                detail={"file_id": doc.file_id, "error": short},
            )
            return ExtractionResult(
                file_id=doc.file_id,
                document_type=input.document_type,
                content=ExtractedContent(),
                field_confidences=[],
                overall_confidence=ConfidenceLevel.LOW,
                extraction_status="FAILED",
                source="VISION_LLM",
                error=short,
            )

        field_confidences, overall_confidence = self._score_fields(content, input.document_type)
        trace.add(
            TraceStage.EXTRACTION,
            "extract_content",
            TraceStatus.SUCCESS,
            f"{doc.file_id}: extracted via Gemini vision, overall confidence {overall_confidence.value}.",
            detail={"file_id": doc.file_id, "document_type": input.document_type.value},
        )
        return ExtractionResult(
            file_id=doc.file_id,
            document_type=input.document_type,
            content=content,
            field_confidences=field_confidences,
            overall_confidence=overall_confidence,
            extraction_status="SUCCESS",
            source="VISION_LLM",
        )

    def _score_fields(
        self, content: ExtractedContent, document_type: DocumentType
    ) -> tuple[list[FieldConfidence], ConfidenceLevel]:
        key_fields = KEY_FIELDS_BY_TYPE.get(document_type, [])
        if not key_fields:
            return [], ConfidenceLevel.MEDIUM

        dumped = content.model_dump()
        field_confidences: list[FieldConfidence] = []
        missing = 0
        for field_name in key_fields:
            value = dumped.get(field_name)
            if value in (None, [], ""):
                field_confidences.append(
                    FieldConfidence(field_name=field_name, confidence=ConfidenceLevel.LOW, reason="Not detected on document.")
                )
                missing += 1
            else:
                field_confidences.append(FieldConfidence(field_name=field_name, confidence=ConfidenceLevel.HIGH))

        if missing == 0:
            overall = ConfidenceLevel.HIGH
        elif missing < len(key_fields):
            overall = ConfidenceLevel.MEDIUM
        else:
            overall = ConfidenceLevel.LOW
        return field_confidences, overall
