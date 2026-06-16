"""Contracts for the Extraction Agent (Stage 2)."""
from __future__ import annotations

from datetime import date as date_type
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.common import DocumentInput, DocumentType


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FieldConfidence(BaseModel):
    field_name: str
    confidence: ConfidenceLevel
    reason: Optional[str] = None


class LineItem(BaseModel):
    description: str
    amount: float


class ExtractedContent(BaseModel):
    """Superset of fields across all document types. Which fields are populated depends on
    ``document_type`` -- this is intentionally a single flat schema so the Extraction Agent has
    one ``response_schema`` to ask Gemini for, regardless of document type."""

    doctor_name: Optional[str] = None
    doctor_registration: Optional[str] = None
    patient_name: Optional[str] = None
    date: Optional[date_type] = None
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    medicines: list[str] = Field(default_factory=list)
    tests_ordered: list[str] = Field(default_factory=list)
    hospital_name: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    total: Optional[float] = None
    lab_name: Optional[str] = None
    test_results: list[dict] = Field(default_factory=list)


class ExtractionInput(BaseModel):
    document: DocumentInput
    document_type: DocumentType
    claim_category: str
    simulate_failure: bool = False


class ExtractionResult(BaseModel):
    file_id: str
    document_type: DocumentType
    content: ExtractedContent
    field_confidences: list[FieldConfidence] = Field(default_factory=list)
    overall_confidence: ConfidenceLevel
    extraction_status: Literal["SUCCESS", "PARTIAL", "FAILED"]
    source: Literal["INJECTED", "VISION_LLM"]
    error: Optional[str] = None
