"""Contracts for the Document Verification Agent (Stage 1)."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.models.common import ClaimInput, DocumentQuality, DocumentType
from app.models.policy import PolicyTerms


class VerificationIssueCode(str, Enum):
    WRONG_DOCUMENT_TYPE = "WRONG_DOCUMENT_TYPE"
    MISSING_REQUIRED_DOCUMENT = "MISSING_REQUIRED_DOCUMENT"
    UNREADABLE_DOCUMENT = "UNREADABLE_DOCUMENT"
    PATIENT_NAME_MISMATCH = "PATIENT_NAME_MISMATCH"


class VerificationIssue(BaseModel):
    code: VerificationIssueCode
    severity: Literal["BLOCKING", "WARNING"]
    file_id: Optional[str] = None
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class DocumentClassification(BaseModel):
    file_id: str
    document_type: DocumentType
    quality: DocumentQuality
    patient_name_on_doc: Optional[str] = None
    source: Literal["INJECTED", "VISION_LLM"]


class VerificationInput(BaseModel):
    claim: ClaimInput
    policy: PolicyTerms

    model_config = {"arbitrary_types_allowed": True}


class VerificationResult(BaseModel):
    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    classified_documents: list[DocumentClassification] = Field(default_factory=list)
