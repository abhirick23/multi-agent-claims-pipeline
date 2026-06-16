"""Shared enums and the top-level ClaimInput contract."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DocumentType(str, Enum):
    PRESCRIPTION = "PRESCRIPTION"
    HOSPITAL_BILL = "HOSPITAL_BILL"
    LAB_REPORT = "LAB_REPORT"
    PHARMACY_BILL = "PHARMACY_BILL"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    DENTAL_REPORT = "DENTAL_REPORT"
    DIAGNOSTIC_REPORT = "DIAGNOSTIC_REPORT"
    UNKNOWN = "UNKNOWN"


class DocumentQuality(str, Enum):
    GOOD = "GOOD"
    POOR = "POOR"
    UNREADABLE = "UNREADABLE"


class ClaimCategory(str, Enum):
    CONSULTATION = "CONSULTATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    PHARMACY = "PHARMACY"
    DENTAL = "DENTAL"
    VISION = "VISION"
    ALTERNATIVE_MEDICINE = "ALTERNATIVE_MEDICINE"


class DocumentInput(BaseModel):
    """A single uploaded document.

    In "live" mode, ``file_path`` points at an image/PDF and the Document Verification /
    Extraction agents call Gemini vision to classify and extract it.

    In "injection" mode (used by the eval harness against test_cases.json), ``actual_type``,
    ``quality`` and ``patient_name_on_doc`` let the Document Verification Agent skip the vision
    classification call, and ``content`` lets the Extraction Agent skip the vision extraction
    call -- both agents simply check "was this given to me already?" first.
    """

    file_id: str
    file_name: Optional[str] = None
    file_path: Optional[str] = None

    # injection-mode fields (test harness / pre-classified data)
    actual_type: Optional[DocumentType] = None
    quality: Optional[DocumentQuality] = None
    patient_name_on_doc: Optional[str] = None
    content: Optional[dict] = None


class ClaimHistoryEntry(BaseModel):
    claim_id: str
    date: date
    amount: float
    provider: Optional[str] = None


class ClaimInput(BaseModel):
    member_id: str
    policy_id: str
    claim_category: ClaimCategory
    treatment_date: date
    submission_date: Optional[date] = None
    claimed_amount: float
    hospital_name: Optional[str] = None
    pre_auth_obtained: bool = False
    ytd_claims_amount: float = 0
    claims_history: list[ClaimHistoryEntry] = Field(default_factory=list)
    simulate_component_failure: bool = False
    documents: list[DocumentInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_submission_date(self) -> "ClaimInput":
        # Assumption (documented in ARCHITECTURE.md): without an explicit submission_date we
        # treat the claim as submitted on the treatment date, so the submission-deadline check
        # is exercised but doesn't spuriously fail historical test data evaluated long after
        # the fact.
        if self.submission_date is None:
            self.submission_date = self.treatment_date
        return self
