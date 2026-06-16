"""Small Pydantic schemas describing the *shape* of structured responses we ask Gemini for.

These are deliberately narrower than the full pipeline contracts in ``app/models`` -- Gemini
should only ever fill in fields it can actually observe from an image or reason about from text,
never bookkeeping fields like ``file_id`` or ``source`` which the calling agent fills in itself.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.common import DocumentQuality, DocumentType


class GeminiDocumentClassification(BaseModel):
    document_type: DocumentType
    quality: DocumentQuality
    patient_name_on_doc: Optional[str] = Field(
        default=None,
        description="The patient's name as printed on the document, if visible.",
    )


class GeminiCanonicalMappingResponse(BaseModel):
    """LLM-assisted fallback for mapping free-text diagnosis/treatment onto policy terms."""

    waiting_period_key: Optional[str] = Field(
        default=None,
        description="One of the candidate waiting-period keys this diagnosis matches, or null.",
    )
    matched_exclusion_terms: list[str] = Field(
        default_factory=list,
        description="Any of the candidate exclusion terms this diagnosis/treatment matches.",
    )
    confidence: float = Field(description="0-1 confidence in this mapping.")
    rationale: str = Field(description="Brief explanation of why these terms were (or weren't) matched.")
