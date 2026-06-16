"""The explainability backbone: every agent appends TraceEntry records describing what it
checked, what it found, and why. The ops UI renders ClaimTrace.entries as a timeline."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TraceStage(str, Enum):
    VERIFICATION = "VERIFICATION"
    EXTRACTION = "EXTRACTION"
    POLICY_EVALUATION = "POLICY_EVALUATION"
    FRAUD_DETECTION = "FRAUD_DETECTION"
    DECISION = "DECISION"
    ORCHESTRATOR = "ORCHESTRATOR"


class TraceStatus(str, Enum):
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    INFO = "INFO"


class TraceEntry(BaseModel):
    stage: TraceStage
    step: str
    status: TraceStatus
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ClaimTrace(BaseModel):
    claim_ref: str
    entries: list[TraceEntry] = Field(default_factory=list)

    def add(
        self,
        stage: TraceStage,
        step: str,
        status: TraceStatus,
        summary: str,
        detail: dict[str, Any] | None = None,
    ) -> TraceEntry:
        entry = TraceEntry(stage=stage, step=step, status=status, summary=summary, detail=detail or {})
        self.entries.append(entry)
        return entry


class DegradedContext(BaseModel):
    """Accumulated record of which pipeline stages failed or degraded, consumed by the
    confidence-scoring step in the Decision Agent."""

    failed_stages: list[str] = Field(default_factory=list)
    degraded_stages: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def mark_failed(self, stage: str, note: str) -> None:
        self.failed_stages.append(stage)
        self.notes.append(note)

    def mark_degraded(self, stage: str, note: str) -> None:
        self.degraded_stages.append(stage)
        self.notes.append(note)
