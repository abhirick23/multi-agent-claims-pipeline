"""Confidence scoring: a single 0-1 number summarizing how much the pipeline had to
compromise (failed/degraded stages, low-confidence extractions, soft policy warnings, LLM-
assisted canonical mapping) before reaching a decision_hint.

These weights are tunable constants -- not given numerically by ``policy_terms.json`` -- chosen
so that a single FAILED extraction (TC011) lands between the two thresholds below (decision
kept, but flagged for advisory manual review), while a clean injected/HIGH-confidence run
(TC004, TC012) stays at 1.0.
"""
from __future__ import annotations

from app.models.extraction import ConfidenceLevel, ExtractionResult
from app.models.policy_eval import CheckStatus, PolicyEvaluationResult
from app.models.trace import DegradedContext

BASE_CONFIDENCE = 1.0

EXTRACTION_FAILED_PENALTY = 0.25
EXTRACTION_PARTIAL_PENALTY = 0.10
EXTRACTION_OVERALL_LOW_PENALTY = 0.10
EXTRACTION_OVERALL_MEDIUM_PENALTY = 0.05
LOW_FIELD_PENALTY_PER_FIELD = 0.05
LOW_FIELD_PENALTY_CAP = 0.10

DEGRADED_STAGE_PENALTY = 0.15
FAILED_STAGE_PENALTY = 0.30

WARN_CHECK_PENALTY = 0.05
LLM_ASSISTED_MAPPING_PENALTY = 0.05
LOW_CONFIDENCE_EXCLUSION_THRESHOLD = 0.75
LOW_CONFIDENCE_EXCLUSION_PENALTY = 0.05

# Decision Agent thresholds.
CONFIDENCE_FORCE_MANUAL_REVIEW = 0.45  # below this, the decision itself is overridden to MANUAL_REVIEW
CONFIDENCE_ADVISORY_REVIEW = 0.75  # below this (but above the floor), keep the decision but recommend review


def compute_confidence(
    extractions: list[ExtractionResult],
    policy_eval: PolicyEvaluationResult,
    degraded_context: DegradedContext,
) -> tuple[float, list[str]]:
    """Returns ``(confidence_score, notes)`` -- notes explain each deduction in plain language
    so they can be surfaced directly in ``ClaimDecision.notes``."""
    score = BASE_CONFIDENCE
    notes: list[str] = []

    for extraction in extractions:
        if extraction.extraction_status == "FAILED":
            score -= EXTRACTION_FAILED_PENALTY
            notes.append(f"Document extraction failed for {extraction.file_id} and was skipped: {extraction.error}")
        elif extraction.extraction_status == "PARTIAL":
            score -= EXTRACTION_PARTIAL_PENALTY
            notes.append(f"Document extraction for {extraction.file_id} was only partially successful.")

        if extraction.overall_confidence == ConfidenceLevel.LOW:
            score -= EXTRACTION_OVERALL_LOW_PENALTY
        elif extraction.overall_confidence == ConfidenceLevel.MEDIUM:
            score -= EXTRACTION_OVERALL_MEDIUM_PENALTY

        num_low_fields = sum(1 for fc in extraction.field_confidences if fc.confidence == ConfidenceLevel.LOW)
        if num_low_fields:
            score -= min(LOW_FIELD_PENALTY_PER_FIELD * num_low_fields, LOW_FIELD_PENALTY_CAP)

    for stage in degraded_context.degraded_stages:
        score -= DEGRADED_STAGE_PENALTY
    for stage in degraded_context.failed_stages:
        score -= FAILED_STAGE_PENALTY
    notes.extend(degraded_context.notes)

    for check in policy_eval.checks:
        if check.status == CheckStatus.WARN:
            score -= WARN_CHECK_PENALTY

    canonical = policy_eval.canonical_mapping
    if canonical.method == "LLM_ASSISTED":
        score -= LLM_ASSISTED_MAPPING_PENALTY
    if any(m.confidence < LOW_CONFIDENCE_EXCLUSION_THRESHOLD for m in canonical.exclusion_matches):
        score -= LOW_CONFIDENCE_EXCLUSION_PENALTY

    return max(0.0, min(1.0, round(score, 2))), notes
