"""Stage 1: Document Verification Agent.

Confirms the right documents were uploaded, are legible, and belong to the same patient -- *before*
any policy logic runs. Any BLOCKING issue here stops the pipeline early with a specific,
member-facing message (the orchestrator never reaches a claim decision in that case).
"""
from __future__ import annotations

from collections import Counter

from app.agents.base import BaseAgent
from app.core.exceptions import GeminiAPIError, PolicyConfigError
from app.models.common import DocumentQuality, DocumentType
from app.models.trace import ClaimTrace, TraceStage, TraceStatus
from app.models.verification import (
    DocumentClassification,
    VerificationIssue,
    VerificationIssueCode,
    VerificationInput,
    VerificationResult,
)


class DocumentVerificationAgent(BaseAgent):
    def run(self, input: VerificationInput, trace: ClaimTrace) -> VerificationResult:
        claim = input.claim
        policy = input.policy

        classified = [self._classify(doc, trace) for doc in claim.documents]

        issues: list[VerificationIssue] = []
        issues.extend(self._check_quality(classified, claim.documents, trace))
        issues.extend(self._check_document_requirements(classified, claim.documents, claim.claim_category.value, policy, trace))
        issues.extend(self._check_patient_name_consistency(classified, claim.documents, trace))

        passed = not any(issue.severity == "BLOCKING" for issue in issues)
        trace.add(
            TraceStage.VERIFICATION,
            "result",
            TraceStatus.SUCCESS if passed else TraceStatus.FAILED,
            f"Document verification {'passed' if passed else 'failed'} "
            f"with {len(issues)} issue(s).",
            detail={"issue_count": len(issues), "passed": passed},
        )
        return VerificationResult(passed=passed, issues=issues, classified_documents=classified)

    def _classify(self, doc, trace: ClaimTrace) -> DocumentClassification:
        if doc.actual_type is not None:
            classification = DocumentClassification(
                file_id=doc.file_id,
                document_type=doc.actual_type,
                quality=doc.quality or DocumentQuality.GOOD,
                patient_name_on_doc=doc.patient_name_on_doc,
                source="INJECTED",
            )
            trace.add(
                TraceStage.VERIFICATION,
                "classify_document",
                TraceStatus.SUCCESS,
                f"{doc.file_id}: classified as {classification.document_type.value} "
                f"({classification.quality.value}), source=INJECTED.",
                detail=classification.model_dump(mode="json"),
            )
            return classification

        if doc.file_path:
            try:
                result = self.gemini.classify_document(doc.file_path)
                classification = DocumentClassification(
                    file_id=doc.file_id,
                    document_type=result.document_type,
                    quality=result.quality,
                    patient_name_on_doc=result.patient_name_on_doc,
                    source="VISION_LLM",
                )
                trace.add(
                    TraceStage.VERIFICATION,
                    "classify_document",
                    TraceStatus.SUCCESS,
                    f"{doc.file_id}: classified as {classification.document_type.value} "
                    f"({classification.quality.value}), source=VISION_LLM.",
                    detail=classification.model_dump(mode="json"),
                )
                return classification
            except GeminiAPIError as exc:
                classification = DocumentClassification(
                    file_id=doc.file_id,
                    document_type=DocumentType.UNKNOWN,
                    quality=DocumentQuality.POOR,
                    patient_name_on_doc=None,
                    source="VISION_LLM",
                )
                trace.add(
                    TraceStage.VERIFICATION,
                    "classify_document",
                    TraceStatus.DEGRADED,
                    f"{doc.file_id}: Gemini classification failed ({exc}); "
                    f"treating as UNKNOWN/POOR.",
                    detail={"error": str(exc)},
                )
                return classification

        classification = DocumentClassification(
            file_id=doc.file_id,
            document_type=DocumentType.UNKNOWN,
            quality=DocumentQuality.POOR,
            patient_name_on_doc=None,
            source="INJECTED",
        )
        trace.add(
            TraceStage.VERIFICATION,
            "classify_document",
            TraceStatus.DEGRADED,
            f"{doc.file_id}: no file content or injected classification provided; "
            f"treating as UNKNOWN/POOR.",
            detail={},
        )
        return classification

    def _check_quality(self, classified, documents, trace: ClaimTrace) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        names_by_id = {d.file_id: (d.file_name or d.file_id) for d in documents}
        for c in classified:
            if c.quality == DocumentQuality.UNREADABLE:
                message = (
                    f"The document '{names_by_id[c.file_id]}' (classified as "
                    f"{c.document_type.value}) is too unclear to read. Please re-upload a "
                    f"clearer photo or scan of this specific document."
                )
                issues.append(
                    VerificationIssue(
                        code=VerificationIssueCode.UNREADABLE_DOCUMENT,
                        severity="BLOCKING",
                        file_id=c.file_id,
                        message=message,
                        detail={"document_type": c.document_type.value, "quality": c.quality.value},
                    )
                )
        trace.add(
            TraceStage.VERIFICATION,
            "check_quality",
            TraceStatus.FAILED if issues else TraceStatus.SUCCESS,
            f"Found {len(issues)} unreadable document(s)."
            if issues
            else "All documents are legible.",
            detail={"unreadable_file_ids": [i.file_id for i in issues]},
        )
        return issues

    def _check_document_requirements(
        self, classified, documents, claim_category: str, policy, trace: ClaimTrace
    ) -> list[VerificationIssue]:
        if claim_category not in policy.document_requirements:
            raise PolicyConfigError(
                f"No document_requirements configured for claim category '{claim_category}'."
            )
        requirements = policy.document_requirements[claim_category]
        required = set(requirements.required)
        optional = set(requirements.optional)

        uploaded_types = [c.document_type.value for c in classified if c.document_type != DocumentType.UNKNOWN]
        uploaded_counter = Counter(uploaded_types)
        missing = [t for t in requirements.required if uploaded_counter.get(t, 0) == 0]

        issues: list[VerificationIssue] = []
        if missing:
            extra_types = [
                t for t, count in uploaded_counter.items() if count > 1 or t not in (required | optional)
            ]
            uploaded_summary = ", ".join(f"{count}x {t}" for t, count in uploaded_counter.items()) or "no recognizable documents"
            missing_summary = ", ".join(missing)

            if extra_types:
                message = (
                    f"This {claim_category} claim requires: {', '.join(requirements.required)}. "
                    f"You uploaded: {uploaded_summary}, but no {missing_summary} was found. "
                    f"Please upload the missing {missing_summary} document."
                )
                code = VerificationIssueCode.WRONG_DOCUMENT_TYPE
            else:
                message = (
                    f"This {claim_category} claim is missing required document(s): {missing_summary}. "
                    f"You uploaded: {uploaded_summary}. Please upload the missing document(s)."
                )
                code = VerificationIssueCode.MISSING_REQUIRED_DOCUMENT

            issues.append(
                VerificationIssue(
                    code=code,
                    severity="BLOCKING",
                    file_id=None,
                    message=message,
                    detail={
                        "required_types": requirements.required,
                        "uploaded_types": uploaded_types,
                        "missing_types": missing,
                    },
                )
            )

        trace.add(
            TraceStage.VERIFICATION,
            "check_document_requirements",
            TraceStatus.FAILED if issues else TraceStatus.SUCCESS,
            f"Required documents for {claim_category}: {', '.join(requirements.required)}. "
            + ("Missing: " + ", ".join(missing) if missing else "All present."),
            detail={
                "required_types": requirements.required,
                "uploaded_types": uploaded_types,
            },
        )
        return issues

    def _check_patient_name_consistency(self, classified, documents, trace: ClaimTrace) -> list[VerificationIssue]:
        names = {}
        for c in classified:
            if c.patient_name_on_doc:
                normalized = c.patient_name_on_doc.strip().lower()
                names.setdefault(normalized, (c.patient_name_on_doc.strip(), []))[1].append(c.file_id)

        issues: list[VerificationIssue] = []
        if len(names) > 1:
            names_summary = ", ".join(f"'{display}' (document {', '.join(ids)})" for display, ids in names.values())
            issues.append(
                VerificationIssue(
                    code=VerificationIssueCode.PATIENT_NAME_MISMATCH,
                    severity="BLOCKING",
                    file_id=None,
                    message=(
                        f"The uploaded documents appear to belong to different patients: "
                        f"{names_summary}. Please confirm all documents are for the same patient "
                        f"and re-upload if needed."
                    ),
                    detail={"names_found": {display: ids for display, ids in names.values()}},
                )
            )

        trace.add(
            TraceStage.VERIFICATION,
            "check_patient_name_consistency",
            TraceStatus.FAILED if issues else TraceStatus.SUCCESS,
            f"Found {len(names)} distinct patient name(s) across documents."
            if names
            else "No patient names available to cross-check.",
            detail={"distinct_names": [display for display, _ in names.values()]},
        )
        return issues
