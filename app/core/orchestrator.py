"""The Orchestrator wires the five agents into a single pipeline and is the only place that
decides whether the pipeline stops early, degrades, or proceeds to a final decision.

Resilience contract: ``process_claim`` never raises. Every stage runs inside its own
``try/except``; on an unexpected exception the stage is recorded in ``DegradedContext`` (consumed
by ``app.core.confidence``) and the pipeline continues with a best-effort default for that stage.
The Decision Agent itself is the final fallback -- if even it raises, the Orchestrator returns a
synthetic ``MANUAL_REVIEW`` decision built directly here.
"""
from __future__ import annotations

from app.core.logging_config import get_logger
from app.agents.decision_agent import DecisionAgent
from app.agents.document_verification_agent import DocumentVerificationAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.fraud_detection_agent import FraudDetectionAgent
from app.agents.policy_evaluation_agent import PolicyEvaluationAgent
from app.core.exceptions import MemberNotFoundError, PolicyConfigError
from app.llm.gemini_client import GeminiClient
from app.models.common import ClaimHistoryEntry, ClaimInput, DocumentType
from app.models.decision import ClaimDecision, DecisionInput
from app.models.extraction import ConfidenceLevel, ExtractedContent, ExtractionInput, ExtractionResult
from app.models.fraud import FraudCheckInput, FraudCheckResult
from app.models.policy_eval import CanonicalMapping, PolicyEvaluationInput, PolicyEvaluationResult
from app.models.result import ClaimResult
from app.models.trace import ClaimTrace, DegradedContext, TraceStage, TraceStatus
from app.models.verification import VerificationInput, VerificationResult
from app.storage.claims_ledger import ClaimsLedger
from app.storage.policy_repository import PolicyRepository


_log = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        policy_repository: PolicyRepository | None = None,
        claims_ledger: ClaimsLedger | None = None,
        gemini_client: GeminiClient | None = None,
    ):
        self._policy_repo = policy_repository or PolicyRepository()
        self._ledger = claims_ledger or ClaimsLedger()
        self._verification_agent = DocumentVerificationAgent(gemini_client)
        self._extraction_agent = ExtractionAgent(gemini_client)
        self._policy_agent = PolicyEvaluationAgent(gemini_client)
        self._fraud_agent = FraudDetectionAgent(gemini_client)
        self._decision_agent = DecisionAgent(gemini_client)

    def process_claim(self, claim: ClaimInput, claim_ref: str | None = None, record_in_ledger: bool = True) -> ClaimResult:
        claim_ref = claim_ref or f"{claim.member_id}_{claim.treatment_date.isoformat()}_{claim.claim_category.value}"
        trace = ClaimTrace(claim_ref=claim_ref)
        degraded = DegradedContext()
        policy = self._policy_repo.policy

        _log.info(
            "[%s] Pipeline START — member_id=%s, category=%s, amount=%.2f, docs=%d",
            claim_ref, claim.member_id, claim.claim_category.value, claim.claimed_amount, len(claim.documents),
        )

        try:
            member = self._policy_repo.get_member(claim.member_id)
            _log.debug("[%s] Member lookup OK — %s", claim_ref, member.name)
        except MemberNotFoundError as exc:
            _log.warning("[%s] Member lookup FAILED — %s", claim_ref, exc)
            trace.add(TraceStage.ORCHESTRATOR, "member_lookup", TraceStatus.FAILED, str(exc))
            return ClaimResult(
                claim_ref=claim_ref, stopped_early=True, trace=trace,
                member_message=(
                    f"We could not find a policy member with ID '{claim.member_id}'. "
                    f"Please verify your member ID and resubmit this claim."
                ),
            )

        # Stage 1: Document Verification
        _log.info("[%s] Stage 1 START — DocumentVerificationAgent (%d doc(s))", claim_ref, len(claim.documents))
        try:
            verification = self._verification_agent.run(VerificationInput(claim=claim, policy=policy), trace)
            _log.info("[%s] Stage 1 END — passed=%s, issues=%d", claim_ref, verification.passed, len(verification.issues))
        except PolicyConfigError as exc:
            _log.error("[%s] Stage 1 FAILED — PolicyConfigError: %s", claim_ref, exc, exc_info=True)
            trace.add(TraceStage.ORCHESTRATOR, "verification", TraceStatus.FAILED, str(exc))
            return self._config_error_result(claim_ref, trace, exc)
        except Exception as exc:  # noqa: BLE001 - last-resort resilience boundary
            _log.error("[%s] Stage 1 FAILED — unexpected error: %s", claim_ref, exc, exc_info=True)
            degraded.mark_failed("VERIFICATION", f"Document verification failed unexpectedly: {exc}")
            verification = VerificationResult(passed=True, issues=[], classified_documents=[])

        if not verification.passed:
            blocking_messages = [issue.message for issue in verification.issues if issue.severity == "BLOCKING"]
            _log.info("[%s] Pipeline STOPPED EARLY — blocking verification issues: %s", claim_ref, blocking_messages)
            trace.add(
                TraceStage.ORCHESTRATOR, "stop_early", TraceStatus.INFO,
                "Stopping early: claim has blocking document verification issue(s).",
                detail={"messages": blocking_messages},
            )
            return ClaimResult(
                claim_ref=claim_ref, stopped_early=True, verification=verification, trace=trace,
                member_message=" ".join(blocking_messages),
            )

        # Stage 2: Extraction (per document)
        _log.info("[%s] Stage 2 START — ExtractionAgent (%d doc(s))", claim_ref, len(claim.documents))
        doc_type_by_id = {c.file_id: c.document_type for c in verification.classified_documents}
        simulate_failure_file_id = claim.documents[-1].file_id if (claim.simulate_component_failure and claim.documents) else None
        extractions: list[ExtractionResult] = []
        for doc in claim.documents:
            doc_type = doc_type_by_id.get(doc.file_id, doc.actual_type or DocumentType.UNKNOWN)
            _log.debug("[%s] Stage 2 extracting %s (%s)", claim_ref, doc.file_id, doc_type.value)
            try:
                extraction = self._extraction_agent.run(
                    ExtractionInput(
                        document=doc,
                        document_type=doc_type,
                        claim_category=claim.claim_category.value,
                        simulate_failure=(doc.file_id == simulate_failure_file_id),
                    ),
                    trace,
                )
                _log.debug("[%s] Stage 2 %s — status=%s, confidence=%s", claim_ref, doc.file_id, extraction.extraction_status, extraction.overall_confidence.value)
            except Exception as exc:  # noqa: BLE001
                _log.error("[%s] Stage 2 FAILED for %s — %s", claim_ref, doc.file_id, exc, exc_info=True)
                degraded.mark_failed("EXTRACTION", f"Extraction failed unexpectedly for {doc.file_id}: {exc}")
                extraction = ExtractionResult(
                    file_id=doc.file_id, document_type=doc_type, content=ExtractedContent(),
                    field_confidences=[], overall_confidence=ConfidenceLevel.LOW,
                    extraction_status="FAILED", source="VISION_LLM", error=str(exc),
                )
            extractions.append(extraction)
        _log.info("[%s] Stage 2 END — %d extraction(s) complete", claim_ref, len(extractions))

        # Stage 3: Policy Evaluation
        _log.info("[%s] Stage 3 START — PolicyEvaluationAgent", claim_ref)
        try:
            policy_eval = self._policy_agent.run(
                PolicyEvaluationInput(claim=claim, member=member, extractions=extractions, policy=policy, degraded_context=degraded), trace
            )
            _log.info("[%s] Stage 3 END — decision_hint=%s, approved_amount=%s", claim_ref, policy_eval.decision_hint, policy_eval.approved_amount)
        except PolicyConfigError as exc:
            _log.error("[%s] Stage 3 FAILED — PolicyConfigError: %s", claim_ref, exc, exc_info=True)
            trace.add(TraceStage.ORCHESTRATOR, "policy_evaluation", TraceStatus.FAILED, str(exc))
            return self._config_error_result(claim_ref, trace, exc, verification=verification, extractions=extractions)
        except Exception as exc:  # noqa: BLE001
            _log.error("[%s] Stage 3 FAILED — unexpected error: %s", claim_ref, exc, exc_info=True)
            degraded.mark_failed("POLICY_EVALUATION", f"Policy evaluation failed unexpectedly: {exc}")
            policy_eval = PolicyEvaluationResult(decision_hint="MANUAL_REVIEW", canonical_mapping=CanonicalMapping())

        # Stage 4: Fraud Detection
        _log.info("[%s] Stage 4 START — FraudDetectionAgent", claim_ref)
        try:
            history = self._ledger.merged_history(claim.member_id, claim.claims_history)
            claim_for_fraud = claim.model_copy(update={"claims_history": history})
            fraud = self._fraud_agent.run(FraudCheckInput(claim=claim_for_fraud, member=member, policy_eval=policy_eval, policy=policy), trace)
            _log.info("[%s] Stage 4 END — fraud_score=%.2f, signals=%d, manual_review=%s", claim_ref, fraud.fraud_score, len(fraud.signals), fraud.requires_manual_review)
        except Exception as exc:  # noqa: BLE001
            _log.error("[%s] Stage 4 FAILED — %s", claim_ref, exc, exc_info=True)
            degraded.mark_failed("FRAUD_DETECTION", f"Fraud detection failed unexpectedly: {exc}")
            fraud = FraudCheckResult(fraud_score=1.0, signals=[], requires_manual_review=True)

        # Stage 5: Decision
        _log.info("[%s] Stage 5 START — DecisionAgent", claim_ref)
        try:
            decision = self._decision_agent.run(
                DecisionInput(claim=claim, verification=verification, extractions=extractions, policy_eval=policy_eval, fraud=fraud, degraded_context=degraded),
                trace,
            )
            _log.info(
                "[%s] Stage 5 END — decision=%s, approved_amount=%.2f, confidence=%.2f, manual_review=%s",
                claim_ref, decision.decision, decision.approved_amount, decision.confidence_score, decision.manual_review_recommended,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("[%s] Stage 5 FAILED — %s", claim_ref, exc, exc_info=True)
            trace.add(TraceStage.ORCHESTRATOR, "decision", TraceStatus.FAILED, f"Decision agent failed: {exc}")
            decision = ClaimDecision(
                decision="MANUAL_REVIEW", approved_amount=0, confidence_score=0.0,
                notes=[f"Decision agent failed: {exc}"], manual_review_recommended=True,
                manual_review_reasons=[f"Decision agent failed: {exc}"],
            )

        if record_in_ledger:
            self._ledger.append(
                claim.member_id,
                ClaimHistoryEntry(claim_id=claim_ref, date=claim.treatment_date, amount=claim.claimed_amount, provider=claim.hospital_name),
            )

        _log.info(
            "[%s] Pipeline END — decision=%s, amount=%.2f, confidence=%.2f",
            claim_ref, decision.decision, decision.approved_amount, decision.confidence_score,
        )
        return ClaimResult(
            claim_ref=claim_ref, stopped_early=False, verification=verification, extractions=extractions,
            policy_evaluation=policy_eval, fraud_check=fraud, decision=decision, trace=trace,
        )

    def _config_error_result(
        self,
        claim_ref: str,
        trace: ClaimTrace,
        exc: Exception,
        verification: VerificationResult | None = None,
        extractions: list[ExtractionResult] | None = None,
    ) -> ClaimResult:
        decision = ClaimDecision(
            decision="MANUAL_REVIEW", approved_amount=0, confidence_score=0.0,
            notes=[f"This claim could not be evaluated automatically: {exc}"],
            manual_review_recommended=True,
            manual_review_reasons=[f"Policy configuration error: {exc}"],
        )
        return ClaimResult(
            claim_ref=claim_ref, stopped_early=False, verification=verification, extractions=extractions,
            decision=decision, trace=trace,
        )
