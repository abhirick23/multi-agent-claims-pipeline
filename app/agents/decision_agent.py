"""Stage 5: Decision Agent -- the final synthesis step.

Combines the Policy Evaluation hint, the Fraud Detection result, and a confidence score
(``app.core.confidence``) into the single ``ClaimDecision`` shown to the user:

1. A fraud signal that requires manual review overrides the policy hint -> ``MANUAL_REVIEW``.
2. Otherwise the policy ``decision_hint`` is used as-is.
3. Confidence below ``CONFIDENCE_FORCE_MANUAL_REVIEW`` overrides *any* decision to
   ``MANUAL_REVIEW`` (a hard floor -- too much of the pipeline was degraded to trust the result).
4. Confidence below ``CONFIDENCE_ADVISORY_REVIEW`` (but above the floor) keeps the decision but
   sets ``manual_review_recommended=True`` (TC011: one extraction failed, claim still APPROVED,
   but flagged for a human to double-check).
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.confidence import CONFIDENCE_ADVISORY_REVIEW, CONFIDENCE_FORCE_MANUAL_REVIEW, compute_confidence
from app.models.decision import ClaimDecision, DecisionInput, RejectionReason
from app.models.policy_eval import CheckStatus, PolicyEvaluationResult
from app.models.trace import ClaimTrace, TraceStage, TraceStatus


class DecisionAgent(BaseAgent):
    def run(self, input: DecisionInput, trace: ClaimTrace) -> ClaimDecision:
        policy_eval, fraud = input.policy_eval, input.fraud

        confidence, notes = compute_confidence(input.extractions, policy_eval, input.degraded_context)

        manual_review_recommended = False
        manual_review_reasons: list[str] = []

        if fraud.requires_manual_review:
            decision = "MANUAL_REVIEW"
            for signal in fraud.signals:
                notes.append(signal.message)
                manual_review_reasons.append(signal.message)
        else:
            decision = policy_eval.decision_hint

        if confidence < CONFIDENCE_FORCE_MANUAL_REVIEW:
            decision = "MANUAL_REVIEW"
            manual_review_reasons.append(
                f"Confidence score ({confidence:.2f}) is below the minimum threshold of "
                f"{CONFIDENCE_FORCE_MANUAL_REVIEW:.2f} required to act on this result automatically."
            )
        elif confidence < CONFIDENCE_ADVISORY_REVIEW and decision != "MANUAL_REVIEW":
            manual_review_recommended = True
            manual_review_reasons.append(
                f"Confidence score ({confidence:.2f}) is below {CONFIDENCE_ADVISORY_REVIEW:.2f} due to "
                f"incomplete processing; manual review is recommended before finalizing this decision."
            )

        approved_amount = policy_eval.approved_amount or 0 if decision in ("APPROVED", "PARTIAL") else 0
        financial_breakdown = policy_eval.financial_breakdown if decision in ("APPROVED", "PARTIAL") else None
        line_item_breakdown = policy_eval.line_item_results

        rejection_reasons = self._build_rejection_reasons(policy_eval)

        trace.add(
            TraceStage.DECISION, "final_decision", TraceStatus.SUCCESS,
            f"Final decision: {decision} (confidence {confidence:.2f}, approved {approved_amount}).",
            detail={
                "decision": decision,
                "approved_amount": approved_amount,
                "confidence_score": confidence,
                "manual_review_recommended": manual_review_recommended,
            },
        )

        return ClaimDecision(
            decision=decision,
            approved_amount=approved_amount,
            rejection_reasons=rejection_reasons,
            confidence_score=confidence,
            notes=notes,
            line_item_breakdown=line_item_breakdown,
            financial_breakdown=financial_breakdown,
            manual_review_recommended=manual_review_recommended,
            manual_review_reasons=manual_review_reasons,
        )

    def _build_rejection_reasons(self, policy_eval: PolicyEvaluationResult) -> list[RejectionReason]:
        reasons: list[RejectionReason] = []
        for code in policy_eval.rejection_reasons:
            check = next((c for c in policy_eval.checks if c.code == code and c.status == CheckStatus.FAIL), None)
            if check:
                reasons.append(RejectionReason(code=check.code, message=check.message, policy_reference=check.policy_reference))
            else:
                reasons.append(RejectionReason(code=code, message=code))
        return reasons
