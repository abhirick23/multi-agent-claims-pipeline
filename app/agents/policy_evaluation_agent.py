"""Stage 3: Policy Evaluation Agent.

Applies ``policy_terms.json`` rules deterministically, in order, to produce a ``decision_hint``
plus a full financial breakdown. Each hard rule that FAILs short-circuits and returns
``REJECTED`` immediately with a specific, member-facing message; soft checks (sub-limits, annual
caps) are recorded as WARN and reduce confidence later without blocking the claim.
"""
from __future__ import annotations

from datetime import date, timedelta

from app.agents.base import BaseAgent
from app.agents.canonical_mapping import CanonicalMapper, classify_dental_procedures
from app.core.exceptions import PolicyConfigError
from app.models.common import ClaimCategory, ClaimInput
from app.models.extraction import ExtractionResult, LineItem
from app.models.policy import MemberRecord, OPDCategoryRules, PolicyTerms
from app.models.policy_eval import (
    CanonicalMapping,
    CheckStatus,
    FinancialBreakdown,
    LineItemResult,
    PolicyCheckResult,
    PolicyEvaluationInput,
    PolicyEvaluationResult,
)
from app.models.trace import ClaimTrace, TraceStage, TraceStatus


def _fmt(amount: float) -> str:
    return f"₹{amount:,.2f}"


def _effective_join_date(member: MemberRecord, policy: PolicyTerms) -> date:
    if member.join_date:
        return date.fromisoformat(member.join_date)
    if member.primary_member_id:
        for m in policy.members:
            if m.member_id == member.primary_member_id and m.join_date:
                return date.fromisoformat(m.join_date)
    raise PolicyConfigError(f"Member '{member.member_id}' has no resolvable join_date.")


class PolicyEvaluationAgent(BaseAgent):
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client)
        self._mapper = CanonicalMapper(gemini_client)

    def run(self, input: PolicyEvaluationInput, trace: ClaimTrace) -> PolicyEvaluationResult:
        claim, policy, member = input.claim, input.policy, input.member
        checks: list[PolicyCheckResult] = []

        category_key = claim.claim_category.value.lower()
        if category_key not in policy.opd_categories:
            raise PolicyConfigError(f"No opd_categories configuration found for claim category '{category_key}'.")
        category_rules = policy.opd_categories[category_key]

        diagnosis_text, treatment_text = self._gather_diagnosis_treatment(input.extractions)
        canonical = self._mapper.map(diagnosis_text, treatment_text, policy, trace)
        canonical.tests_ordered = self._gather_tests_ordered(input.extractions)

        # Step 2: minimum claim amount
        if claim.claimed_amount < policy.submission_rules.minimum_claim_amount:
            message = (
                f"The claimed amount ({_fmt(claim.claimed_amount)}) is below the minimum "
                f"claimable amount of {_fmt(policy.submission_rules.minimum_claim_amount)}."
            )
            checks.append(PolicyCheckResult(code="MIN_CLAIM_AMOUNT", status=CheckStatus.FAIL, message=message, policy_reference="submission_rules.minimum_claim_amount"))
            return self._rejected(checks, ["MIN_CLAIM_AMOUNT"], canonical, message, trace)

        # Step 3: submission deadline
        days_elapsed = (claim.submission_date - claim.treatment_date).days
        if days_elapsed > policy.submission_rules.deadline_days_from_treatment:
            message = (
                f"This claim was submitted {days_elapsed} day(s) after the treatment date, "
                f"exceeding the {policy.submission_rules.deadline_days_from_treatment}-day submission deadline."
            )
            checks.append(PolicyCheckResult(code="SUBMISSION_DEADLINE_EXCEEDED", status=CheckStatus.FAIL, message=message, policy_reference="submission_rules.deadline_days_from_treatment"))
            return self._rejected(checks, ["SUBMISSION_DEADLINE_EXCEEDED"], canonical, message, trace)
        checks.append(PolicyCheckResult(code="SUBMISSION_DEADLINE", status=CheckStatus.PASS, message=f"Submitted {days_elapsed} day(s) after treatment (within {policy.submission_rules.deadline_days_from_treatment}-day deadline)."))

        # Step 4: waiting periods
        join_date = _effective_join_date(member, policy)
        initial_eligible = join_date + timedelta(days=policy.waiting_periods.initial_waiting_period_days)
        if claim.treatment_date < initial_eligible:
            message = (
                f"Your policy has an initial waiting period of {policy.waiting_periods.initial_waiting_period_days} "
                f"days from your join date ({join_date.isoformat()}). You will be eligible for claims from "
                f"{initial_eligible.isoformat()} onwards."
            )
            checks.append(PolicyCheckResult(code="WAITING_PERIOD", status=CheckStatus.FAIL, message=message, policy_reference="waiting_periods.initial_waiting_period_days"))
            return self._rejected(checks, ["WAITING_PERIOD"], canonical, message, trace)

        if canonical.waiting_period_key:
            days = policy.waiting_periods.specific_conditions[canonical.waiting_period_key]
            eligible_date = join_date + timedelta(days=days)
            if claim.treatment_date < eligible_date:
                condition_label = canonical.waiting_period_key.replace("_", " ")
                message = (
                    f"Claims related to {condition_label} are subject to a {days}-day waiting period "
                    f"from your policy join date ({join_date.isoformat()}). You will be eligible for "
                    f"{condition_label}-related claims from {eligible_date.isoformat()} onwards. This "
                    f"claim's treatment date ({claim.treatment_date.isoformat()}) is before that date."
                )
                checks.append(PolicyCheckResult(code="WAITING_PERIOD", status=CheckStatus.FAIL, message=message, policy_reference=f"waiting_periods.specific_conditions.{canonical.waiting_period_key}"))
                return self._rejected(checks, ["WAITING_PERIOD"], canonical, message, trace)
        checks.append(PolicyCheckResult(code="WAITING_PERIOD", status=CheckStatus.PASS, message="No applicable waiting period restricts this claim."))

        # Step 5: exclusions (whole-claim)
        whole_claim_exclusions = [m for m in canonical.exclusion_matches if m.scope == "WHOLE_CLAIM"]
        if whole_claim_exclusions:
            terms = ", ".join(f"'{m.policy_term}'" for m in whole_claim_exclusions)
            basis = "; ".join(t for t in (diagnosis_text, treatment_text) if t) or "the submitted documents"
            message = f"This claim is not covered: it falls under the policy exclusion(s) {terms}, based on: {basis}."
            checks.append(PolicyCheckResult(code="EXCLUDED_CONDITION", status=CheckStatus.FAIL, message=message, policy_reference="exclusions.conditions"))
            return self._rejected(checks, ["EXCLUDED_CONDITION"], canonical, message, trace)
        checks.append(PolicyCheckResult(code="EXCLUDED_CONDITION", status=CheckStatus.PASS, message="No policy exclusions apply to this claim."))

        # Step 6: pre-authorization
        pre_auth_tests = category_rules.high_value_tests_requiring_pre_auth
        if pre_auth_tests and category_rules.pre_auth_threshold is not None:
            matched_tests = [
                t for t in canonical.tests_ordered if any(kw.lower() in t.lower() for kw in pre_auth_tests)
            ]
            if matched_tests and claim.claimed_amount > category_rules.pre_auth_threshold and not claim.pre_auth_obtained:
                message = (
                    f"Pre-authorization is required for {', '.join(matched_tests)} when the claimed amount "
                    f"exceeds {_fmt(category_rules.pre_auth_threshold)} (this claim is {_fmt(claim.claimed_amount)}), "
                    f"but no pre-authorization was obtained. Please obtain pre-authorization from {policy.insurer} "
                    f"and resubmit this claim along with the pre-authorization reference number."
                )
                checks.append(PolicyCheckResult(code="PRE_AUTH_MISSING", status=CheckStatus.FAIL, message=message, policy_reference="pre_authorization.required_for"))
                return self._rejected(checks, ["PRE_AUTH_MISSING"], canonical, message, trace)
        checks.append(PolicyCheckResult(code="PRE_AUTHORIZATION", status=CheckStatus.PASS, message="No outstanding pre-authorization requirement for this claim."))

        # Step 7: per-claim limit. DENTAL is exempt -- it has its own (higher) category sub-limit
        # and is evaluated at the line-item level in Step 9, which is the more specific rule.
        if claim.claim_category != ClaimCategory.DENTAL and claim.claimed_amount > policy.coverage.per_claim_limit:
            message = (
                f"The claimed amount ({_fmt(claim.claimed_amount)}) exceeds the per-claim limit of "
                f"{_fmt(policy.coverage.per_claim_limit)} under your policy."
            )
            checks.append(PolicyCheckResult(code="PER_CLAIM_EXCEEDED", status=CheckStatus.FAIL, message=message, policy_reference="coverage.per_claim_limit"))
            return self._rejected(checks, ["PER_CLAIM_EXCEEDED"], canonical, message, trace)
        checks.append(PolicyCheckResult(code="PER_CLAIM_LIMIT", status=CheckStatus.PASS, message=f"Claimed amount ({_fmt(claim.claimed_amount)}) is within the per-claim limit of {_fmt(policy.coverage.per_claim_limit)}."))

        # Step 8: annual OPD limit (informational)
        projected_ytd = claim.ytd_claims_amount + claim.claimed_amount
        if projected_ytd > policy.coverage.annual_opd_limit:
            checks.append(PolicyCheckResult(
                code="ANNUAL_OPD_LIMIT_EXCEEDED", status=CheckStatus.WARN,
                message=f"Year-to-date OPD claims ({_fmt(projected_ytd)}) exceed the annual OPD limit of {_fmt(policy.coverage.annual_opd_limit)}.",
                policy_reference="coverage.annual_opd_limit",
            ))

        # Step 9: category-specific line items / eligible base
        if claim.claim_category == ClaimCategory.DENTAL:
            decision_hint, eligible_base, line_item_results, rejection_reasons = self._evaluate_dental(
                input.extractions, category_rules, checks
            )
            canonical.dental_procedures = classify_dental_procedures(self._gather_line_items(input.extractions), category_rules)
            if decision_hint == "REJECTED":
                message = "; ".join(rejection_reasons)
                checks.append(PolicyCheckResult(code="DENTAL_PROCEDURES_NOT_COVERED", status=CheckStatus.FAIL, message=message, policy_reference="opd_categories.dental.excluded_procedures"))
                return self._rejected(checks, ["DENTAL_PROCEDURES_NOT_COVERED"], canonical, message, trace, line_item_results=line_item_results)
        else:
            eligible_base = claim.claimed_amount
            line_item_results = None
            sub_limit = category_rules.sub_limit
            if eligible_base > sub_limit:
                checks.append(PolicyCheckResult(
                    code="SUB_LIMIT_EXCEEDED", status=CheckStatus.WARN,
                    message=f"The claimed amount ({_fmt(eligible_base)}) exceeds the typical sub-limit of {_fmt(sub_limit)} for {category_key} claims; flagged for review.",
                    policy_reference=f"opd_categories.{category_key}.sub_limit",
                ))
            decision_hint = "APPROVED"

        # Step 10: financial calculation (discount applied before co-pay)
        is_network = bool(claim.hospital_name) and claim.hospital_name in policy.network_hospitals
        network_discount_percent = category_rules.network_discount_percent if is_network else 0.0
        amount_after_discount = eligible_base * (1 - network_discount_percent / 100)
        copay_percent = category_rules.copay_percent
        copay_amount = round(amount_after_discount * (copay_percent / 100), 2)
        approved_amount = round(amount_after_discount - copay_amount, 2)

        financial_breakdown = FinancialBreakdown(
            claimed_amount=claim.claimed_amount,
            eligible_base=eligible_base,
            network_hospital=is_network,
            network_discount_percent=network_discount_percent,
            amount_after_discount=round(amount_after_discount, 2),
            copay_percent=copay_percent,
            copay_amount=copay_amount,
            approved_amount=approved_amount,
        )
        checks.append(PolicyCheckResult(
            code="FINANCIAL_CALCULATION", status=CheckStatus.INFO,
            message=(
                f"Eligible base {_fmt(eligible_base)}"
                + (f", network discount {network_discount_percent:.0f}% -> {_fmt(amount_after_discount)}" if network_discount_percent else "")
                + (f", co-pay {copay_percent:.0f}% (-{_fmt(copay_amount)})" if copay_percent else "")
                + f" = approved {_fmt(approved_amount)}."
            ),
            detail=financial_breakdown.model_dump(),
        ))

        trace.add(
            TraceStage.POLICY_EVALUATION, "result", TraceStatus.SUCCESS,
            f"Policy evaluation result: {decision_hint}, approved {_fmt(approved_amount)} of {_fmt(claim.claimed_amount)}.",
            detail={"decision_hint": decision_hint, "approved_amount": approved_amount},
        )
        return PolicyEvaluationResult(
            decision_hint=decision_hint,
            checks=checks,
            rejection_reasons=[],
            approved_amount=approved_amount,
            financial_breakdown=financial_breakdown,
            line_item_results=line_item_results,
            canonical_mapping=canonical,
        )

    def _evaluate_dental(
        self, extractions: list[ExtractionResult], category_rules: OPDCategoryRules, checks: list[PolicyCheckResult]
    ) -> tuple[str, float, list[LineItemResult], list[str]]:
        line_items = self._gather_line_items(extractions)
        classifications = classify_dental_procedures(line_items, category_rules)

        line_item_results: list[LineItemResult] = []
        eligible_base = 0.0
        rejection_reasons: list[str] = []
        for c in classifications:
            if c.status == "COVERED":
                line_item_results.append(LineItemResult(description=c.description, amount=c.amount, status="APPROVED"))
                eligible_base += c.amount
            elif c.status == "EXCLUDED":
                reason = f"'{c.description}' is a cosmetic/excluded dental procedure not covered under this policy."
                line_item_results.append(LineItemResult(description=c.description, amount=c.amount, status="REJECTED", reason=reason))
                rejection_reasons.append(reason)
            else:
                reason = f"'{c.description}' is not a recognized covered dental procedure and has been flagged for manual review."
                line_item_results.append(LineItemResult(description=c.description, amount=c.amount, status="REJECTED", reason=reason))
                rejection_reasons.append(reason)
                checks.append(PolicyCheckResult(code="UNRECOGNIZED_DENTAL_PROCEDURE", status=CheckStatus.WARN, message=reason))

        sub_limit = category_rules.sub_limit
        if eligible_base > sub_limit:
            checks.append(PolicyCheckResult(
                code="SUB_LIMIT_EXCEEDED", status=CheckStatus.WARN,
                message=f"The eligible dental amount ({_fmt(eligible_base)}) exceeds the dental sub-limit of {_fmt(sub_limit)}; capped at the sub-limit.",
                policy_reference="opd_categories.dental.sub_limit",
            ))
            eligible_base = sub_limit

        total_claimed = sum(c.amount for c in classifications)
        if eligible_base <= 0:
            decision_hint = "REJECTED"
        elif eligible_base < total_claimed:
            decision_hint = "PARTIAL"
        else:
            decision_hint = "APPROVED"

        return decision_hint, eligible_base, line_item_results, rejection_reasons

    def _rejected(
        self,
        checks: list[PolicyCheckResult],
        rejection_reasons: list[str],
        canonical: CanonicalMapping,
        message: str,
        trace: ClaimTrace,
        line_item_results: list[LineItemResult] | None = None,
    ) -> PolicyEvaluationResult:
        trace.add(
            TraceStage.POLICY_EVALUATION, "result", TraceStatus.SUCCESS,
            f"Policy evaluation result: REJECTED ({', '.join(rejection_reasons)}).",
            detail={"decision_hint": "REJECTED", "rejection_reasons": rejection_reasons, "message": message},
        )
        return PolicyEvaluationResult(
            decision_hint="REJECTED",
            checks=checks,
            rejection_reasons=rejection_reasons,
            approved_amount=0,
            financial_breakdown=None,
            line_item_results=line_item_results,
            canonical_mapping=canonical,
        )

    def _gather_diagnosis_treatment(self, extractions: list[ExtractionResult]) -> tuple[str | None, str | None]:
        diagnoses = [e.content.diagnosis for e in extractions if e.content.diagnosis]
        treatments = [e.content.treatment for e in extractions if e.content.treatment]
        return ("; ".join(diagnoses) or None, "; ".join(treatments) or None)

    def _gather_tests_ordered(self, extractions: list[ExtractionResult]) -> list[str]:
        tests: list[str] = []
        for e in extractions:
            tests.extend(e.content.tests_ordered)
        return tests

    def _gather_line_items(self, extractions: list[ExtractionResult]) -> list[LineItem]:
        items: list[LineItem] = []
        for e in extractions:
            items.extend(e.content.line_items)
        return items
