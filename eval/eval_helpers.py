"""Comparison helpers for ``run_eval.py``.

Each test case's ``expected`` block mixes hard quantitative assertions (``decision``,
``approved_amount``, ``rejection_reasons``, ``confidence_score``) with qualitative
``system_must`` requirements written in prose. ``evaluate_case`` checks the quantitative fields
generically and delegates the qualitative ``system_must`` bullets to a per-case function below --
each one inspects the actual ``ClaimResult`` for the concrete, checkable signal that satisfies
that bullet (e.g. "tell the member what document type is needed" -> the member-facing message
actually names both the uploaded and required document types).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.result import ClaimResult
from app.models.trace import ClaimTrace


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str


def evaluate_case(case: dict, result: ClaimResult) -> list[CheckResult]:
    expected = case.get("expected", {})
    checks: list[CheckResult] = []

    if "decision" in expected:
        exp_decision = expected["decision"]
        if exp_decision is None:
            actual = result.decision.decision if result.decision else None
            passed = result.stopped_early and result.decision is None
            checks.append(CheckResult("decision", passed, f"expected the pipeline to stop early with no decision; got stopped_early={result.stopped_early}, decision={actual}"))
        else:
            actual = result.decision.decision if result.decision else None
            passed = actual == exp_decision
            checks.append(CheckResult("decision", passed, f"expected {exp_decision!r}, got {actual!r}"))

    if "approved_amount" in expected:
        actual_amount = result.decision.approved_amount if result.decision else None
        passed = actual_amount == expected["approved_amount"]
        checks.append(CheckResult("approved_amount", passed, f"expected {expected['approved_amount']}, got {actual_amount}"))

    if "rejection_reasons" in expected:
        actual_codes = [r.code for r in result.decision.rejection_reasons] if result.decision else []
        passed = actual_codes == expected["rejection_reasons"]
        checks.append(CheckResult("rejection_reasons", passed, f"expected {expected['rejection_reasons']}, got {actual_codes}"))

    if "confidence_score" in expected:
        spec = expected["confidence_score"]
        threshold = float(spec.split()[-1])
        actual_conf = result.decision.confidence_score if result.decision else None
        passed = actual_conf is not None and actual_conf > threshold
        checks.append(CheckResult("confidence_score", passed, f"expected {spec}, got {actual_conf}"))

    case_id = case["case_id"]
    system_must_fn = SYSTEM_MUST_CHECKS.get(case_id)
    if system_must_fn:
        checks.extend(system_must_fn(case, result))
    elif expected.get("system_must"):
        for item in expected["system_must"]:
            checks.append(CheckResult(f"system_must: {item}", True, "not automatically checked (no qualitative checker registered for this case)"))

    return checks


def format_trace(trace: ClaimTrace) -> list[str]:
    return [f"[{e.stage.value}/{e.step}] {e.status.value}: {e.summary}" for e in trace.entries]


# --- Per-case qualitative ("system_must") checkers -----------------------------------------


def _tc001_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    checks = []
    stopped = result.stopped_early and result.decision is None
    checks.append(CheckResult("system_must: stop before any claim decision", stopped, f"stopped_early={result.stopped_early}, decision={result.decision}"))

    msg = result.member_message or ""
    names_present = "PRESCRIPTION" in msg and "HOSPITAL_BILL" in msg
    checks.append(CheckResult(
        "system_must: message names both the uploaded and required document types",
        names_present, f"member_message={msg!r}",
    ))
    checks.append(CheckResult(
        "system_must: not a generic error message",
        names_present and len(msg) > 40, f"message length={len(msg)}",
    ))
    return checks


def _tc002_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    msg = result.member_message or ""
    checks = []
    checks.append(CheckResult(
        "system_must: identifies the unreadable document",
        "blurry_bill.jpg" in msg or "PHARMACY_BILL" in msg, f"member_message={msg!r}",
    ))
    checks.append(CheckResult(
        "system_must: asks for a specific re-upload",
        "re-upload" in msg.lower(), f"member_message={msg!r}",
    ))
    not_rejected = result.stopped_early and result.decision is None
    checks.append(CheckResult(
        "system_must: does not reject the claim outright",
        not_rejected, f"stopped_early={result.stopped_early}, decision={result.decision}",
    ))
    return checks


def _tc003_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    msg = result.member_message or ""
    checks = []
    checks.append(CheckResult(
        "system_must: detects documents belong to different people",
        "different patients" in msg.lower(), f"member_message={msg!r}",
    ))
    checks.append(CheckResult(
        "system_must: surfaces the specific names found",
        "Rajesh Kumar" in msg and "Arjun Mehta" in msg, f"member_message={msg!r}",
    ))
    no_decision = result.stopped_early and result.decision is None
    checks.append(CheckResult(
        "system_must: does not proceed to a claim decision",
        no_decision, f"stopped_early={result.stopped_early}, decision={result.decision}",
    ))
    return checks


def _tc004_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    fb = result.decision.financial_breakdown if result.decision else None
    copay_ok = fb is not None and fb.copay_percent == 10 and fb.copay_amount == 150
    return [CheckResult(
        "system_must: 10% co-pay applied on consultation (₹150 deducted)",
        copay_ok, f"financial_breakdown={fb.model_dump() if fb else None}",
    )]


def _tc005_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    reasons = result.decision.rejection_reasons if result.decision else []
    msg = reasons[0].message if reasons else ""
    return [CheckResult(
        "system_must: states the date member becomes eligible for diabetes claims",
        "2024-11-30" in msg, f"rejection message={msg!r}",
    )]


def _tc006_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    items = result.decision.line_item_breakdown if result.decision else None
    checks = []
    has_both = bool(items) and {i.status for i in items} == {"APPROVED", "REJECTED"}
    checks.append(CheckResult(
        "system_must: itemizes approved vs. rejected line items",
        has_both, f"line_item_breakdown={[i.model_dump() for i in items] if items else None}",
    ))
    rejected_have_reasons = bool(items) and all(i.reason for i in items if i.status == "REJECTED")
    checks.append(CheckResult(
        "system_must: states a reason for each rejected line item",
        rejected_have_reasons, "every REJECTED line item has a non-empty `reason`",
    ))
    return checks


def _tc007_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    reasons = result.decision.rejection_reasons if result.decision else []
    msg = reasons[0].message if reasons else ""
    checks = []
    checks.append(CheckResult(
        "system_must: explains pre-authorization was required and not obtained",
        "pre-authorization" in msg.lower(), f"rejection message={msg!r}",
    ))
    checks.append(CheckResult(
        "system_must: tells the member how to resubmit with pre-auth",
        "resubmit" in msg.lower() and "pre-auth" in msg.lower(), f"rejection message={msg!r}",
    ))
    return checks


def _tc008_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    reasons = result.decision.rejection_reasons if result.decision else []
    msg = reasons[0].message if reasons else ""
    return [CheckResult(
        "system_must: states both the per-claim limit and the claimed amount",
        "5,000" in msg and "7,500" in msg, f"rejection message={msg!r}",
    )]


def _tc009_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    fraud = result.fraud_check
    signals = fraud.signals if fraud else []
    checks = []
    checks.append(CheckResult(
        "system_must: flags the unusual same-day claim pattern",
        any(s.code == "SAME_DAY_CLAIMS_EXCEEDED" for s in signals), f"signals={[s.code for s in signals]}",
    ))
    checks.append(CheckResult(
        "system_must: routes to manual review rather than auto-rejecting",
        result.decision is not None and result.decision.decision == "MANUAL_REVIEW", f"decision={result.decision.decision if result.decision else None}",
    ))
    checks.append(CheckResult(
        "system_must: includes the specific signals that triggered the flag",
        len(signals) > 0 and len(result.decision.manual_review_reasons) > 0 if result.decision else False,
        f"manual_review_reasons={result.decision.manual_review_reasons if result.decision else None}",
    ))
    return checks


def _tc010_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    fb = result.decision.financial_breakdown if result.decision else None
    checks = []
    discount_before_copay = (
        fb is not None
        and fb.amount_after_discount == 3600
        and fb.copay_amount == 360
        and fb.approved_amount == 3240
    )
    checks.append(CheckResult(
        "system_must: network discount applied before co-pay",
        discount_before_copay, f"financial_breakdown={fb.model_dump() if fb else None}",
    ))
    checks.append(CheckResult(
        "system_must: breakdown of discount and co-pay shown in output",
        fb is not None, "ClaimDecision.financial_breakdown is populated",
    ))
    return checks


def _tc011_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    checks = []
    checks.append(CheckResult("system_must: does not crash", True, "process_claim returned a ClaimResult"))
    failed = [e for e in (result.extractions or []) if e.extraction_status == "FAILED"]
    checks.append(CheckResult(
        "system_must: indicates a component failed and was skipped",
        len(failed) > 0, f"failed extractions: {[e.file_id for e in failed]}",
    ))
    checks.append(CheckResult(
        "system_must: confidence score lower than a normal full-pipeline approval",
        result.decision is not None and result.decision.confidence_score < 1.0,
        f"confidence_score={result.decision.confidence_score if result.decision else None} (TC004 baseline = 1.0)",
    ))
    checks.append(CheckResult(
        "system_must: includes a note that manual review is recommended",
        result.decision is not None and result.decision.manual_review_recommended and len(result.decision.manual_review_reasons) > 0,
        f"manual_review_recommended={result.decision.manual_review_recommended if result.decision else None}",
    ))
    return checks


def _tc012_checks(case: dict, result: ClaimResult) -> list[CheckResult]:
    canonical = result.policy_evaluation.canonical_mapping if result.policy_evaluation else None
    matched_terms = [m.policy_term for m in canonical.exclusion_matches] if canonical else []
    return [CheckResult(
        "system_must: identifies the specific policy exclusion that applies",
        "Obesity and weight loss programs" in matched_terms, f"exclusion_matches={matched_terms} (method={canonical.method if canonical else None})",
    )]


SYSTEM_MUST_CHECKS = {
    "TC001": _tc001_checks,
    "TC002": _tc002_checks,
    "TC003": _tc003_checks,
    "TC004": _tc004_checks,
    "TC005": _tc005_checks,
    "TC006": _tc006_checks,
    "TC007": _tc007_checks,
    "TC008": _tc008_checks,
    "TC009": _tc009_checks,
    "TC010": _tc010_checks,
    "TC011": _tc011_checks,
    "TC012": _tc012_checks,
}
