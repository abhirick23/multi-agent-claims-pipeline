"""Shared Streamlit rendering helpers for a ``ClaimResult``: the decision summary, financial /
line-item breakdowns, fraud signals, and the full explainability trace. Used by every page so the
"why did the system decide this" view is consistent across the demo, eval suite, and live form.
"""
from __future__ import annotations

import streamlit as st

from app.models.result import ClaimResult
from app.models.trace import ClaimTrace

STATUS_ICONS = {"SUCCESS": "✅", "DEGRADED": "⚠️", "FAILED": "❌", "INFO": "ℹ️"}
DECISION_ICONS = {"APPROVED": "✅", "PARTIAL": "🟡", "REJECTED": "❌", "MANUAL_REVIEW": "🔎"}


def render_trace(trace: ClaimTrace) -> None:
    """Render every TraceEntry as an expander, in order, with its structured detail as JSON."""
    if not trace.entries:
        st.caption("No trace entries.")
        return
    for entry in trace.entries:
        icon = STATUS_ICONS.get(entry.status.value, "•")
        with st.expander(f"{icon} [{entry.stage.value}] {entry.step} — {entry.summary}"):
            if entry.detail:
                st.json(entry.detail)


def render_decision(result: ClaimResult) -> None:
    """Render the member-facing outcome: stopped-early message, or the full decision with
    financial breakdown, line-item breakdown, rejection reasons, and fraud signals."""
    if result.stopped_early:
        st.error("⛔ Processing stopped before a claim decision could be made.")
        st.markdown(f"**Message to member:**\n\n> {result.member_message}")
        return

    decision = result.decision
    if decision is None:
        st.warning("No decision was produced for this claim.")
        return

    icon = DECISION_ICONS.get(decision.decision, "•")
    cols = st.columns(3)
    cols[0].metric("Decision", f"{icon} {decision.decision}")
    cols[1].metric("Approved amount", f"₹{decision.approved_amount:,.2f}")
    cols[2].metric("Confidence", f"{decision.confidence_score:.2f}")

    if decision.manual_review_recommended:
        st.warning("**Manual review recommended:** " + " ".join(decision.manual_review_reasons))

    if decision.rejection_reasons:
        st.markdown("**Rejection reason(s):**")
        for r in decision.rejection_reasons:
            st.markdown(f"- **{r.code}**: {r.message}")

    if decision.financial_breakdown:
        fb = decision.financial_breakdown
        st.markdown("**Financial breakdown:**")
        st.table({
            "Claimed amount": [f"₹{fb.claimed_amount:,.2f}"],
            "Eligible base": [f"₹{fb.eligible_base:,.2f}"],
            "Network hospital": ["Yes" if fb.network_hospital else "No"],
            "Network discount": [f"{fb.network_discount_percent:.0f}%"],
            "After discount": [f"₹{fb.amount_after_discount:,.2f}"],
            "Co-pay": [f"{fb.copay_percent:.0f}% (₹{fb.copay_amount:,.2f})"],
            "Approved amount": [f"₹{fb.approved_amount:,.2f}"],
        })

    if decision.line_item_breakdown:
        st.markdown("**Line-item breakdown:**")
        for item in decision.line_item_breakdown:
            item_icon = "✅" if item.status == "APPROVED" else "❌"
            line = f"{item_icon} {item.description} — ₹{item.amount:,.2f} ({item.status})"
            if item.reason:
                line += f" — {item.reason}"
            st.markdown(f"- {line}")

    if result.fraud_check and result.fraud_check.signals:
        st.markdown("**Fraud signals:**")
        for signal in result.fraud_check.signals:
            st.markdown(f"- **{signal.code}** ({signal.severity}): {signal.message}")

    if decision.notes:
        st.markdown("**Notes:**")
        for note in decision.notes:
            st.markdown(f"- {note}")
