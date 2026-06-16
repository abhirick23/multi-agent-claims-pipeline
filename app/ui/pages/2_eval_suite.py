"""Run the full ``test_cases.json`` evaluation suite (all 12 cases, injection mode, zero API
calls) from the UI and inspect every case's checks, decision, and trace -- the same logic that
produces ``docs/EVAL_REPORT.md``, with an optional button to refresh that file on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.core.logging_config import get_logger
from app.ui.components.trace_renderer import render_decision, render_trace
from eval.run_eval import REPORT_PATH, render_report, run_all

_log = get_logger(__name__)

st.set_page_config(page_title="Eval Suite — Plum Claims Processor", page_icon="🧪", layout="wide")
st.title("🧪 Evaluation Suite")
st.markdown(
    "Runs the full pipeline against all 12 test scenarios and verifies each decision, "
    "approved amount, and rejection reason against the expected outcome."
)

col1, col2 = st.columns(2)
run_clicked = col1.button("Run evaluation suite", type="primary")
write_clicked = col2.button("Run and refresh docs/EVAL_REPORT.md")

if run_clicked or write_clicked:
    _log.info("[UI:eval_suite] Eval suite triggered (write_report=%s)", write_clicked)
    with st.spinner("Running all 12 test cases..."):
        test_cases, rows = run_all()
    _log.info("[UI:eval_suite] Eval suite complete — %d/%d cases passed", sum(1 for r in rows if all(c.passed for c in r["checks"])), len(rows))

    total_checks = sum(len(r["checks"]) for r in rows)
    passed_checks = sum(1 for r in rows for c in r["checks"] if c.passed)
    cases_fully_passed = sum(1 for r in rows if all(c.passed for c in r["checks"]))

    if write_clicked:
        report = render_report(rows)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        st.success(f"Wrote {REPORT_PATH.relative_to(ROOT)}")

    st.metric("Cases passing every check", f"{cases_fully_passed}/{len(rows)}")
    st.caption(f"{passed_checks}/{total_checks} individual checks passed.")

    st.subheader("Summary")
    summary_rows = []
    for row in rows:
        case, result, checks = row["case"], row["result"], row["checks"]
        expected_decision = case.get("expected", {}).get("decision")
        expected_str = "stopped early (no decision)" if expected_decision is None and "decision" in case.get("expected", {}) else str(expected_decision)
        actual_str = "stopped early (no decision)" if result.stopped_early else (result.decision.decision if result.decision else "—")
        verdict = "✅ PASS" if all(c.passed for c in checks) else "❌ FAIL"
        summary_rows.append({
            "Case": case["case_id"],
            "Name": case["case_name"],
            "Expected": expected_str,
            "Actual": actual_str,
            "Result": verdict,
        })
    st.table(summary_rows)

    st.subheader("Per-case detail")
    for row in rows:
        case, result, checks = row["case"], row["result"], row["checks"]
        overall = "✅ PASS" if all(c.passed for c in checks) else "❌ FAIL"
        with st.expander(f"{overall} {case['case_id']} — {case['case_name']}"):
            st.caption(case["description"])
            st.markdown("**Checks:**")
            for c in checks:
                icon = "✅" if c.passed else "❌"
                st.markdown(f"- {icon} **{c.label}** — {c.detail}")

            st.markdown("**Decision**")
            render_decision(result)

            st.markdown("**Explainability trace**")
            render_trace(result.trace)
