"""Home page: an overview of the pipeline plus a "quick demo" that replays any of the 12
``test_cases.json`` scenarios through the real Orchestrator in injection mode (zero Gemini API
calls), showing the decision and the full explainability trace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.core.logging_config import get_logger
from app.core.orchestrator import Orchestrator
from app.models.common import ClaimInput
from app.ui.components.trace_renderer import render_decision, render_trace

_log = get_logger(__name__)

st.set_page_config(page_title="Plum Claims Processor", page_icon="🏥", layout="wide")

st.title("🏥 Plum Health Insurance Claims Processor")
st.markdown(
    "A multi-agent pipeline turns a claim submission into an explainable decision:\n\n"
    "**Document Verification → Extraction → Policy Evaluation → Fraud Detection → Decision**\n\n"
    "Every step is recorded in a trace, every policy rule is read from `policy_terms.json` "
    "(nothing is hardcoded), and the pipeline never crashes — failures degrade gracefully and "
    "lower the confidence score instead.\n\n"
    "Use the sidebar to **submit a new claim** or **run the full evaluation suite**. "
    "Below, replay any of the 12 scripted test scenarios."
)


@st.cache_resource
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@st.cache_data
def load_test_cases() -> list[dict]:
    with open(ROOT / "ass_detail" / "test_cases.json", encoding="utf-8") as f:
        return json.load(f)["test_cases"]


st.header("Quick demo: replay a sample case")
test_cases = load_test_cases()
labels = [f"{tc['case_id']} — {tc['case_name']}" for tc in test_cases]
choice = st.selectbox("Sample case", labels)
case = test_cases[labels.index(choice)]

st.caption(case["description"])
with st.expander("Raw claim input"):
    st.json(case["input"])

if st.button("Run this claim through the pipeline", type="primary"):
    _log.info("[UI:home] Quick demo triggered — case=%s", case["case_id"])
    orchestrator = get_orchestrator()
    claim = ClaimInput.model_validate(case["input"])
    result = orchestrator.process_claim(claim, claim_ref=case["case_id"], record_in_ledger=False)
    _log.info("[UI:home] Quick demo complete — case=%s, decision=%s", case["case_id"], result.decision.decision if result.decision else "stopped_early")

    st.subheader("Decision")
    render_decision(result)

    st.subheader("Explainability trace")
    render_trace(result.trace)
