"""Submit a new claim through the pipeline."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.core.logging_config import get_logger
from app.core.orchestrator import Orchestrator
from app.models.common import ClaimCategory, ClaimInput, DocumentInput
from app.storage.policy_repository import PolicyRepository
from app.ui.components.trace_renderer import render_decision, render_trace

_log = get_logger(__name__)

st.set_page_config(page_title="Submit Claim — Plum Claims Processor", page_icon="📝", layout="wide")
st.title("📝 Submit a New Claim")


@st.cache_resource
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@st.cache_resource
def get_policy_repo() -> PolicyRepository:
    return PolicyRepository()


policy_repo = get_policy_repo()
policy = policy_repo.policy

member_options = {f"{m.member_id} — {m.name} ({m.relationship})": m.member_id for m in policy.members}
hospital_options = ["(none)"] + policy.network_hospitals + ["Other (not in network)"]

with st.form("claim_form"):
    st.subheader("Claim details")
    col1, col2, col3 = st.columns(3)
    with col1:
        member_label = st.selectbox("Member", list(member_options.keys()))
        member_id = member_options[member_label]
    with col2:
        claim_category = st.selectbox("Claim category", [c.value for c in ClaimCategory])
    with col3:
        claimed_amount = st.number_input("Claimed amount (₹)", min_value=0.0, step=100.0, value=1500.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        treatment_date = st.date_input("Treatment date")
    with col5:
        submission_date = st.date_input("Submission date", value=treatment_date)
    with col6:
        pre_auth_obtained = st.checkbox("Pre-authorization obtained?")

    hospital_choice = st.selectbox("Hospital / provider", hospital_options)
    if hospital_choice == "(none)":
        hospital_name = None
    elif hospital_choice == "Other (not in network)":
        hospital_name = st.text_input("Hospital / provider name")
    else:
        hospital_name = hospital_choice

    st.subheader("Documents")
    num_docs = st.number_input("Number of documents", min_value=1, max_value=5, value=2, step=1)

    uploads = []
    for i in range(int(num_docs)):
        upload = st.file_uploader(
            f"Document {i + 1} (prescription, bill, lab report, etc.)",
            type=["png", "jpg", "jpeg", "pdf"],
            key=f"upload_{i}",
        )
        uploads.append(upload)

    submitted = st.form_submit_button("Submit claim", type="primary")

if submitted:
    documents: list[DocumentInput] = []
    for i, upload in enumerate(uploads):
        file_id = f"F{i + 1:03d}"
        file_path = None
        file_name = None
        if upload is not None:
            suffix = Path(upload.name).suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(upload.getvalue())
            tmp.close()
            file_path = tmp.name
            file_name = upload.name

        documents.append(DocumentInput(
            file_id=file_id,
            file_name=file_name,
            file_path=file_path,
        ))

    claim = ClaimInput(
        member_id=member_id,
        policy_id=policy.policy_id,
        claim_category=ClaimCategory(claim_category),
        treatment_date=treatment_date,
        submission_date=submission_date,
        claimed_amount=claimed_amount,
        hospital_name=hospital_name,
        pre_auth_obtained=pre_auth_obtained,
        documents=documents,
    )

    _log.info(
        "[UI:submit_claim] Claim submitted — member=%s, category=%s, amount=%.2f, docs=%d",
        member_id, claim_category, claimed_amount, len(documents),
    )

    with st.spinner("Processing claim..."):
        orchestrator = get_orchestrator()
        result = orchestrator.process_claim(claim)

    _log.info(
        "[UI:submit_claim] Result — decision=%s, amount=%.2f",
        result.decision.decision if result.decision else "stopped_early",
        result.decision.approved_amount if result.decision else 0,
    )

    st.subheader("Decision")
    render_decision(result)

    st.subheader("Explainability trace")
    render_trace(result.trace)
