"""Prompt templates for Gemini vision/text calls. Kept separate from gemini_client.py so the
wording can be iterated on without touching call/retry plumbing."""
from __future__ import annotations

CLASSIFY_DOCUMENT_PROMPT = """You are reviewing a document submitted as part of a health insurance claim.

Look at the image and determine:
1. `document_type` -- one of: PRESCRIPTION, HOSPITAL_BILL, LAB_REPORT, PHARMACY_BILL,
   DISCHARGE_SUMMARY, DENTAL_REPORT, DIAGNOSTIC_REPORT, or UNKNOWN if it doesn't match any of these.
2. `quality` -- GOOD if the document is fully legible, POOR if parts are hard to read but the
   document type and key fields are still identifiable, or UNREADABLE if the image is too blurry,
   dark, cropped, or corrupted to make out the content.
3. `patient_name_on_doc` -- the patient's full name as printed on the document, if visible.
   Return null if no patient name is visible.

Be conservative: if you cannot confidently identify the document type, return UNKNOWN rather than
guessing."""


EXTRACT_CONTENT_PROMPT = """You are extracting structured data from a health insurance claim document
of type `{document_type}`.

Read the image carefully and populate every field you can find evidence for. Leave fields null /
empty if they are not present on this document -- do not guess or fabricate values. For
`line_items`, list each billed item with its description and amount. For dates, use YYYY-MM-DD
format. For `medicines` and `tests_ordered`, list each as a separate string exactly as written."""


CANONICAL_MAPPING_PROMPT = """You are mapping a claim's diagnosis/treatment description onto a health
insurance policy's defined terms.

Diagnosis text: "{diagnosis_text}"
Treatment text: "{treatment_text}"

Candidate waiting-period condition keys (the policy imposes an extra waiting period for these
conditions): {waiting_period_keys}

Candidate exclusion terms (the policy does not cover these at all): {exclusion_terms}

For each candidate list, decide whether the diagnosis/treatment text above refers to that same
condition or procedure (consider synonyms, abbreviations, and related medical terminology -- e.g.
"T2DM" matches "diabetes", "bariatric surgery" matches "obesity and weight loss programs").

Return:
- `waiting_period_key`: the single best-matching key from the waiting-period candidates, or null
  if none match.
- `matched_exclusion_terms`: every exclusion term from the candidates that this diagnosis/treatment
  falls under (can be empty).
- `confidence`: your confidence (0-1) in this overall mapping.
- `rationale`: one or two sentences explaining your reasoning."""
