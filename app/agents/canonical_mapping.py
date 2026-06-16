"""Canonical mapping: maps a claim's free-text diagnosis/treatment onto the policy's own
vocabulary (waiting-period condition keys, exclusion terms) and classifies dental line items.

**Tier 1 (always run, no LLM)**: deterministic keyword/synonym matching. The keyword lists below
are *synonyms for the policy's own terms* -- every ``policy_term`` / waiting-period key produced
is filtered against ``policy_terms.json`` so we never invent a term the policy doesn't define.

**Exclusions are checked before waiting periods**: if a diagnosis/treatment is permanently
excluded, any waiting period that might also apply to it is moot -- the claim is rejected for the
exclusion, not a temporary waiting period. So ``waiting_period_key`` is only considered when no
exclusion matched.

**Tier 2 (LLM fallback)**: only runs when Tier 1 found nothing *and* there is diagnosis/treatment
text to reason about. On any Gemini failure, the Tier 1 (empty) result is kept and the fallback
attempt is recorded in the trace -- this sub-step never fails the pipeline.
"""
from __future__ import annotations

import re

from app.agents.base import BaseAgent
from app.core.exceptions import GeminiAPIError
from app.models.extraction import LineItem
from app.models.policy import OPDCategoryRules, PolicyTerms
from app.models.policy_eval import CanonicalMapping, DentalProcedureClassification, ExclusionMatch
from app.models.trace import ClaimTrace, TraceStage, TraceStatus

# Synonyms for policy.waiting_periods.specific_conditions keys.
WAITING_PERIOD_KEYWORDS: dict[str, list[str]] = {
    "diabetes": ["diabetes", "diabetic", "t2dm", "t1dm"],
    "hypertension": ["hypertension", "high blood pressure"],
    "thyroid_disorders": ["thyroid", "hypothyroid", "hyperthyroid"],
    "joint_replacement": ["joint replacement", "knee replacement", "hip replacement"],
    "maternity": ["maternity", "pregnancy", "antenatal", "delivery"],
    "mental_health": ["depression", "anxiety", "psychiatric", "mental health"],
    "obesity_treatment": ["obesity", "bariatric", "weight loss", "bmi"],
    "hernia": ["hernia"],
    "cataract": ["cataract"],
}

# Synonyms for entries in policy.exclusions.conditions.
EXCLUSION_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Obesity and weight loss programs", ["obesity", "weight loss", "diet plan", "bariatric consultation", "nutrition program", "bmi"]),
    ("Bariatric surgery", ["bariatric surgery"]),
    ("Cosmetic or aesthetic procedures", ["cosmetic", "aesthetic", "teeth whitening", "veneers"]),
    ("Substance abuse treatment", ["substance abuse", "alcohol rehab", "drug rehab", "de-addiction"]),
    ("Self-inflicted injuries", ["self-inflicted", "suicide attempt"]),
    ("Infertility and assisted reproduction", ["infertility", "ivf", "assisted reproduction"]),
    ("Vaccination (non-medically necessary)", ["vaccination", "vaccine"]),
    ("Health supplements and tonics", ["supplement", "tonic"]),
    ("Experimental treatments", ["experimental treatment", "clinical trial"]),
]


def _contains_keyword(text: str, keyword: str) -> bool:
    """Whole-word/phrase match so e.g. "hernia" doesn't match inside "herniation"."""
    return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None


class CanonicalMapper(BaseAgent):
    def map(self, diagnosis_text: str | None, treatment_text: str | None, policy: PolicyTerms, trace: ClaimTrace) -> CanonicalMapping:
        combined = " ".join(t for t in (diagnosis_text, treatment_text) if t).lower()

        exclusion_matches: list[ExclusionMatch] = []
        for policy_term, keywords in EXCLUSION_KEYWORDS:
            if policy_term not in policy.exclusions.conditions:
                continue
            if any(_contains_keyword(combined, kw) for kw in keywords):
                exclusion_matches.append(
                    ExclusionMatch(policy_term=policy_term, matched_via="KEYWORD_MATCH", scope="WHOLE_CLAIM", confidence=0.95)
                )

        waiting_period_key: str | None = None
        if not exclusion_matches:
            for key, keywords in WAITING_PERIOD_KEYWORDS.items():
                if key not in policy.waiting_periods.specific_conditions:
                    continue
                if any(_contains_keyword(combined, kw) for kw in keywords):
                    waiting_period_key = key
                    break

        method = "KEYWORD_MATCH" if (exclusion_matches or waiting_period_key) else "NONE"
        rationale: str | None = None

        if method == "NONE" and combined:
            method, exclusion_matches, waiting_period_key, rationale = self._llm_fallback(
                diagnosis_text, treatment_text, policy, trace
            )

        trace.add(
            TraceStage.POLICY_EVALUATION,
            "canonical_mapping",
            TraceStatus.SUCCESS,
            f"Canonical mapping ({method}): waiting_period_key={waiting_period_key}, "
            f"exclusion_matches={[m.policy_term for m in exclusion_matches]}.",
            detail={
                "diagnosis_text": diagnosis_text,
                "treatment_text": treatment_text,
                "waiting_period_key": waiting_period_key,
                "exclusion_matches": [m.model_dump() for m in exclusion_matches],
                "method": method,
                "rationale": rationale,
            },
        )

        return CanonicalMapping(
            waiting_period_key=waiting_period_key,
            exclusion_matches=exclusion_matches,
            method=method,
            raw_diagnosis_text=diagnosis_text,
            raw_treatment_text=treatment_text,
            rationale=rationale,
        )

    def _llm_fallback(
        self, diagnosis_text: str | None, treatment_text: str | None, policy: PolicyTerms, trace: ClaimTrace
    ) -> tuple[str, list[ExclusionMatch], str | None, str | None]:
        candidate_wp_keys = list(policy.waiting_periods.specific_conditions.keys())
        candidate_exclusions = policy.exclusions.conditions
        try:
            resp = self.gemini.classify_canonical_category(
                diagnosis_text or "", treatment_text or "", candidate_wp_keys, candidate_exclusions
            )
        except GeminiAPIError as exc:
            short = str(exc).split("\n")[0][:120]
            trace.add(
                TraceStage.POLICY_EVALUATION,
                "canonical_mapping_llm_fallback",
                TraceStatus.DEGRADED,
                f"LLM-assisted canonical mapping unavailable (Gemini API error); keeping keyword-only result.",
                detail={"error": short},
            )
            return "NONE", [], None, None

        exclusion_matches = [
            ExclusionMatch(policy_term=t, matched_via="LLM_ASSISTED", scope="WHOLE_CLAIM", confidence=resp.confidence)
            for t in resp.matched_exclusion_terms
            if t in candidate_exclusions
        ]
        waiting_period_key = resp.waiting_period_key if (not exclusion_matches and resp.waiting_period_key in candidate_wp_keys) else None
        method = "LLM_ASSISTED" if (exclusion_matches or waiting_period_key) else "NONE"
        return method, exclusion_matches, waiting_period_key, resp.rationale


def classify_dental_procedures(line_items: list[LineItem], category_rules: OPDCategoryRules) -> list[DentalProcedureClassification]:
    """Exact (case-insensitive) match of each billed line item against the policy's
    ``covered_procedures`` / ``excluded_procedures`` lists for the DENTAL category."""
    covered = {p.lower() for p in category_rules.covered_procedures}
    excluded = {p.lower() for p in category_rules.excluded_procedures}

    results: list[DentalProcedureClassification] = []
    for item in line_items:
        key = item.description.strip().lower()
        if key in excluded:
            results.append(DentalProcedureClassification(description=item.description, amount=item.amount, status="EXCLUDED", matched_via="KEYWORD_MATCH"))
        elif key in covered:
            results.append(DentalProcedureClassification(description=item.description, amount=item.amount, status="COVERED", matched_via="KEYWORD_MATCH"))
        else:
            results.append(DentalProcedureClassification(description=item.description, amount=item.amount, status="UNKNOWN", matched_via=None))
    return results
