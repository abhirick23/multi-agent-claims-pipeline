"""Typed model of policy_terms.json. Loaded once by PolicyRepository and treated as read-only
configuration -- no policy logic is hardcoded in agents, only read from these structures."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class FamilyFloater(BaseModel):
    enabled: bool
    combined_limit: float
    covered_relationships: list[str] = []


class Coverage(BaseModel):
    sum_insured_per_employee: float
    annual_opd_limit: float
    per_claim_limit: float
    family_floater: FamilyFloater


class OPDCategoryRules(BaseModel):
    """Superset of fields across all opd_categories entries. Most fields are category-specific
    and optional -- absence simply means that rule does not apply to this category."""

    model_config = ConfigDict(extra="ignore")

    sub_limit: float
    copay_percent: float = 0
    network_discount_percent: float = 0
    requires_prescription: bool = False
    requires_pre_auth: bool = False
    covered: bool = True

    # diagnostic-specific
    pre_auth_threshold: Optional[float] = None
    high_value_tests_requiring_pre_auth: list[str] = []

    # pharmacy-specific
    branded_drug_copay_percent: Optional[float] = None
    generic_mandatory: Optional[bool] = None

    # dental-specific
    requires_dental_report: Optional[bool] = None
    covered_procedures: list[str] = []
    excluded_procedures: list[str] = []

    # vision-specific
    covered_items: list[str] = []
    excluded_items: list[str] = []

    # alternative-medicine-specific
    requires_registered_practitioner: Optional[bool] = None
    max_sessions_per_year: Optional[int] = None
    covered_systems: list[str] = []


class WaitingPeriods(BaseModel):
    initial_waiting_period_days: int
    pre_existing_conditions_days: int
    specific_conditions: dict[str, int] = {}


class Exclusions(BaseModel):
    conditions: list[str] = []
    dental_exclusions: list[str] = []
    vision_exclusions: list[str] = []


class PreAuthorization(BaseModel):
    required_for: list[str] = []
    validity_days: int


class SubmissionRules(BaseModel):
    deadline_days_from_treatment: int
    minimum_claim_amount: float
    currency: str = "INR"


class DocumentRequirement(BaseModel):
    required: list[str] = []
    optional: list[str] = []


class FraudThresholds(BaseModel):
    same_day_claims_limit: int
    monthly_claims_limit: int
    high_value_claim_threshold: float
    auto_manual_review_above: float
    fraud_score_manual_review_threshold: float


class MemberRecord(BaseModel):
    member_id: str
    name: str
    date_of_birth: str
    gender: str
    relationship: str
    # Dependents inherit the primary member's join_date for waiting-period purposes; only
    # present on roster entries for primary (SELF) members in policy_terms.json.
    join_date: Optional[str] = None
    dependents: list[str] = []
    primary_member_id: Optional[str] = None


class PolicyTerms(BaseModel):
    policy_id: str
    policy_name: str
    insurer: str
    coverage: Coverage
    opd_categories: dict[str, OPDCategoryRules]
    waiting_periods: WaitingPeriods
    exclusions: Exclusions
    pre_authorization: PreAuthorization
    network_hospitals: list[str] = []
    submission_rules: SubmissionRules
    document_requirements: dict[str, DocumentRequirement]
    fraud_thresholds: FraudThresholds
    members: list[MemberRecord] = []
