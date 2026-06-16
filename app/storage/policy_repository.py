"""Loads policy_terms.json into the typed PolicyTerms model and provides lookups.

This is the single point where the raw policy configuration enters the system -- agents never
read policy_terms.json directly, they go through PolicyRepository so the rest of the codebase
works against the typed PolicyTerms / MemberRecord models.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.exceptions import MemberNotFoundError, PolicyConfigError
from app.models.policy import MemberRecord, PolicyTerms

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "ass_detail" / "policy_terms.json"


class PolicyRepository:
    def __init__(self, policy_path: Path | str = DEFAULT_POLICY_PATH):
        self._policy_path = Path(policy_path)
        self._policy: PolicyTerms = self._load()

    def _load(self) -> PolicyTerms:
        with open(self._policy_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return PolicyTerms.model_validate(raw)

    @property
    def policy(self) -> PolicyTerms:
        return self._policy

    def get_member(self, member_id: str) -> MemberRecord:
        for member in self._policy.members:
            if member.member_id == member_id:
                return member
        raise MemberNotFoundError(f"Member '{member_id}' was not found in the policy roster.")

    def get_effective_join_date(self, member: MemberRecord) -> str:
        """Dependents don't carry their own join_date in policy_terms.json -- they inherit the
        primary member's join_date for waiting-period purposes."""
        if member.join_date:
            return member.join_date
        if member.primary_member_id:
            primary = self.get_member(member.primary_member_id)
            if primary.join_date:
                return primary.join_date
        raise PolicyConfigError(
            f"Member '{member.member_id}' has no join_date and no resolvable primary member."
        )

    def get_category_rules(self, claim_category: str):
        key = claim_category.lower()
        if key not in self._policy.opd_categories:
            raise PolicyConfigError(
                f"No opd_categories configuration found for claim category '{claim_category}'."
            )
        return self._policy.opd_categories[key]

    def get_document_requirements(self, claim_category: str):
        key = claim_category.upper()
        if key not in self._policy.document_requirements:
            raise PolicyConfigError(
                f"No document_requirements configuration found for claim category '{claim_category}'."
            )
        return self._policy.document_requirements[key]
