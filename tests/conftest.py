"""Shared pytest fixtures: the real policy_terms.json (typed) and test_cases.json loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.policy import PolicyTerms

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def policy() -> PolicyTerms:
    with open(ROOT / "ass_detail" / "policy_terms.json", "r", encoding="utf-8") as f:
        return PolicyTerms.model_validate(json.load(f))


@pytest.fixture(scope="session")
def test_cases() -> list[dict]:
    with open(ROOT / "ass_detail" / "test_cases.json", "r", encoding="utf-8") as f:
        return json.load(f)["test_cases"]


def get_case(test_cases: list[dict], case_id: str) -> dict:
    for tc in test_cases:
        if tc["case_id"] == case_id:
            return tc
    raise KeyError(case_id)
