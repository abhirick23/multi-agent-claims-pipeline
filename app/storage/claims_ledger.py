"""A simple JSON-file claims ledger used by the Fraud Detection Agent to look up a member's
recent claim history (same-day / monthly counts) in the live Streamlit demo.

For the eval harness, ``ClaimInput.claims_history`` is supplied directly per test case and the
ledger is bypassed -- the Fraud Detection Agent merges whichever entries it's given with whatever
the ledger already holds for that member.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.models.common import ClaimHistoryEntry

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parents[2] / "data" / "claims_ledger.json"


class ClaimsLedger:
    def __init__(self, ledger_path: Path | str = DEFAULT_LEDGER_PATH):
        self._path = Path(ledger_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({})

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        with open(self._path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def get_history(self, member_id: str) -> list[ClaimHistoryEntry]:
        data = self._read()
        return [ClaimHistoryEntry.model_validate(entry) for entry in data.get(member_id, [])]

    def append(self, member_id: str, entry: ClaimHistoryEntry) -> None:
        data = self._read()
        data.setdefault(member_id, []).append(json.loads(entry.model_dump_json()))
        self._write(data)

    def merged_history(
        self, member_id: str, provided: list[ClaimHistoryEntry]
    ) -> list[ClaimHistoryEntry]:
        """Combine ledger history with whatever the caller already provided (eval injection),
        de-duplicating by claim_id."""
        seen: dict[str, ClaimHistoryEntry] = {}
        for entry in self.get_history(member_id) + provided:
            seen[entry.claim_id] = entry
        return list(seen.values())
