"""The accuracy budget, read from the gateway that enforces it.

Every judged run spends through one Bifrost governance virtual key with a hard USD cap.
The cap is the gateway's to enforce; what this tool adds is the discipline the owner asked
for around it: the key's ``current_usage`` is read before and after every judged arm, a
phase may spend at most its own cap, and no arm is launched whose projected cost would
leave the key under its floor. The key is named by its *id* (``BENCH_BUDGET_KEY_ID``), which
is not a secret; the token that authenticates calls never passes through here and nothing
this tool prints is the gateway's raw response.

    python -m benchmark.budget read --checkpoint "before A0"
    python -m benchmark.budget guard --projected-usd 0.20

The ledger (``benchmark/results/phase7/budget.json``) records the usage at the start of the
phase and every checkpoint, so the spend per arm is a subtraction anyone can repeat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.common import RESULTS

#: the gateway's governance endpoint for one virtual key
GOVERNANCE_PATH = "/api/governance/virtual-keys/{key_id}"
#: what Phase 7 may spend in total, and what must stay on the key when it stops
PHASE_CAP_USD = 6.0
FLOOR_USD = 2.0
LEDGER = RESULTS / "phase7" / "budget.json"


@dataclass(frozen=True)
class Budget:
    key_name: str
    max_limit_usd: float
    current_usage_usd: float

    @property
    def remaining_usd(self) -> float:
        return self.max_limit_usd - self.current_usage_usd

    def as_dict(self) -> dict[str, float | str]:
        return {
            "key_name": self.key_name,
            "max_limit_usd": self.max_limit_usd,
            "current_usage_usd": round(self.current_usage_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
        }


def gateway_root() -> str:
    """The gateway's root URL: ``BENCH_GATEWAY_URL``, else ``BIFROST_URL`` without ``/v1``."""
    explicit = os.environ.get("BENCH_GATEWAY_URL")
    if explicit:
        return explicit.rstrip("/")
    inference = os.environ.get("BIFROST_URL", "http://localhost:8091/v1").rstrip("/")
    return inference[: -len("/v1")] if inference.endswith("/v1") else inference


def read_budget(root: str, key_id: str) -> Budget:
    """The key's cap and usage, and nothing else of what the gateway returns."""
    with urllib.request.urlopen(root + GOVERNANCE_PATH.format(key_id=key_id), timeout=15) as res:  # noqa: S310 - the gateway is configuration
        payload = json.load(res)
    key = payload.get("virtual_key", payload)
    budgets = key.get("budgets") or []
    if not budgets:
        raise SystemExit(
            f"virtual key {key.get('name', key_id)!r} carries no budget; refusing to spend"
        )
    first = budgets[0]
    return Budget(
        key_name=str(key.get("name", key_id)),
        max_limit_usd=float(first["max_limit"]),
        current_usage_usd=float(first.get("current_usage", 0.0)),
    )


def guard(
    budget: Budget,
    *,
    projected_usd: float,
    phase_spent_usd: float,
    phase_cap_usd: float = PHASE_CAP_USD,
    floor_usd: float = FLOOR_USD,
) -> tuple[bool, str]:
    """Whether a run projected to cost ``projected_usd`` may start, and why."""
    if projected_usd < 0:
        raise ValueError("a projection cannot be negative")
    if phase_spent_usd + projected_usd > phase_cap_usd:
        return False, (
            f"refused: phase spend {phase_spent_usd:.4f} + projected {projected_usd:.4f} "
            f"exceeds the phase cap {phase_cap_usd:.2f} USD"
        )
    if budget.remaining_usd - projected_usd < floor_usd:
        return False, (
            f"refused: {budget.remaining_usd:.4f} USD left on {budget.key_name}, projected "
            f"{projected_usd:.4f} would leave less than the {floor_usd:.2f} USD floor"
        )
    return True, (
        f"ok: {projected_usd:.4f} USD projected, {budget.remaining_usd:.4f} left on "
        f"{budget.key_name}, phase spend {phase_spent_usd:.4f} of {phase_cap_usd:.2f}"
    )


class Ledger:
    """The phase's own record of the key: usage at the start, then every checkpoint."""

    def __init__(self, path: Path = LEDGER) -> None:
        self.path = path
        self.data: dict[str, Any] = (
            json.loads(path.read_text())
            if path.is_file()
            else {"phase_start_usage_usd": None, "checkpoints": []}
        )

    @property
    def phase_start(self) -> float | None:
        return self.data.get("phase_start_usage_usd")

    def phase_spent(self, budget: Budget) -> float:
        start = self.phase_start
        return 0.0 if start is None else max(0.0, budget.current_usage_usd - start)

    def checkpoint(self, label: str, budget: Budget) -> dict[str, Any]:
        if self.phase_start is None:
            self.data["phase_start_usage_usd"] = budget.current_usage_usd
        entry = {
            "label": label,
            "when": datetime.now(UTC).isoformat(),
            **budget.as_dict(),
            "phase_spent_usd": round(self.phase_spent(budget), 6),
        }
        self.data["checkpoints"].append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")
        return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-id", default=os.environ.get("BENCH_BUDGET_KEY_ID"))
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser(
        "read", help="print the key's cap and usage; optionally record a checkpoint"
    )
    read.add_argument("--checkpoint", default=None, help="label to record this reading under")
    check = commands.add_parser("guard", help="exit 1 unless a run of the projected cost may start")
    check.add_argument("--projected-usd", type=float, required=True)
    check.add_argument("--phase-cap", type=float, default=PHASE_CAP_USD)
    check.add_argument("--floor", type=float, default=FLOOR_USD)
    args = parser.parse_args(argv)
    if not args.key_id:
        raise SystemExit("set BENCH_BUDGET_KEY_ID (the governance key's id, not its token)")
    budget = read_budget(gateway_root(), args.key_id)
    ledger = Ledger(args.ledger)
    if args.command == "read":
        entry = ledger.checkpoint(args.checkpoint, budget) if args.checkpoint else budget.as_dict()
        print(json.dumps(entry, indent=2))
        return 0
    allowed, reason = guard(
        budget,
        projected_usd=args.projected_usd,
        phase_spent_usd=ledger.phase_spent(budget),
        phase_cap_usd=args.phase_cap,
        floor_usd=args.floor,
    )
    print(reason, file=sys.stderr)
    return 0 if allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
