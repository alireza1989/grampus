#!/usr/bin/env python3
"""Print per-test cost breakdown from test_costs.json.

Written by the real_llm test session's cost_budget fixture.
Exit code 0 always (informational only — CI should not fail on cost reporting).
"""

from __future__ import annotations

import json
import pathlib
import sys

COSTS_FILE = pathlib.Path("test_costs.json")
WARNING_THRESHOLD_USD = 1.00


def main() -> None:
    if not COSTS_FILE.exists():
        print("No test_costs.json found — no real LLM tests ran.")
        return

    try:
        records = json.loads(COSTS_FILE.read_text())
    except json.JSONDecodeError as e:
        print(f"Cannot parse test_costs.json: {e}", file=sys.stderr)
        return

    if not records:
        print("test_costs.json is empty.")
        return

    print("\n── Real LLM Test Cost Breakdown ─────────────────────────")
    print(f"{'Test':<60} {'Cost':>10}")
    print("─" * 72)

    total = 0.0
    for record in records:
        name = record.get("test", "unknown")[:58]
        cost = record.get("cost_usd", 0.0)
        total += cost
        print(f"{name:<60} ${cost:>8.6f}")

    print("─" * 72)
    print(f"{'TOTAL':<60} ${total:>8.6f}")

    if total > WARNING_THRESHOLD_USD:
        print(f"\nWARNING: Total cost ${total:.4f} exceeds ${WARNING_THRESHOLD_USD:.2f} threshold.")
    else:
        print(f"\nTotal cost ${total:.6f} is within budget.")


if __name__ == "__main__":
    main()
