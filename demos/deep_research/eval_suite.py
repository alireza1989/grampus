#!/usr/bin/env python
"""Standalone eval runner for the Deep Research Demo.

Usage:
    python demos/deep_research/eval_suite.py
    nexus eval demos/deep_research/eval_suite.py --format text
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed directly as a script
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


async def main() -> None:
    from demos.deep_research.eval.suite import create_suite

    suite = create_suite()
    result = await suite.run()

    print(f"\n{'=' * 60}")
    print("  Deep Research Eval Results")
    print(f"{'=' * 60}")
    print(f"  Suite:    {result.suite_name}")
    print(f"  Cases:    {result.total_cases}")
    print(f"  Passed:   {result.passed}")
    print(f"  Failed:   {result.failed}")
    print(f"  Errors:   {result.errors}")
    print(f"  Pass rate: {result.pass_rate:.0%}")
    print(f"  Cost:     ${result.total_cost_usd:.4f}")
    print(f"  Avg time: {result.avg_duration_seconds:.2f}s/case")
    print(f"\n{'─' * 60}")
    print("  Per-case results:")
    print(f"{'─' * 60}")

    for cr in result.case_results:
        status = "✓ PASS" if cr.passed else ("✗ FAIL" if not cr.error else "⚠ ERROR")
        tags = f" [{', '.join(cr.tags)}]" if cr.tags else ""
        print(f"  {status}  {cr.case_name}{tags}")
        if not cr.passed:
            for ar in cr.assertion_results:
                if not ar.passed:
                    print(f"           → {ar.assertion_type}: {ar.detail}")
            if cr.error:
                print(f"           → error: {cr.error}")

    print()
    threshold = 0.75
    if result.pass_rate < threshold:
        print(f"  ⚠ Pass rate {result.pass_rate:.0%} is below threshold {threshold:.0%}")
    else:
        print(f"  ✓ Pass rate {result.pass_rate:.0%} meets threshold {threshold:.0%}")
    print()


# Expose create_suite and get_baseline at module level for ``nexus eval``
from demos.deep_research.eval.baseline import get_baseline  # noqa: E402
from demos.deep_research.eval.suite import create_suite  # noqa: E402

__all__ = ["create_suite", "get_baseline"]

if __name__ == "__main__":
    asyncio.run(main())
