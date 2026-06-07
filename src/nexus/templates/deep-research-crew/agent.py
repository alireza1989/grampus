"""Entry point for nexus run agent.py. Wraps crew.py for CLI compatibility."""

from __future__ import annotations

import asyncio

from crew import main

asyncio.run(main())
