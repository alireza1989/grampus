"""Event loop compatibility for running async code in Jupyter notebooks."""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any


def ensure_async_compatible() -> None:
    """Patch the running event loop to allow nested asyncio.run() calls.

    Applies nest_asyncio if available. No-op if not installed or if there
    is no running loop (e.g. in regular Python scripts).
    """
    try:
        import asyncio

        import nest_asyncio

        try:
            loop = asyncio.get_running_loop()
            nest_asyncio.apply(loop)
        except RuntimeError:
            pass  # no running loop — asyncio.run() works fine
    except ImportError:
        pass  # nest_asyncio not installed — user must use await directly


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine, handling both Jupyter (running loop) and script contexts.

    In Jupyter: uses the running loop via nest_asyncio if available,
    otherwise raises a helpful error directing the user to ``await`` directly.
    In scripts: uses asyncio.run().
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # We're in Jupyter or another async context.
        try:
            import nest_asyncio

            nest_asyncio.apply(loop)
            future = asyncio.ensure_future(coro)
            return loop.run_until_complete(future)
        except ImportError as err:
            raise RuntimeError(
                "In Jupyter notebooks, use 'await' directly:\n"
                "  result = await notebook.run('your task')\n"
                "Or install nest_asyncio: pip install nest_asyncio"
            ) from err
    except RuntimeError as exc:
        if (
            "no running event loop" in str(exc).lower()
            or "no current event loop" in str(exc).lower()
        ):
            return asyncio.run(coro)
        raise
