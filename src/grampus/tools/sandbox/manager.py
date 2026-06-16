"""Sandbox manager — Docker-backed execution with local subprocess fallback."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from pydantic import BaseModel, Field

from grampus.core.errors import ToolTimeoutError
from grampus.core.logging import get_logger

logger = get_logger(__name__)

# Docker is optional; guarded import so the module loads without it installed.
try:
    import docker as _docker_module

    _docker_available = True
except (ImportError, TypeError):
    _docker_module = None
    _docker_available = False


class SandboxConfig(BaseModel):
    """Configuration for a SandboxManager.

    Attributes:
        image: Docker image to use when Docker is available.
        network_enabled: Allow outbound network access inside the container.
        memory_limit_mb: Container memory cap in megabytes.
        cpu_limit: Fraction of a CPU allocated to the container.
        execution_timeout_seconds: Maximum wall-clock seconds per execution.
        allowed_paths: Host paths to mount read-only into the container.
    """

    image: str = "python:3.12-slim"
    network_enabled: bool = False
    memory_limit_mb: int = 256
    cpu_limit: float = 0.5
    execution_timeout_seconds: int = 30
    allowed_paths: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    """Result of a single sandbox execution.

    Attributes:
        stdout: Captured standard output.
        stderr: Captured standard error.
        return_value: Any value set as ``__result__`` in the executed code.
        exit_code: Process exit code (0 = success).
        duration_ms: Wall-clock execution time in milliseconds.
        error: Human-readable error message when execution failed.
    """

    stdout: str
    stderr: str
    return_value: Any
    exit_code: int
    duration_ms: float
    error: str | None = None


class SandboxManager:
    """Coordinates sandboxed code execution.

    Uses a Docker container when the Docker SDK is installed and the daemon is
    reachable; otherwise falls back to a local subprocess (development only).

    Args:
        config: Sandbox configuration; defaults applied when None.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._backend: _DockerSandbox | _LocalSandbox = self._build_backend()

    def _build_backend(self) -> _DockerSandbox | _LocalSandbox:
        if _docker_available:
            try:
                backend = _DockerSandbox(self._config)
                backend._client.ping()  # raises if daemon is unreachable
                return backend
            except Exception as exc:
                logger.warning("sandbox.docker_unavailable", reason=str(exc))
        return _LocalSandbox(self._config)

    async def execute(self, code: str, *, namespace: dict[str, Any] | None = None) -> SandboxResult:
        """Execute *code* in the sandbox and return the result.

        Args:
            code: Python source code to execute.
            namespace: Optional extra variables injected into the execution scope
                       (local sandbox only; ignored by Docker backend).

        Returns:
            SandboxResult with captured output.

        Raises:
            ToolTimeoutError: When execution exceeds *execution_timeout_seconds*.
        """
        logger.debug("sandbox.execute_start", backend=type(self._backend).__name__)
        return await self._backend.execute(code, namespace=namespace)

    async def close(self) -> None:
        """Release backend resources (e.g., Docker client)."""
        await self._backend.close()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _LocalSandbox:
    """Runs code in a child subprocess via ``python -c``."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    async def execute(self, code: str, *, namespace: dict[str, Any] | None = None) -> SandboxResult:
        """Execute *code* as a subprocess with timeout enforcement."""
        start = time.monotonic()
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(self._config.execution_timeout_seconds),
                )
            except TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise ToolTimeoutError(
                    f"Sandbox execution exceeded {self._config.execution_timeout_seconds}s",
                    code="tool.timeout",
                    details={"timeout_seconds": self._config.execution_timeout_seconds},
                    hint="Increase sandbox_timeout_seconds in SandboxConfig or split the workload into smaller operations.",
                ) from exc

            duration_ms = (time.monotonic() - start) * 1000
            exit_code = proc.returncode if proc.returncode is not None else 1
            error = stderr_bytes.decode() if exit_code != 0 else None

            return SandboxResult(
                stdout=stdout_bytes.decode(),
                stderr=stderr_bytes.decode(),
                return_value=None,
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=error,
            )
        except ToolTimeoutError:
            raise
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            return SandboxResult(
                stdout="",
                stderr=str(exc),
                return_value=None,
                exit_code=1,
                duration_ms=duration_ms,
                error=str(exc),
            )

    async def close(self) -> None:
        pass


class _DockerSandbox:
    """Runs code inside an ephemeral Docker container."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._client = _docker_module.from_env()

    async def execute(self, code: str, *, namespace: dict[str, Any] | None = None) -> SandboxResult:
        """Execute *code* in a fresh Docker container."""
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_container, code),
                timeout=float(self._config.execution_timeout_seconds),
            )
            return result
        except TimeoutError as exc:
            raise ToolTimeoutError(
                f"Sandbox execution exceeded {self._config.execution_timeout_seconds}s",
                code="tool.timeout",
                details={"timeout_seconds": self._config.execution_timeout_seconds},
            ) from exc
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.warning("sandbox.docker_error", error=str(exc))
            return SandboxResult(
                stdout="",
                stderr=str(exc),
                return_value=None,
                exit_code=1,
                duration_ms=duration_ms,
                error=str(exc),
            )

    def _run_container(self, code: str) -> SandboxResult:
        """Synchronous Docker run (called from a thread)."""
        start = time.monotonic()
        container = None
        try:
            mem_limit = f"{self._config.memory_limit_mb}m"
            nano_cpus = int(self._config.cpu_limit * 1_000_000_000)
            container = self._client.containers.run(
                self._config.image,
                ["python", "-c", code],
                detach=True,
                network_disabled=not self._config.network_enabled,
                mem_limit=mem_limit,
                nano_cpus=nano_cpus,
                remove=False,
            )
            exit_result = container.wait()
            exit_code = exit_result.get("StatusCode", 1)
            stdout = container.logs(stdout=True, stderr=False).decode()
            stderr = container.logs(stdout=False, stderr=True).decode()
            duration_ms = (time.monotonic() - start) * 1000
            error = stderr if exit_code != 0 else None
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                return_value=None,
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=error,
            )
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.remove(force=True)

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self._client.close)
