"""Run pipeline generation with timeout and cancellation support."""

import logging
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any

from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.runner import PipelineResult
from mathviz.pipeline.runner import run as run_pipeline

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300
_ENV_VAR = "MATHVIZ_GENERATION_TIMEOUT"


def get_timeout_seconds() -> int:
    """Return the configured generation timeout in seconds."""
    raw = os.environ.get(_ENV_VAR, "")
    if raw.strip():
        try:
            value = int(raw)
            if value > 0:
                return value
            logger.warning("%s must be positive, using default %d", _ENV_VAR, DEFAULT_TIMEOUT_SECONDS)
        except ValueError:
            logger.warning("Invalid %s=%r, using default %d", _ENV_VAR, raw, DEFAULT_TIMEOUT_SECONDS)
    return DEFAULT_TIMEOUT_SECONDS


def _run_pipeline_in_process(
    generator: str,
    params: dict[str, Any] | None,
    seed: int,
    resolution_kwargs: dict[str, Any] | None,
    container: Container,
    placement: PlacementPolicy,
) -> PipelineResult:
    """Target function executed in the subprocess."""
    return run_pipeline(
        generator,
        params=params,
        seed=seed,
        resolution_kwargs=resolution_kwargs,
        container=container,
        placement=placement,
    )


@dataclass
class GenerationTask:
    """Tracks a running generation for timeout and cancellation."""

    future: Future
    cancelled: bool = False


class GenerationExecutor:
    """Manages pipeline generation with timeout and cancel support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_task: GenerationTask | None = None
        self._pool: ProcessPoolExecutor | None = None

    def _get_pool(self) -> ProcessPoolExecutor:
        """Lazily create the process pool."""
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1)
        return self._pool

    def submit(
        self,
        generator: str,
        params: dict[str, Any] | None,
        seed: int,
        resolution_kwargs: dict[str, Any] | None,
        container: Container,
        placement: PlacementPolicy,
    ) -> PipelineResult:
        """Run the pipeline with timeout. Raises TimeoutError or CancelledError."""
        timeout = get_timeout_seconds()
        pool = self._get_pool()

        future = pool.submit(
            _run_pipeline_in_process,
            generator,
            params,
            seed,
            resolution_kwargs,
            container,
            placement,
        )

        task = GenerationTask(future=future)
        with self._lock:
            self._current_task = task

        try:
            result = future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            self._terminate_pool()
            raise
        finally:
            with self._lock:
                self._current_task = None

        return result

    def cancel(self) -> bool:
        """Cancel the running generation. Returns True if cancelled."""
        with self._lock:
            task = self._current_task
            if task is None:
                return False
            task.cancelled = True
            task.future.cancel()

        self._terminate_pool()
        return True

    def is_running(self) -> bool:
        """Return True if a generation is currently running."""
        with self._lock:
            return self._current_task is not None

    def _terminate_pool(self) -> None:
        """Kill the pool and recreate it to ensure the process is dead."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def shutdown(self) -> None:
        """Shut down the executor cleanly."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
