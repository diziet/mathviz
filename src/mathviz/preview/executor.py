"""Run pipeline generation with timeout and cancellation support."""

import logging
import os
import threading
import time
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    TimeoutError,
    as_completed,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any

from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.runner import PipelineResult
from mathviz.pipeline.runner import run as run_pipeline

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300
BATCH_TIMEOUT_ERROR = "Batch timed out"
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


@dataclass
class BatchPanelResult:
    """Result for a single panel in a batch generation."""

    index: int
    pipeline_result: PipelineResult | None = None
    error: str | None = None


@dataclass
class BatchResult:
    """Result of a batch generation across multiple panels."""

    panels: list[BatchPanelResult] = field(default_factory=list)
    timed_out: bool = False


class GenerationExecutor:
    """Manages pipeline generation with timeout and cancel support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_task: GenerationTask | None = None
        self._pool: ProcessPoolExecutor | None = None
        self._batch_pool: ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> ProcessPoolExecutor:
        """Create the process pool if needed. Must be called under self._lock."""
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1)
        return self._pool

    def _ensure_batch_pool(self, worker_count: int) -> ProcessPoolExecutor:
        """Create the batch pool if needed. Must be called under self._lock."""
        if self._batch_pool is None:
            self._batch_pool = ProcessPoolExecutor(max_workers=worker_count)
        return self._batch_pool

    def submit(
        self,
        generator: str,
        params: dict[str, Any] | None,
        seed: int,
        resolution_kwargs: dict[str, Any] | None,
        container: Container,
        placement: PlacementPolicy,
        timeout_override: int | None = None,
    ) -> PipelineResult:
        """Run the pipeline with timeout. Raises TimeoutError or CancelledError."""
        timeout = timeout_override if timeout_override and timeout_override > 0 else get_timeout_seconds()

        with self._lock:
            pool = self._ensure_pool()
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

    def submit_batch(
        self,
        jobs: list[dict[str, Any]],
    ) -> BatchResult:
        """Run multiple pipeline jobs in parallel with a shared timeout."""
        timeout = get_timeout_seconds()
        worker_count = min(len(jobs), os.cpu_count() or 1)

        with self._lock:
            pool = self._ensure_batch_pool(worker_count)

        return self._run_batch(pool, jobs, timeout)

    def _run_batch(
        self,
        pool: ProcessPoolExecutor,
        jobs: list[dict[str, Any]],
        timeout: float,
    ) -> BatchResult:
        """Execute batch jobs on pool, collecting results until timeout."""
        future_to_idx: dict[Future, int] = {}
        for i, job in enumerate(jobs):
            future = pool.submit(
                _run_pipeline_in_process,
                job["generator"],
                job.get("params"),
                job.get("seed", 42),
                job.get("resolution_kwargs"),
                job["container"],
                job["placement"],
            )
            future_to_idx[future] = i

        deadline = time.monotonic() + timeout
        result = BatchResult()
        result.panels = [BatchPanelResult(index=i) for i in range(len(jobs))]

        try:
            for future in as_completed(future_to_idx, timeout=timeout):
                idx = future_to_idx[future]
                try:
                    pipeline_result = future.result(timeout=0)
                    result.panels[idx].pipeline_result = pipeline_result
                except (KeyError, ValueError) as exc:
                    result.panels[idx].error = "Generation failed"
                    logger.error("Batch panel %d failed: %s", idx, exc)
                except (BrokenProcessPool, CancelledError) as exc:
                    result.panels[idx].error = "Generation failed"
                    logger.error("Batch panel %d failed: %s", idx, exc)
        except TimeoutError:
            pass  # Some futures didn't complete — handled below

        # Mark any panels that didn't complete as timed out
        for future, idx in future_to_idx.items():
            panel = result.panels[idx]
            if panel.pipeline_result is None and panel.error is None:
                result.timed_out = True
                future.cancel()
                panel.error = BATCH_TIMEOUT_ERROR

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
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None

    def shutdown(self) -> None:
        """Shut down the executor cleanly."""
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
            if self._batch_pool is not None:
                self._batch_pool.shutdown(wait=True)
                self._batch_pool = None
