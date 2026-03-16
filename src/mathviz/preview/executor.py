"""Run pipeline generation with timeout and cancellation support."""

import logging
import os
import threading
import time
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
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
MAX_TIMEOUT_SECONDS = 3600
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


def _run_pipeline_in_thread(
    generator: str,
    params: dict[str, Any] | None,
    seed: int,
    resolution_kwargs: dict[str, Any] | None,
    container: Container,
    placement: PlacementPolicy,
    cancel_event: threading.Event,
) -> PipelineResult:
    """Target function executed in the thread pool."""
    return run_pipeline(
        generator,
        params=params,
        seed=seed,
        resolution_kwargs=resolution_kwargs,
        container=container,
        placement=placement,
        cancel_event=cancel_event,
    )


def _run_pipeline_in_process(
    generator: str,
    params: dict[str, Any] | None,
    seed: int,
    resolution_kwargs: dict[str, Any] | None,
    container: Container,
    placement: PlacementPolicy,
) -> PipelineResult:
    """Target function executed in the subprocess (batch only)."""
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
    cancel_event: threading.Event
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


_CANCEL_GRACE_SECONDS = 2


class GenerationExecutor:
    """Manages pipeline generation with timeout and cancel support.

    Single generations run in a ThreadPoolExecutor to avoid subprocess
    pickle/import overhead. The numba JIT kernel releases the GIL, so
    CPU-bound compute does not block the event loop. Cancellation is
    cooperative via a threading.Event checked between pipeline stages.
    If a thread does not exit within the grace period after cancellation,
    the pool is recreated as a last resort.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_task: GenerationTask | None = None
        self._thread_pool: ThreadPoolExecutor | None = None
        self._batch_pool: ProcessPoolExecutor | None = None

    def _ensure_thread_pool(self) -> ThreadPoolExecutor:
        """Create the thread pool if needed. Must be called under self._lock."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=1)
        return self._thread_pool

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
        timeout = timeout_override if timeout_override is not None and timeout_override > 0 else get_timeout_seconds()
        cancel_event = threading.Event()

        with self._lock:
            pool = self._ensure_thread_pool()
            future = pool.submit(
                _run_pipeline_in_thread,
                generator,
                params,
                seed,
                resolution_kwargs,
                container,
                placement,
                cancel_event,
            )
            task = GenerationTask(future=future, cancel_event=cancel_event)
            self._current_task = task

        try:
            result = future.result(timeout=timeout)
        except TimeoutError:
            cancel_event.set()
            future.cancel()
            self._wait_or_recreate_pool(future)
            raise
        except CancelledError:
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

    def _wait_or_recreate_pool(self, future: Future) -> None:
        """Wait briefly for the thread to exit; recreate the pool if it doesn't."""
        try:
            future.result(timeout=_CANCEL_GRACE_SECONDS)
        except (TimeoutError, CancelledError, KeyError, ValueError):
            pass
        except Exception:
            pass

        if not future.done():
            logger.warning(
                "Generation thread did not exit within %ds grace period; "
                "recreating thread pool",
                _CANCEL_GRACE_SECONDS,
            )
            with self._lock:
                if self._thread_pool is not None:
                    self._thread_pool.shutdown(wait=False, cancel_futures=True)
                    self._thread_pool = None

    def cancel(self) -> bool:
        """Cancel the running generation. Returns True if cancelled."""
        with self._lock:
            task = self._current_task
            if task is None:
                return False
            task.cancelled = True
            task.cancel_event.set()
            task.future.cancel()

        self._wait_or_recreate_pool(task.future)
        return True

    def is_running(self) -> bool:
        """Return True if a generation is currently running."""
        with self._lock:
            return self._current_task is not None

    def shutdown(self) -> None:
        """Shut down the executor cleanly."""
        with self._lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
            if self._batch_pool is not None:
                self._batch_pool.shutdown(wait=True)
                self._batch_pool = None
