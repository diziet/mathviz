"""PipelineTimer: context-manager utility for per-stage wall-clock timing."""

import logging
import time
from types import TracebackType

logger = logging.getLogger(__name__)


class PipelineTimer:
    """Accumulates wall-clock timing for named pipeline stages."""

    def __init__(self) -> None:
        """Initialize with empty timings."""
        self._timings: dict[str, float] = {}
        self._current_stage: str | None = None
        self._start_time: float = 0.0

    @property
    def timings(self) -> dict[str, float]:
        """Return a copy of the accumulated stage timings."""
        return dict(self._timings)

    def stage(self, name: str) -> "_StageContext":
        """Return a context manager that times a named stage."""
        return _StageContext(self, name)

    def _begin(self, name: str) -> None:
        """Record the start of a stage."""
        if name in self._timings:
            logger.warning("Stage %r already timed; overwriting previous timing", name)
        self._current_stage = name
        self._start_time = time.monotonic()

    def _end(self) -> None:
        """Record the end of the current stage."""
        if self._current_stage is None:
            return
        elapsed = time.monotonic() - self._start_time
        self._timings[self._current_stage] = elapsed
        logger.debug("Stage %s took %.4fs", self._current_stage, elapsed)
        self._current_stage = None


class _StageContext:
    """Context manager for a single pipeline stage timing."""

    def __init__(self, timer: PipelineTimer, name: str) -> None:
        self._timer = timer
        self._name = name

    def __enter__(self) -> "_StageContext":
        self._timer._begin(self._name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._timer._end()
