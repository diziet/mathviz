"""Shared test constants for generator tests."""

# Reduced from 5000 to speed up test suite; sufficient for verifying
# trajectory properties (finite, non-NaN, non-degenerate, deterministic).
# Must exceed DEFAULT_TRANSIENT_STEPS (1000) by at least 10.
TEST_STEPS_FAST = 1050

# Explicit transient for tests that override the default transient_steps.
TEST_TRANSIENT_FAST = 200
