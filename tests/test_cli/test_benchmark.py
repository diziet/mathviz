"""Tests for the mathviz benchmark CLI command."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from mathviz.cli import app

runner = CliRunner()

# Use two lightweight generators — enough to test multi-generator behavior
FAST_GENERATORS = "lorenz,torus"


@dataclass
class BenchmarkResult:
    """Holds shared benchmark run output."""

    result: Result
    output_path: Path
    html_content: str


def _run_benchmark(
    tmp_path: Path,
    generators: str = FAST_GENERATORS,
    runs: str = "1",
    workers: str = "1",
) -> tuple[Result, Path]:
    """Run benchmark command and return (result, output_path)."""
    output = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--generators", generators,
            "--runs", runs,
            "--workers", workers,
            "--output", str(output),
        ],
    )
    return result, output


def _build_benchmark(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    **kwargs: str,
) -> BenchmarkResult:
    """Run benchmark once and wrap in BenchmarkResult."""
    tmp_path = tmp_path_factory.mktemp(name)
    result, output = _run_benchmark(tmp_path, **kwargs)
    html_content = output.read_text(encoding="utf-8") if output.exists() else ""
    return BenchmarkResult(result=result, output_path=output, html_content=html_content)


@pytest.fixture(scope="class")
def shared_benchmark(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkResult:
    """Run the default benchmark once and share across TestBenchmarkCommand."""
    return _build_benchmark(tmp_path_factory, "benchmark")


class TestBenchmarkCommand:
    """Test that benchmark command runs and produces correct output."""

    def test_benchmark_runs_without_error(
        self, shared_benchmark: BenchmarkResult,
    ) -> None:
        """mathviz benchmark runs without error on at least 2 generators."""
        assert shared_benchmark.result.exit_code == 0, (
            f"Exit code {shared_benchmark.result.exit_code}: "
            f"{shared_benchmark.result.output}"
        )

    def test_html_file_created_with_table(
        self, shared_benchmark: BenchmarkResult,
    ) -> None:
        """Output HTML file is created and contains a <table> element."""
        assert shared_benchmark.result.exit_code == 0
        assert shared_benchmark.output_path.exists()
        assert "<table" in shared_benchmark.html_content

    def test_generator_rows_have_timing_columns(
        self, shared_benchmark: BenchmarkResult,
    ) -> None:
        """Each generator row has timing columns for all pipeline stages."""
        assert shared_benchmark.result.exit_code == 0
        for name in FAST_GENERATORS.split(","):
            assert name in shared_benchmark.html_content
        for stage in ["Generate", "Represent", "Transform", "Validate", "Total"]:
            assert stage in shared_benchmark.html_content

    def test_generators_flag_limits_selection(
        self, shared_benchmark: BenchmarkResult,
    ) -> None:
        """--generators flag limits the benchmark to selected generators only."""
        assert shared_benchmark.result.exit_code == 0
        html = shared_benchmark.html_content
        assert "lorenz" in html
        assert "torus" in html
        # Exactly 2 generators requested — header + 2 data rows minimum
        assert html.count("<tr>") >= 3
        # Generators NOT in FAST_GENERATORS should be absent
        assert "mobius_strip" not in html
        assert "klein_bottle" not in html

    def test_single_run_produces_results(
        self, shared_benchmark: BenchmarkResult,
    ) -> None:
        """--runs 1 produces results with a single run per generator."""
        assert shared_benchmark.result.exit_code == 0
        assert "torus" in shared_benchmark.html_content
        assert "Runs: 1" in shared_benchmark.html_content

    def test_text_summary_printed_to_stdout(
        self, shared_benchmark: BenchmarkResult,
    ) -> None:
        """Text summary is printed to stdout."""
        assert shared_benchmark.result.exit_code == 0
        assert "Benchmark" in shared_benchmark.result.output
        assert "torus" in shared_benchmark.result.output


class TestBenchmarkErrorHandling:
    """Test error handling in benchmark command."""

    def test_failed_generator_in_report_not_crash(
        self, tmp_path: Path,
    ) -> None:
        """Failed generators appear in the report with error messages."""
        result, output = _run_benchmark(
            tmp_path, generators="torus,nonexistent_generator_xyz",
        )
        assert result.exit_code == 0
        html_content = output.read_text(encoding="utf-8")
        assert "nonexistent_generator_xyz" in html_content
        assert "torus" in html_content


class TestBenchmarkTimings:
    """Test timing accuracy in benchmark results."""

    def test_stage_timings_sum_to_total(self) -> None:
        """Per-stage timings sum approximately to total time."""
        from mathviz.cli_benchmark import _run_single_generator

        result = _run_single_generator("torus", 1)
        assert result.error is None

        stage_sum = sum(result.stage_timings.values())
        assert stage_sum <= result.total_time * 1.1, (
            f"Stage sum {stage_sum:.4f} exceeds "
            f"total {result.total_time:.4f} by >10%"
        )
        assert stage_sum >= result.total_time * 0.5, (
            f"Stage sum {stage_sum:.4f} is less than "
            f"50% of total {result.total_time:.4f}"
        )
