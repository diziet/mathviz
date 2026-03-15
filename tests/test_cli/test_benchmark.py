"""Tests for the mathviz benchmark CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from mathviz.cli import app

runner = CliRunner()

# Use lightweight generators that are fast to run
FAST_GENERATORS = "lorenz,torus,mobius_strip"


def _run_benchmark(
    tmp_path: Path,
    generators: str = FAST_GENERATORS,
    runs: str = "1",
    workers: str = "1",
) -> tuple[object, Path]:
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


class TestBenchmarkCommand:
    """Test that benchmark command runs and produces correct output."""

    def test_benchmark_runs_without_error(self, tmp_path: Path) -> None:
        """mathviz benchmark runs without error on at least 3 generators."""
        result, _ = _run_benchmark(tmp_path)
        assert result.exit_code == 0, (
            f"Exit code {result.exit_code}: {result.output}"
        )

    def test_html_file_created_with_table(self, tmp_path: Path) -> None:
        """Output HTML file is created and contains a <table> element."""
        result, output = _run_benchmark(tmp_path)
        assert result.exit_code == 0
        assert output.exists()
        html_content = output.read_text(encoding="utf-8")
        assert "<table" in html_content

    def test_generator_rows_have_timing_columns(self, tmp_path: Path) -> None:
        """Each generator row has timing columns for all pipeline stages."""
        result, output = _run_benchmark(tmp_path)
        assert result.exit_code == 0
        html_content = output.read_text(encoding="utf-8")
        for name in FAST_GENERATORS.split(","):
            assert name in html_content
        for stage in ["Generate", "Represent", "Transform", "Validate", "Total"]:
            assert stage in html_content

    def test_generators_flag_limits_selection(self, tmp_path: Path) -> None:
        """--generators lorenz,torus limits the benchmark to those generators."""
        result, output = _run_benchmark(tmp_path, generators="lorenz,torus")
        assert result.exit_code == 0
        html_content = output.read_text(encoding="utf-8")
        assert "lorenz" in html_content
        assert "torus" in html_content
        assert html_content.count("<tr>") >= 3  # header + 2 data rows

    def test_single_run_produces_results(self, tmp_path: Path) -> None:
        """--runs 1 produces results with a single run per generator."""
        result, output = _run_benchmark(tmp_path, generators="torus")
        assert result.exit_code == 0
        html_content = output.read_text(encoding="utf-8")
        assert "torus" in html_content
        assert "Runs: 1" in html_content

    def test_text_summary_printed_to_stdout(self, tmp_path: Path) -> None:
        """Text summary is printed to stdout."""
        result, _ = _run_benchmark(tmp_path, generators="torus")
        assert result.exit_code == 0
        assert "Benchmark" in result.output
        assert "torus" in result.output


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
