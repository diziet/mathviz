"""HTML report generation for pipeline benchmarks."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mathviz.cli_benchmark import BenchmarkResult, BenchmarkSuite

PIPELINE_STAGES = ["generate", "represent", "transform", "validate"]


def generate_html_report(suite: BenchmarkSuite) -> str:
    """Generate a self-contained HTML benchmark report."""
    rows_html = _build_table_rows(suite.results)
    bar_chart_html = _build_bar_chart(suite.results)

    fastest, slowest = _find_extremes(suite.results)

    return _HTML_TEMPLATE.format(
        cpu=html.escape(suite.cpu),
        python_version=html.escape(suite.python_version),
        date=html.escape(suite.date),
        worker_count=suite.worker_count,
        runs_per_generator=suite.runs_per_generator,
        total_generators=len(suite.results),
        fastest=html.escape(fastest),
        slowest=html.escape(slowest),
        table_rows=rows_html,
        bar_chart=bar_chart_html,
    )


def _color_class(seconds: float) -> str:
    """Return CSS class name based on timing threshold."""
    ms = seconds * 1000
    if ms < 100:
        return "fast"
    if ms < 1000:
        return "medium"
    return "slow"


def _format_ms(seconds: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{seconds * 1000:.1f}ms"


def _build_table_rows(results: list[BenchmarkResult]) -> str:
    """Build HTML table rows for all benchmark results."""
    sorted_results = sorted(results, key=lambda r: r.total_time)
    rows: list[str] = []

    for result in sorted_results:
        if result.error:
            error_msg = html.escape(result.error)
            cells = "".join(
                f'<td class="slow">—</td>' for _ in PIPELINE_STAGES
            )
            rows.append(
                f"<tr><td>{html.escape(result.generator_name)}</td>"
                f'{cells}<td class="slow">—</td>'
                f'<td class="slow">{error_msg}</td></tr>'
            )
            continue

        cells: list[str] = []
        for stage in PIPELINE_STAGES:
            val = result.stage_timings.get(stage, 0.0)
            css = _color_class(val)
            cells.append(f'<td class="{css}">{_format_ms(val)}</td>')

        total_css = _color_class(result.total_time)
        cells.append(f'<td class="{total_css}">{_format_ms(result.total_time)}</td>')
        cells.append("<td>OK</td>")

        name = html.escape(result.generator_name)
        rows.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")

    return "\n".join(rows)


def _build_bar_chart(results: list[BenchmarkResult]) -> str:
    """Build CSS-only horizontal bar chart for each generator."""
    successful = [r for r in results if r.error is None]
    if not successful:
        return "<p>No successful benchmarks to chart.</p>"

    max_total = max(r.total_time for r in successful) or 1.0
    bars: list[str] = []

    stage_colors = {
        "generate": "#4caf50",
        "represent": "#2196f3",
        "transform": "#ff9800",
        "validate": "#9c27b0",
    }

    for result in sorted(successful, key=lambda r: r.total_time):
        segments: list[str] = []
        for stage in PIPELINE_STAGES:
            val = result.stage_timings.get(stage, 0.0)
            pct = (val / max_total) * 100
            color = stage_colors.get(stage, "#999")
            if pct > 0.5:
                segments.append(
                    f'<div class="bar-seg" style="width:{pct:.1f}%;'
                    f'background:{color}" title="{stage}: {_format_ms(val)}">'
                    f"</div>"
                )

        name = html.escape(result.generator_name)
        bars.append(
            f'<div class="bar-row">'
            f'<span class="bar-label">{name}</span>'
            f'<div class="bar-track">{"".join(segments)}</div>'
            f'<span class="bar-total">{_format_ms(result.total_time)}</span>'
            f"</div>"
        )

    legend_items = "".join(
        f'<span class="legend-item">'
        f'<span class="legend-swatch" style="background:{color}"></span>'
        f"{stage}</span>"
        for stage, color in stage_colors.items()
    )

    return f'<div class="legend">{legend_items}</div>\n' + "\n".join(bars)


def _find_extremes(results: list[BenchmarkResult]) -> tuple[str, str]:
    """Find fastest and slowest generator names."""
    successful = [r for r in results if r.error is None]
    if not successful:
        return ("N/A", "N/A")
    fastest = min(successful, key=lambda r: r.total_time)
    slowest = max(successful, key=lambda r: r.total_time)
    return (fastest.generator_name, slowest.generator_name)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MathViz Pipeline Benchmark</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; }}
h1 {{ color: #333; }}
.info {{ background: #e3f2fd; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; }}
.info span {{ margin-right: 2rem; }}
.highlight {{ font-weight: bold; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #f5f5f5; cursor: pointer; user-select: none; }}
th:first-child, td:first-child {{ text-align: left; }}
.fast {{ background: #c8e6c9; }}
.medium {{ background: #fff9c4; }}
.slow {{ background: #ffcdd2; }}
.bar-row {{ display: flex; align-items: center; margin: 4px 0; }}
.bar-label {{ width: 180px; text-align: right; padding-right: 8px; font-size: 0.85rem; }}
.bar-track {{ display: flex; flex: 1; height: 20px; background: #eee; border-radius: 3px; overflow: hidden; }}
.bar-seg {{ height: 100%; }}
.bar-total {{ width: 80px; padding-left: 8px; font-size: 0.85rem; }}
.legend {{ margin-bottom: 0.5rem; }}
.legend-item {{ margin-right: 1rem; font-size: 0.85rem; }}
.legend-swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: middle; border-radius: 2px; }}
</style>
</head>
<body>
<h1>MathViz Pipeline Benchmark</h1>
<div class="info">
<span>CPU: {cpu}</span>
<span>Python: {python_version}</span>
<span>Date: {date}</span>
<span>Workers: {worker_count}</span>
<span>Runs: {runs_per_generator}</span>
<span>Generators: {total_generators}</span>
</div>
<p>Fastest: <span class="highlight">{fastest}</span> &mdash;
Slowest: <span class="highlight">{slowest}</span></p>
<table id="results">
<thead>
<tr>
<th onclick="sortTable(0)">Generator</th>
<th onclick="sortTable(1)">Generate</th>
<th onclick="sortTable(2)">Represent</th>
<th onclick="sortTable(3)">Transform</th>
<th onclick="sortTable(4)">Validate</th>
<th onclick="sortTable(5)">Total</th>
<th onclick="sortTable(6)">Status</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
<h2>Per-Stage Breakdown</h2>
{bar_chart}
<script>
function sortTable(col) {{
  var table = document.getElementById("results");
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.rows);
  var asc = table.dataset.sortCol == col && table.dataset.sortDir == "asc";
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? "desc" : "asc";
  rows.sort(function(a, b) {{
    var av = a.cells[col].textContent.trim();
    var bv = b.cells[col].textContent.trim();
    var an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? bn - an : an - bn;
    return asc ? bv.localeCompare(av) : av.localeCompare(bv);
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
}}
</script>
</body>
</html>
"""
