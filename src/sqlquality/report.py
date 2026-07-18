"""Render a GateReport to a JSON payload and a self-contained HTML document."""

from __future__ import annotations

import html as _html

from sqlquality.gate import GateReport


def gate_payload(report: GateReport, neighbors: list[str]) -> dict:
    """JSON-serializable summary of a gate report."""
    return {
        "passed": report.passed,
        "regressions": report.regressions,
        "neighbors": neighbors,
        "models": [
            {
                "unique_id": d.unique_id,
                "baseline": d.baseline,
                "candidate": d.candidate,
                "delta": d.delta,
                "is_new": d.is_new,
            }
            for d in report.deltas
        ],
    }


def render_html(report: GateReport) -> str:
    """A self-contained HTML report (no external assets)."""
    verdict = "PASS" if report.passed else "FAIL"
    color = "#137333" if report.passed else "#b3261e"
    rows = []
    for d in report.deltas:
        tag = " (new)" if d.is_new else ""
        flag = "⚠️" if d.unique_id in report.regressions else ""
        rows.append(
            "<tr>"
            f"<td>{_html.escape(d.unique_id)}{tag}</td>"
            f"<td>{d.baseline}</td>"
            f"<td>{d.candidate}</td>"
            f"<td>{d.delta:+}</td>"
            f"<td>{flag}</td>"
            "</tr>"
        )
    table_body = "\n".join(rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>sqlquality report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
.banner {{ color: #fff; background: {color}; padding: .6rem 1rem; border-radius: 6px; font-weight: 600; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
th, td {{ border: 1px solid #ddd; padding: .4rem .8rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style></head>
<body>
<div class="banner">sqlquality: {verdict}</div>
<table>
<thead><tr><th>model</th><th>baseline</th><th>candidate</th><th>delta</th><th></th></tr></thead>
<tbody>
{table_body}
</tbody>
</table>
</body></html>
"""
