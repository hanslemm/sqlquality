"""Render a GateReport to a JSON payload and a self-contained HTML document."""

from __future__ import annotations

import html as _html

from sqlquality.gate import GateReport


def _md_escape(value: object) -> str:
    """Neutralize a value for a markdown table cell / inline text.

    Escapes `|` (table cell breakout) and backticks, and HTML-escapes `<>&`
    so a hostile unique_id or skip reason cannot inject markup or fake columns.
    """
    text = str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("|", "\\|").replace("`", "\\`")


def verdict_label(report: GateReport, *, emoji: bool) -> str:
    """Human verdict string. `emoji` toggles the decorated vs plain variant."""
    if report.warned:
        head = "⚠️ WARN" if emoji else "WARN"
        return f"{head} ({len(report.regressions)} regressions, gate mode: {report.mode})"
    if report.passed:
        return "✅ PASS" if emoji else "PASS"
    return "❌ FAIL" if emoji else "FAIL"


def gate_payload(
    report: GateReport, neighbors: list[str], skipped: list[tuple[str, str]] | None = None
) -> dict:
    """JSON-serializable summary of a gate report."""
    return {
        "passed": report.passed,
        "mode": report.mode,
        "warned": report.warned,
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
        "skipped": [{"unique_id": uid, "reason": reason} for uid, reason in (skipped or [])],
    }


def render_markdown(report: GateReport, skipped: list[tuple[str, str]] | None = None) -> str:
    """Render a gate report as markdown (suitable for a PR comment)."""
    lines = [
        f"# sqlquality: {verdict_label(report, emoji=True)}",
        "",
        "| model | baseline | candidate | delta | |",
        "|---|---:|---:|---:|:--:|",
    ]
    for d in report.deltas:
        flag = "⚠️" if d.unique_id in report.regressions else ("🆕" if d.is_new else "")
        lines.append(
            f"| {_md_escape(d.unique_id)} | {d.baseline} | {d.candidate} | {d.delta:+} | {flag} |"
        )
    for uid, reason in skipped or []:
        if len(lines) and not lines[-1].startswith("_skipped_"):
            lines.append("")
        lines.append(f"_skipped_ `{_md_escape(uid)}`: {_md_escape(reason)}")
    return "\n".join(lines) + "\n"


def render_html(report: GateReport, skipped: list[tuple[str, str]] | None = None) -> str:
    """A self-contained HTML report (no external assets)."""
    verdict = verdict_label(report, emoji=False)
    if report.warned:
        color = "#a15c00"  # amber: passed, but regressions slipped through warn mode
    elif report.passed:
        color = "#137333"
    else:
        color = "#b3261e"
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
    skipped_rows = "\n".join(
        f"<li>{_html.escape(uid)}: {_html.escape(reason)}</li>" for uid, reason in (skipped or [])
    )
    skipped_html = f"<h3>Skipped</h3>\n<ul>\n{skipped_rows}\n</ul>" if skipped else ""
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
<div class="banner">sqlquality: {_html.escape(verdict)}</div>
<table>
<thead><tr><th>model</th><th>baseline</th><th>candidate</th><th>delta</th><th></th></tr></thead>
<tbody>
{table_body}
</tbody>
</table>
{skipped_html}
</body></html>
"""
