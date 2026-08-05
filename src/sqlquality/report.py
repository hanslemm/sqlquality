"""Render a GateReport to a JSON payload and a self-contained HTML document."""

from __future__ import annotations

import html as _html

from sqlquality.gate import GateReport
from sqlquality.models import (
    Aggregation,
    Proposal,
    Workload,
    analyzed_query_groups,
    cost_share_of,
)
from sqlquality.workload.fingerprint import fingerprint_id


def _md_escape(value: object) -> str:
    """Neutralize a value for a markdown table cell / inline text.

    Escapes `|` (table cell breakout) and backticks, HTML-escapes `<>&` so a
    hostile unique_id or skip reason cannot inject markup or fake columns, and
    collapses embedded newlines so a multi-line identifier (a real threat: see
    the DDL-rendering `_comment_lines` precedent in workload/postgres.py)
    cannot split a table row into a bogus second row or truncate a heading.
    """
    text = str(value)
    text = text.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("|", "\\|").replace("`", "\\`")


def verdict_label(report: GateReport, *, emoji: bool) -> str:
    """Human verdict string. `emoji` toggles the decorated vs plain variant."""
    if report.warned:
        head = "⚠️ WARN" if emoji else "WARN"
        n = len(report.regressions)
        noun = "regression" if n == 1 else "regressions"
        return f"{head} ({n} {noun}, gate mode: {report.mode})"
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


def advise_payload(
    proposals: list[Proposal],
    workload: Workload,
    aggregation: Aggregation,
    *,
    engine: str,
    redacted: bool,
    degraded: list[tuple[str, str]],
    dbt: dict | None = None,
    window_facts: dict[str, object] | None = None,
    physical_state: dict[str, dict[str, object]] | None = None,
) -> dict:
    """JSON-serializable summary of an advise run.

    The `"dbt"` key is *omitted entirely* when `dbt` is `None` (the default, and what every
    existing caller before this key existed still gets) rather than present with a `None`
    value: the dbt-free path is first-class, and this is what lets a no-manifest `advise`
    invocation stay byte-identical to the payload from before dbt enrichment existed, not
    merely equal apart from one known extra key. A consumer wanting the manifest count
    unconditionally can still do `payload.get("dbt")`, which behaves the same either way.
    When a manifest did load, the caller (`cli.advise`) is responsible for handing in a
    plain, JSON-serializable dict (a `Path` or a `Relation` is not — `cli.advise` already
    stringifies the manifest path before building this dict), since this function does not
    itself normalize it the way `_jsonable` normalizes proposal evidence.

    `window_facts` is the adapter's own `WorkloadAdapter.window_facts()` — `{}` (the
    default) when the caller has none, exactly like every existing caller before this
    parameter existed. `"window"` is an object rather than the bare string it used to be:
    a prose sentence cannot be compared across two runs, which is what `verify` needs to
    do. Each fact is present-but-null rather than absent when unknown — see the comment
    below — because an *absent* key is how `verify` tells a pre-this-feature artifact
    apart from one that genuinely has nothing to report for a field.

    `physical_state` is the adapter's own `WorkloadAdapter.physical_state(relations)`,
    called by the caller (`cli.advise`) with `models.proposal_relations(proposals)` so its
    size scales with findings rather than with schema size. Unlike `dbt`, this key is
    **always present** — `{}` (the default) when the caller has none — never omitted:
    `verify` (a later task) treats an *absent* `physical_state` key as an artifact from a
    version that predates this feature and refuses it outright, which is a materially
    different response than "this run found nothing physical to report." Blurring that
    distinction by omitting the key on an empty result would make a pre-this-feature
    artifact and a genuinely-empty one indistinguishable to the one caller that needs to
    tell them apart.

    The top-level `"query_groups"` key is a **different key from, and coexists with**,
    `payload["analyzed"]["query_groups"]` above — that one is an integer count (how many
    query groups this run understood), this one is a `list[dict]` of the specific query
    groups a proposal actually cites in its `evidence["fingerprint_digests"]`, each carrying
    `digest`/`calls`/`total_time_ms`/`mean_ms`. Two keys of the same name holding different
    types at different nesting levels of the same payload is unusual enough that a consumer
    should not have to discover it by surprise: JSON nesting disambiguates the two (one
    lives under `"analyzed"`, the other at the payload's root), and renaming the older,
    already-shipped `analyzed.query_groups` to make room would be a breaking payload change
    for existing consumers, which is outside what this feature set is here to do. Like
    `physical_state`, this key is **always present** (`[]` when no proposal cites any query
    group) rather than omitted-when-empty: `verify` (a later task) uses an absent key, not
    an empty list, to recognize an artifact from a version that predates this feature.
    """
    window_facts = dict(window_facts or {})
    payload = {
        "engine": engine,
        "redacted": redacted,
        "window": {
            "description": workload.window_description,
            "engine": engine,
            # Present-but-null rather than absent: `verify` treats an absent key as a
            # payload from a version that predates this feature and refuses the artifact,
            # while null means "this engine cannot tell you", which is comparable
            # information.
            "stats_reset_at": window_facts.get("stats_reset_at"),
            "since": window_facts.get("since"),
            "limit": window_facts.get("limit"),
        },
        "analyzed": {
            # The count of groups whose usage was actually extracted — not `len(stats)`,
            # which includes the unresolvable and ambiguous groups reported under "skipped"
            # below and so contradicted them under a key named "analyzed". The window total
            # is kept beside it rather than dropped, since a consumer computing "how much of
            # the window did this run understand" needs both numbers.
            "query_groups": analyzed_query_groups(workload, aggregation),
            "query_groups_in_window": len(workload.stats),
            "total_cost_ms": workload.total_cost_ms,
            "tables": sorted(str(relation) for relation in aggregation.tables),
        },
        "skipped": {
            "unparseable": workload.skipped_unparseable,
            "noise": workload.skipped_noise,
            "unqualifiable": aggregation.skipped_unqualifiable,
            "ambiguous": aggregation.skipped_ambiguous,
        },
        "degraded": [{"capability": cap, "reason": reason} for cap, reason in degraded],
        "proposals": [
            {
                "code": p.code,
                "title": p.title,
                "rationale": p.rationale,
                "confidence": p.confidence.value,
                # Evidence values include tuples and frozensets; normalize for JSON.
                "evidence": {k: _jsonable(v) for k, v in p.evidence.items()},
                "ddl": p.ddl,
            }
            for p in proposals
        ],
        # Always present, unlike "dbt" below — see this function's docstring for why an
        # absent key here must mean something different (a pre-this-feature artifact) than
        # an empty one (this run had nothing physical to report).
        "physical_state": dict(physical_state or {}),
        # Always present too, for the identical reason — see this function's docstring for
        # why this coexists with (and is a different key from) "analyzed"."query_groups".
        "query_groups": _query_groups_payload(proposals, workload),
    }
    if dbt is not None:
        payload["dbt"] = dbt
    return payload


def _query_groups_payload(proposals: list[Proposal], workload: Workload) -> list[dict]:
    """The workload query groups actually cited by some proposal's `fingerprint_digests`.

    Every proposal that names a query group does so with a digest (see
    `workload.fingerprint.fingerprint_id`), never the group's full canonical SQL — so this
    is the one place a digest is resolved back to its timings, and it does so by recomputing
    the same digest from each `QueryStat.fingerprint` and matching on that, rather than by
    trusting any string a proposal's evidence happens to carry.

    Scoped to *referenced* groups, not every group in `workload.stats`: a query group this
    run only glanced at, without any rule finding it worth proposing on, is not evidence for
    anything and would only bloat the payload. `verify` (a later task) needs exactly the
    groups that back a proposal, to compare their timings against a later run's — not the
    whole workload.

    `mean_ms` is `None`, never `0.0`, for a group with zero recorded calls: a group that was
    never called has no meaningful mean latency to report, and `0.0` reads as "instant",
    the opposite of "unknown". This mirrors the present-but-null idiom `window` and
    `physical_state` already use elsewhere in this payload for "this run could not tell
    you", as against a real measurement of zero.

    Order follows `workload.stats` (cost descending, then fingerprint — see `ingest()`),
    not a fresh sort by digest: that order is already deterministic run-to-run for the same
    workload, and re-sorting by digest would throw away the cost ordering for no gain.
    """
    referenced: set[str] = set()
    for proposal in proposals:
        digests = proposal.evidence.get("fingerprint_digests")
        if isinstance(digests, (tuple, list)):
            referenced.update(str(d) for d in digests)
    groups = []
    for stat in workload.stats:
        digest = fingerprint_id(stat.fingerprint)
        if digest not in referenced:
            continue
        groups.append(
            {
                "digest": digest,
                "calls": stat.calls,
                "total_time_ms": stat.total_time_ms,
                "mean_ms": (stat.total_time_ms / stat.calls) if stat.calls else None,
            }
        )
    return groups


def _jsonable(value: object) -> object:
    """Coerce evidence values (tuples, sets) into JSON-friendly types."""
    if isinstance(value, (tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def render_advise_markdown(
    proposals: list[Proposal],
    workload: Workload,
    aggregation: Aggregation,
    *,
    engine: str,
    redacted: bool,
    degraded: list[tuple[str, str]],
    dbt: dict | None = None,
) -> str:
    """Render advise proposals as markdown (suitable for a ticket or PR comment).

    `dbt` defaults to `None` — every existing caller omits it — so a no-manifest run
    renders exactly the markdown it always has; the section below only appears when a
    manifest actually loaded.
    """
    lines = [
        f"# sqlquality advise — {_md_escape(engine)}",
        "",
        f"**Window:** {_md_escape(workload.window_description)}",
        # "N of M", the same form and the same numbers as the terminal's coverage line.
        # Printing `len(workload.stats)` alone as *analyzed* contradicted the "Skipped:"
        # line directly beneath it — "8 analyzed" above "2 ambiguous" out of 8 groups.
        f"**Query groups analyzed:** {analyzed_query_groups(workload, aggregation)} of "
        f"{len(workload.stats)}  ",
        f"**Literals:** {'redacted' if redacted else 'retained (--keep-literals)'}",
        "",
        (
            # "filtered", not "introspection/DDL": it covers session control, DDL,
            # maintenance, introspection, a whole-table COPY, and a cursor statement
            # carrying no query (FETCH, CLOSE). DECLARE ... CURSOR FOR SELECT and
            # COPY (SELECT ...) TO — ordinary reads — are unwrapped to their inner query
            # before this count is taken, so they land in "analyzed" instead.
            # See cli._coverage_line and workload.fingerprint.unwrap.
            f"Skipped: {workload.skipped_unparseable} unparseable, "
            f"{workload.skipped_noise} filtered as non-workload (session control, DDL, "
            f"maintenance, introspection, whole-table COPY, or a cursor statement "
            f"carrying no query), "
            f"{aggregation.skipped_unqualifiable} unresolvable against the schema, "
            f"{aggregation.skipped_ambiguous} ambiguous across the introspected schemas."
        ),
        "",
    ]
    if degraded:
        lines.append("## Reduced coverage")
        lines.append("")
        for capability, reason in degraded:
            lines.append(f"- `{_md_escape(capability)}`: {_md_escape(reason)}")
        lines.append("")

    if dbt is not None:
        lines.append("## dbt enrichment")
        lines.append("")
        lines.append(f"- manifest: `{_md_escape(dbt['manifest'])}`")
        lines.append(f"- models indexed: {dbt['models']}")
        if dbt.get("dropped_collisions"):
            lines.append(f"- cross-database collisions dropped: {dbt['dropped_collisions']}")
        lines.append("")

    if not proposals:
        lines.append("No proposals — nothing in the analyzed workload met the thresholds.")
        return "\n".join(lines) + "\n"

    lines += [
        "| code | confidence | cost share | proposal |",
        "|---|---|---:|---|",
    ]
    for p in proposals:
        share = cost_share_of(p.evidence)
        share_text = f"{share:.1%}" if share is not None else "—"
        lines.append(
            f"| {_md_escape(p.code)} | {p.confidence.value} | {share_text} "
            f"| {_md_escape(p.title)} |"
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for p in proposals:
        lines.append(f"### {_md_escape(p.code)} — {_md_escape(p.title)}")
        lines.append("")
        lines.append(_md_escape(p.rationale))
        lines.append("")
        # Keys are escaped too. They are a closed static vocabulary today, but the
        # asymmetry is the kind that stops being true quietly.
        #
        # Values go through `_jsonable` first so the markdown reader and the JSON reader
        # see the same shape. Without it `str(("status",))` renders the Python repr
        # `('status',)` where JSON shows `["status"]` — the same run described two ways.
        evidence = ", ".join(
            f"{_md_escape(k)}={_md_escape(_jsonable(v))}" for k, v in sorted(p.evidence.items())
        )
        lines.append(f"Evidence: {evidence}")
        lines.append("")
        if p.ddl:
            fence = _code_fence(p.ddl)
            lines.append(f"{fence}sql")
            lines.append(p.ddl)
            lines.append(fence)
            lines.append("")
    return "\n".join(lines) + "\n"


def _code_fence(text: str) -> str:
    """A backtick fence guaranteed longer than any backtick run inside `text`.

    `ddl` is generated from live schema identifiers (see `_quote_ident` in
    workload/postgres.py, which does not forbid backticks). A fixed ```` ```` ````
    fence would let an identifier containing a triple-backtick close the fence early
    and inject arbitrary markdown — a fake heading, a fake second code block — after
    it. Sizing the fence past the longest run in the content is the standard
    CommonMark technique for nesting a fence around backtick-bearing content.
    """
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return "`" * max(3, longest + 1)
