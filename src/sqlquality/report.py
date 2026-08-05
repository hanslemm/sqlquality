"""Render a GateReport to a JSON payload and a self-contained HTML document."""

from __future__ import annotations

import html as _html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from sqlquality.gate import GateReport
from sqlquality.models import (
    Aggregation,
    Confidence,
    Proposal,
    Workload,
    analyzed_query_groups,
    cost_share_of,
)
from sqlquality.verify import (
    ProposalIndex,
    ProposalVerdict,
    VerifyOutcome,
    WindowLimits,
    WindowRelation,
    classify_windows,
    confidence_for,
    index_proposals,
    window_limits,
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

    `"since_duration_seconds"` is the *requested* `--since` duration (e.g. `7d`'s
    `604800.0`), distinct from `"since"`'s *absolute cutoff*: `sqlquality verify` grades
    two windows comparable on equal durations, not equal cutoffs, since two runs a week
    apart with the identical `--since 7d` bind two different absolute cutoffs but request
    the same duration — see `sqlquality.verify.classify_windows`'s docstring. Postgres
    reports it as `None` unconditionally, for the same reason it reports `"since"` as
    `None`: it cannot apply `--since` at all, so even the bare duration must not be echoed
    back as though a filter had been applied.

    `physical_state` is the adapter's own `WorkloadAdapter.physical_state(relations)`,
    called by the caller (`cli.advise`) with `models.proposal_relations(proposals)`
    *unioned with* `aggregation.tables` — every relation some proposal targets, plus
    every relation the run's workload analysis actually touched — not the first set
    alone: a relation whose proposal got resolved between two `advise` runs (the
    recommended index now exists, so the rule stops firing) would otherwise have no
    `physical_state` entry at all in the very run `sqlquality verify` needs it in, to
    confirm the fix landed. Sizing against `aggregation.tables` rather than the whole
    introspected schema keeps the original intent (payload size scales with the workload,
    not with schema size) while closing that blind spot. Unlike `dbt`, this key is
    **always present** — `{}` (the default) when the caller has none — never omitted:
    `verify` (a later task) treats an *absent* `physical_state` key as an artifact from a
    version that predates this feature and refuses it outright, which is a materially
    different response than "this run found nothing physical to report." Blurring that
    distinction by omitting the key on an empty result would make a pre-this-feature
    artifact and a genuinely-empty one indistinguishable to the one caller that needs to
    tell them apart.

    The top-level `"query_groups"` key is a **different key from, and coexists with**,
    `payload["analyzed"]["query_groups"]` above — that one is an integer count (how many
    query groups this run understood), this one is a `list[dict]` of **every** query group
    `workload.stats` carries for this run, each with `digest`/`calls`/`total_time_ms`/
    `mean_ms`. **Not scoped to what some proposal currently cites** — see
    `_query_groups_payload`'s docstring for why a citation-scoped list (this key's original
    shape) reproduced, one payload key over, the exact blind spot the sanctioned
    `physical_state` widening above exists to close: a query group whose proposal got
    resolved between two runs (so the later run no longer cites it) is precisely the case
    `sqlquality verify` needs this list *for*. Two keys of the same name holding different
    types at different nesting levels of the same payload is unusual enough that a consumer
    should not have to discover it by surprise: JSON nesting disambiguates the two (one
    lives under `"analyzed"`, the other at the payload's root), and renaming the older,
    already-shipped `analyzed.query_groups` to make room would be a breaking payload change
    for existing consumers, which is outside what this feature set is here to do. Like
    `physical_state`, this key is **always present** (`[]` when the run analysed no query
    groups at all — not when no proposal cites one) rather than omitted-when-empty:
    `verify` (a later task) uses an absent key, not an empty list, to recognize an artifact
    from a version that predates this feature.
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
            "since_duration_seconds": window_facts.get("since_duration_seconds"),
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
        # Scoped to the whole workload, not to `proposals`' citations — see
        # `_query_groups_payload`'s docstring for why.
        "query_groups": _query_groups_payload(workload),
    }
    if dbt is not None:
        payload["dbt"] = dbt
    return payload


def _query_groups_payload(workload: Workload) -> list[dict]:
    """Every query group this run's workload carries — **not only the ones some proposal
    currently cites.**

    **This function's first version scoped the result to `proposals`' own
    `fingerprint_digests` citations, and that was a Critical bug (Task 6's fix round 3),
    the identical blind spot the sanctioned `physical_state` scoping fix (see
    `advise_payload`'s docstring) closed one payload key over.** Walk the same happy path
    that motivated that fix: run A proposes an index because a query group is slow; the
    user creates the index; run B's rule no longer fires, so run B *cites* none of that
    group's digest — and a citation-scoped `query_groups` would omit the group entirely
    from `after`, even though it is still running, unchanged, twelve times over. `verify`
    would then read the group's absence as `DISAPPEARED` and never reach `IMPROVED` in
    exactly the success case this whole feature exists to celebrate. Task 3's original
    rationale for scoping to citations — "a group nobody proposed on is not evidence *for
    a proposal*" — is still true of evidence, but `query_groups` is not only evidence for
    *this* run's proposals: it is the record `sqlquality verify` compares against a
    *second* run, which cannot know in advance what the first run happened to cite, or
    whether the second run will still cite it at all. So this now returns one entry per
    group in `workload.stats`, unconditionally — bounded by the same `--limit` that
    already bounds `workload.stats` itself, and countable in the payload today via
    `analyzed.query_groups`/`analyzed.query_groups_in_window`, so a consumer is never
    left guessing at how large this list can get.

    Every proposal that names a query group does so with a digest (see
    `workload.fingerprint.fingerprint_id`), never the group's full canonical SQL, so this
    is still the one place a digest is resolved back to its timings — it recomputes the
    same digest from each `QueryStat.fingerprint` rather than trusting any string a
    proposal's evidence happens to carry; that part of the original design is unchanged.

    `mean_ms` is `None`, never `0.0`, for a group with zero recorded calls: a group that was
    never called has no meaningful mean latency to report, and `0.0` reads as "instant",
    the opposite of "unknown". This mirrors the present-but-null idiom `window` and
    `physical_state` already use elsewhere in this payload for "this run could not tell
    you", as against a real measurement of zero.

    Order follows `workload.stats` (cost descending, then fingerprint — see `ingest()`),
    not a fresh sort by digest: that order is already deterministic run-to-run for the same
    workload, and re-sorting by digest would throw away the cost ordering for no gain.

    No longer takes `proposals` at all — it has nothing left to filter by. A dangling
    citation is consequently no longer reachable from this function's side: every digest
    any proposal's `fingerprint_digests` can possibly name resolves to an entry here,
    since every group in `workload.stats` does, regardless of `PostgresWorkloadAdapter.
    _collapse_index_prefixes`/`_dedupe_by_ddl` discarding an absorbed proposal's own
    citations (Task 3) — that collapse can still narrow what a *proposal* cites, but it
    can no longer narrow what `query_groups` *reports*.
    """
    groups = []
    for stat in workload.stats:
        groups.append(
            {
                "digest": fingerprint_id(stat.fingerprint),
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


#: Evidence keys present in `--json` but suppressed from `--markdown`'s generic evidence
#: line. `fingerprint_digests` is a machine correlation key — a list of 12-character
#: digests, one per backing query group, added so `sqlquality verify` can match a
#: proposal's evidence against `query_groups` across two artifacts. It carries no meaning
#: to a human reading a report: the human-relevant number is already right beside it as
#: `fingerprints` / `co_occurring_fingerprints`. Rendering it generically here — as every
#: other evidence key is — would print, for a proposal backed by thirty query groups,
#: thirty opaque hashes in a line meant for a human to read. This is the same bloat
#: `fingerprint_id`'s own docstring already describes for the raw fingerprint text itself
#: ("printed the whole statement a second time... most of the proposal's evidence block,
#: duplicated"), just for a different payload: markdown is the human surface, `--json` is
#: the machine surface, and a correlation key belongs only on the machine surface.
_MARKDOWN_SUPPRESSED_EVIDENCE_KEYS = frozenset({"fingerprint_digests"})


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
        #
        # `_MARKDOWN_SUPPRESSED_EVIDENCE_KEYS` is filtered here, not upstream in `evidence`
        # itself: `--json` still carries every key unfiltered, since `verify` (a later
        # task) needs `fingerprint_digests` there. Only this human-facing render omits it.
        evidence = ", ".join(
            f"{_md_escape(k)}={_md_escape(_jsonable(v))}"
            for k, v in sorted(p.evidence.items())
            if k not in _MARKDOWN_SUPPRESSED_EVIDENCE_KEYS
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


# --- `sqlquality verify`: the pair-level facts, and the three surfaces that render them ---
#
# Everything below is derived from two already-parsed `advise --json` payloads and the
# `ProposalVerdict` list `sqlquality.verify.verdicts` produced from them. It touches no
# database and no filesystem, exactly like `verify.py` itself.
#
# The derivation lives here, once, rather than in `cli.verify`: the terminal, `--json` and
# `--markdown` surfaces must not each re-derive the window relation or recount the query
# groups, because three copies of a disclosure are three places for one of them to fall
# behind — and a *missing* disclosure is this feature's recurring defect, not a cosmetic
# problem.


@dataclass(frozen=True)
class ArtifactFacts:
    """One artifact's own facts, as `verify` reports them beside the verdict table.

    Every field is `| None` where the artifact might not carry it, and `None` means "this
    artifact does not say", never a substituted `0`. A `total_cost_ms` of `0.0` is a real
    measurement of an idle window; a `total_cost_ms` of `None` is a malformed or truncated
    artifact, and rendering the second as the first would put a number on the screen that
    no run ever measured.

    `collisions` holds the keys `index_proposals` could not resolve to a single proposal
    **within this one artifact** (see `ProposalIndex`): a local ambiguity, disclosed, never
    a disappearance and never a new finding.
    """

    engine: str | None
    window_description: str | None
    total_cost_ms: float | None
    query_groups: int | None
    degraded: tuple[tuple[str, str], ...]
    matched_proposals: int
    collisions: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class VerifyContext:
    """The pair-level facts every `verify` surface needs, derived once.

    `ceiling` is `confidence_for(relation)` — the confidence no verdict in this run may
    exceed — carried here so the report can state the ceiling even when the verdict list is
    empty, which is precisely when a user most needs to know why.

    `new_in_after` holds keys matched unambiguously in `after` and absent from `before`'s
    *matched* keys **and** from `before`'s collision keys. Excluding the collisions is
    Ruling 4 of Task 4: a key that identified two proposals in `before` is ambiguous there,
    not missing from there, so calling its `after` counterpart "new" would turn a local
    matching ambiguity into a claim about the user's database.
    """

    before: ArtifactFacts
    after: ArtifactFacts
    relation: WindowRelation
    ceiling: Confidence
    limits: WindowLimits
    new_in_after: tuple[tuple[str, ...], ...]


def _artifact_float(payload: dict[str, object], section: str, field: str) -> float | None:
    """`payload[section][field]` as a `float`, or `None` when it is absent or not a number.

    Excludes `bool` (`isinstance(True, int)` is `True` in Python), following
    `verify.py`'s own `_window_duration_seconds`: a stray boolean must not be rendered as a
    millisecond total.
    """
    parent = payload.get(section)
    if not isinstance(parent, dict):
        return None
    value = parent.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _artifact_facts(payload: dict[str, object], index: ProposalIndex) -> ArtifactFacts:
    """One payload's `ArtifactFacts`. Never raises: every field is type-checked, and an
    unexpected shape degrades to `None`/`()` rather than to a fabricated value.
    """
    window = payload.get("window")
    description = window.get("description") if isinstance(window, dict) else None
    engine = payload.get("engine")
    groups = payload.get("query_groups")
    degraded: list[tuple[str, str]] = []
    entries = payload.get("degraded")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            capability = entry.get("capability")
            reason = entry.get("reason")
            if isinstance(capability, str):
                degraded.append((capability, reason if isinstance(reason, str) else "(no reason)"))
    return ArtifactFacts(
        engine=engine if isinstance(engine, str) else None,
        window_description=description if isinstance(description, str) else None,
        total_cost_ms=_artifact_float(payload, "analyzed", "total_cost_ms"),
        query_groups=len(groups) if isinstance(groups, list) else None,
        degraded=tuple(degraded),
        matched_proposals=len(index.matched),
        collisions=tuple(sorted(index.collisions)),
    )


def verify_context(before: dict[str, object], after: dict[str, object]) -> VerifyContext:
    """Everything `verify`'s three surfaces share, computed from the two payloads once."""
    before_index = index_proposals(before)
    after_index = index_proposals(after)
    relation = classify_windows(before, after)
    new_in_after = tuple(
        sorted(set(after_index.matched) - set(before_index.matched) - set(before_index.collisions))
    )
    return VerifyContext(
        before=_artifact_facts(before, before_index),
        after=_artifact_facts(after, after_index),
        relation=relation,
        ceiling=confidence_for(relation),
        limits=window_limits(before, after),
        new_in_after=new_in_after,
    )


#: What each window relation means for the verdicts below it, in the words a user needs
#: rather than the enum's name. A plain lookup, like `verify.py`'s `_CONFIDENCE_BY_RELATION`
#: and `_MISMATCH_EFFECT`, so a fifth relation added to the enum raises `KeyError` here
#: instead of silently printing nothing —
#: `test_every_window_relation_has_a_user_facing_caveat` pins it directly.
#:
#: `NESTED`'s entry is the load-bearing one, and the reason this text exists at all: it is
#: the common Postgres path (counters baselined once, never reset), and on it a real
#: improvement is *understated*, so a user who is not told will read `unchanged` as "it did
#: not work". `pg_stat_statements_reset()` is named because it is the remedy, and it is
#: named as something the *user* runs: sqlquality never writes to the user's database.
_WINDOW_RELATION_CAVEAT: Final[dict[WindowRelation, str]] = {
    WindowRelation.DISJOINT: (
        "Window relation: disjoint — the two runs report different stats_reset_at instants "
        "and neither restricted its window, so the counters were cleared between them and "
        "the two measurements are independent samples. This is the cleanest comparison "
        "available; verdicts may claim high confidence."
    ),
    WindowRelation.COMPARABLE: (
        "Window relation: comparable — both runs requested the same --since duration, so "
        "the two windows are the same length by construction (the absolute cutoffs differ "
        "and are deliberately not compared). Verdicts may claim high confidence."
    ),
    WindowRelation.NESTED: (
        "Window relation: nested — both runs report the same stats_reset_at and neither "
        "restricted its window, so the after run's cumulative pg_stat_statements counters "
        "contain the before run's. Every pre-change execution is still averaged into the "
        "after mean, so a real improvement is understated here, sometimes badly: a proposal "
        "that genuinely helped can read as 'unchanged'. Verdicts are capped at medium "
        "confidence for that reason. For an undiluted comparison, call "
        "pg_stat_statements_reset() yourself right after applying a change and take the "
        "after artifact from there — sqlquality never writes to your database, this reset "
        "included."
    ),
    WindowRelation.INCOMPARABLE: (
        "Window relation: incomparable — the two windows cannot be placed relative to one "
        "another (one run restricted its window with --since and the other did not, or the "
        "two requested different durations, or stats_reset_at is unknown on at least one "
        "side). Every verdict below is capped at low confidence, and none of them "
        "establishes that a change did or did not help."
    ),
}

#: Ruling 1 of Task 7, disclosed on every run rather than assumed away. An `advise` payload
#: carries no run timestamp — deliberately: Task 3 established byte-determinism, and a
#: `generated_at` field would break it — so `verify` cannot detect a swapped pair in general.
#: The one detectable case is refused up front (`cli.verify`): `stats_reset_at` is monotonic
#: per server, so a non-null `after` value strictly earlier than a non-null `before` one
#: demonstrates the two artifacts were passed in the wrong order. Everything else is taken on
#: trust from the argument order, and saying so is the whole point — the alternative is a user
#: reading a reversed comparison as a regression.
_RUN_ORDER_CAVEAT: Final[str] = (
    "Run order is taken from the argument order: BEFORE, then AFTER. An advise artifact "
    "carries no run timestamp (its bytes are reproducible by design), so a swapped pair is "
    "only detectable when both runs report a stats_reset_at and the after run's is earlier — "
    "which is refused outright. Otherwise sqlquality cannot tell the two apart, and does not "
    "guess."
)


def _keys_text(keys: Sequence[tuple[str, ...]]) -> str:
    """Proposal keys as a readable list — `ADV104 public.orders`, comma separated."""
    return ", ".join(verify_proposal_label(key) for key in keys)


def verify_caveats(context: VerifyContext) -> list[str]:
    """Every disclosure this artifact pair earns, in the order a reader needs them.

    Each entry is a complete sentence, rendered identically on all three surfaces (stderr,
    `--markdown`, and `--json`'s `caveats` list). None of them is optional decoration: each
    names a condition under which something the verdict table shows means less than it
    appears to, and the whole point of this feature is that such a condition is disclosed
    rather than silently folded into a verdict.
    """
    caveats = [_WINDOW_RELATION_CAVEAT[context.relation], _RUN_ORDER_CAVEAT]
    if context.limits.may_be_sampling_artifact:
        caveats.append(
            "The two runs sampled different (or unknown) numbers of query groups "
            f"(--limit before={context.limits.before}, after={context.limits.after}). A "
            "query group present in one artifact and absent from the other may therefore be "
            "a sampling artifact rather than a real disappearance, so no verdict below is "
            "graded 'disappeared' on the strength of such an absence alone."
        )
    for side, facts in (("before", context.before), ("after", context.after)):
        for capability, reason in facts.degraded:
            caveats.append(
                f"The {side} run ran with reduced coverage — {capability}: {reason}. A read "
                "that could not run produces the same emptiness a real change would, so any "
                "verdict resting on something being absent from that run says so."
            )
    for side, facts in (("before", context.before), ("after", context.after)):
        if facts.collisions:
            caveats.append(
                f"{len(facts.collisions)} recommendation(s) in the {side} run could not be "
                "matched unambiguously — their key identified more than one proposal within "
                f"that run's own artifact — so they carry no verdict below: "
                f"{_keys_text(facts.collisions)}. This is a fact about the {side} artifact, "
                "not a disappearance and not a new finding."
            )
    if context.new_in_after:
        caveats.append(
            f"{len(context.new_in_after)} proposal(s) appear only in the after run and "
            "therefore carry no verdict: verify grades each before-run proposal against the "
            "after run, and a finding with no counterpart to compare against is new rather "
            f"than changed: {_keys_text(context.new_in_after)}."
        )
    return caveats


def verify_workload_line(context: VerifyContext) -> str:
    """The workload-context line: total window cost and query-group count, both runs.

    Printed on every run, because a global workload shift is the confound this whole feature
    was designed around (see the design spec's decision 4): `cost_share` falling while the
    mean per call held steady is *not* an improvement, and a reader can only see that if
    both runs' totals are on screen rather than deduced.
    """

    def side(facts: ArtifactFacts) -> str:
        cost = "unknown" if facts.total_cost_ms is None else f"{facts.total_cost_ms:.1f} ms"
        groups = "unknown" if facts.query_groups is None else str(facts.query_groups)
        return f"{cost} across {groups} query group(s)"

    return f"workload: before {side(context.before)}; after {side(context.after)}"


def verify_proposal_label(key: Sequence[str]) -> str:
    """One proposal key as a single cell: the code, then whatever identifies it.

    `("ADV001", "public", "orders", "status")` renders as `ADV001 public.orders (status)`,
    and the statement-scoped `("ADV005", "a1b2c3d4e5f6")` as `ADV005 a1b2c3d4e5f6` — the
    key's own parts, never re-derived from the proposal, so two proposals `verify` considers
    distinct can never print identically.
    """
    if not key:
        return "(unkeyed)"
    code, *rest = key
    if len(rest) >= 2:
        relation = f"{rest[0]}.{rest[1]}"
        remainder = ", ".join(rest[2:])
        return f"{code} {relation}" + (f" ({remainder})" if remainder else "")
    return f"{code} {' '.join(rest)}".rstrip()


def verify_applied_label(applied: bool | None) -> str:
    """`applied` as a word. `None` is "unknown", never "no": the distinction between "you
    did not do it" and "we could not tell whether you did" is the whole reason the field is
    `bool | None` (see `ProposalVerdict`).
    """
    if applied is None:
        return "unknown"
    return "yes" if applied else "no"


def verify_mean_cell(verdict: ProposalVerdict) -> str:
    """`100.0 → 40.0 ms`, with an em dash for a side that has no comparable mean.

    A `None` mean prints as `—`, never as `0.0`: `0.0` reads as "instantaneous", the
    opposite of "no usable measurement" (the same distinction `_query_groups_payload` keeps
    for `mean_ms`).
    """
    if verdict.mean_before is None and verdict.mean_after is None:
        return "—"
    before = "—" if verdict.mean_before is None else f"{verdict.mean_before:.1f}"
    after = "—" if verdict.mean_after is None else f"{verdict.mean_after:.1f}"
    return f"{before} → {after} ms"


def verify_summary(verdicts: Sequence[ProposalVerdict]) -> dict[str, object]:
    """Headline counts: how many proposals were graded, how many were applied, and how many
    landed in each outcome. Every `VerifyOutcome` member appears, `0` included, so a
    consumer never has to distinguish "none of these" from "this version has no such
    outcome".
    """
    return {
        "proposals": len(verdicts),
        "applied": sum(1 for v in verdicts if v.applied is True),
        "not_applied": sum(1 for v in verdicts if v.applied is False),
        "applied_unknown": sum(1 for v in verdicts if v.applied is None),
        "outcomes": {
            outcome.value: sum(1 for v in verdicts if v.outcome is outcome)
            for outcome in VerifyOutcome
        },
    }


def _facts_payload(facts: ArtifactFacts) -> dict[str, object]:
    return {
        "engine": facts.engine,
        "window_description": facts.window_description,
        "total_cost_ms": facts.total_cost_ms,
        "query_groups": facts.query_groups,
        "degraded": [{"capability": cap, "reason": reason} for cap, reason in facts.degraded],
        "matched_proposals": facts.matched_proposals,
        "unmatched_keys": [list(key) for key in facts.collisions],
    }


def verify_payload(
    verdicts: Sequence[ProposalVerdict],
    before: dict[str, object],
    after: dict[str, object],
) -> dict:
    """JSON-serializable summary of a `verify` run.

    Carries the caveats as data (`caveats`), not only as terminal text: a machine consumer
    that reports "4 improved" without the window relation and its consequences would
    reproduce, one layer up, exactly the over-claim this feature exists to prevent.
    """
    context = verify_context(before, after)
    return {
        "before": _facts_payload(context.before),
        "after": _facts_payload(context.after),
        "window_relation": context.relation.value,
        "confidence_ceiling": context.ceiling.value,
        "limit": {
            "before": context.limits.before,
            "after": context.limits.after,
            "may_be_sampling_artifact": context.limits.may_be_sampling_artifact,
        },
        "caveats": verify_caveats(context),
        "summary": verify_summary(verdicts),
        "new_in_after": [list(key) for key in context.new_in_after],
        "verdicts": [
            {
                "key": list(verdict.key),
                "code": verdict.code,
                "applied": verdict.applied,
                "outcome": verdict.outcome.value,
                "confidence": verdict.confidence.value,
                "mean_ms_before": verdict.mean_before,
                "mean_ms_after": verdict.mean_after,
                "cost_share_before": verdict.cost_share_before,
                "cost_share_after": verdict.cost_share_after,
                "note": verdict.note,
            }
            for verdict in verdicts
        ],
    }


def render_verify_markdown(
    verdicts: Sequence[ProposalVerdict],
    before: dict[str, object],
    after: dict[str, object],
) -> str:
    """Render a `verify` run as markdown (suitable for a ticket or PR comment).

    Every value goes through `_md_escape` for the same reason `render_advise_markdown`'s do:
    a proposal key is built from live catalog identifiers, and one containing a `|` or a
    newline would otherwise fabricate table columns or rows.
    """
    context = verify_context(before, after)
    summary = verify_summary(verdicts)
    outcomes = summary["outcomes"]
    improved = outcomes["improved"] if isinstance(outcomes, dict) else 0
    lines = [
        f"# sqlquality verify — {_md_escape(context.before.engine)}",
        "",
        f"**Proposals graded:** {summary['proposals']}  ",
        f"**Applied:** {summary['applied']} (not applied {summary['not_applied']}, "
        f"could not tell {summary['applied_unknown']})  ",
        f"**Improved:** {improved}  ",
        f"**Window relation:** {context.relation.value} "
        f"(confidence ceiling: {context.ceiling.value})",
        "",
        f"**Before window:** {_md_escape(context.before.window_description)}  ",
        f"**After window:** {_md_escape(context.after.window_description)}",
        "",
        _md_escape(verify_workload_line(context)),
        "",
        "## What this comparison can and cannot say",
        "",
    ]
    lines += [f"- {_md_escape(caveat)}" for caveat in verify_caveats(context)]
    lines.append("")
    if not verdicts:
        lines.append(
            "No proposal in the before run could be graded — see the caveats above for why."
        )
        return "\n".join(lines) + "\n"
    lines += [
        "| proposal | applied | outcome | mean per call | confidence |",
        "|---|---|---|---|---|",
    ]
    for verdict in verdicts:
        lines.append(
            f"| {_md_escape(verify_proposal_label(verdict.key))} "
            f"| {verify_applied_label(verdict.applied)} "
            f"| {verdict.outcome.value} "
            f"| {_md_escape(verify_mean_cell(verdict))} "
            f"| {verdict.confidence.value} |"
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for verdict in verdicts:
        lines.append(f"### {_md_escape(verify_proposal_label(verdict.key))}")
        lines.append("")
        lines.append(
            f"- applied: {verify_applied_label(verdict.applied)}, "
            f"outcome: {verdict.outcome.value}, confidence: {verdict.confidence.value}"
        )
        lines.append(f"- mean per call: {_md_escape(verify_mean_cell(verdict))}")
        # `cost_share` rides along as context — whether the finding still *matters* — never
        # as the measure of whether it got better (the design spec's decision 4).
        lines.append(
            "- cost share (context, not a speed measure): "
            f"{_share_text(verdict.cost_share_before)} → {_share_text(verdict.cost_share_after)}"
        )
        if verdict.note:
            lines.append(f"- {_md_escape(verdict.note)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _share_text(share: float | None) -> str:
    """A cost share as a percentage, or an em dash when the artifact does not carry one.

    Matches `cli.advise`'s and `render_advise_markdown`'s existing rendering of the same
    value, so the same number does not appear in two formats across two commands.
    """
    return f"{share:.1%}" if share is not None else "—"
