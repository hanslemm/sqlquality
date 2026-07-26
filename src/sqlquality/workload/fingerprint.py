"""Ingest raw query history: capture literal-derived flags, redact, fingerprint, group.

Redaction lives here — in engine-agnostic code — so there is exactly one place to audit
for literal leakage, rather than one per adapter.
"""

from __future__ import annotations

import re
from collections import defaultdict

from sqlglot import exp

from sqlquality.antipatterns import has_select_star
from sqlquality.models import QueryStat, Workload, WorkloadFetch
from sqlquality.sqlast import SqlParseError, parse

#: A LIKE/ILIKE pattern starting with '%' is non-sargable. Detectable only before
#: redaction, so it is captured as a flag and the literal is then discarded.
FLAG_LEADING_WILDCARD_LIKE = "leading_wildcard_like"
#: The query group projects a star. Captured pre-qualify (qualify runs expand_stars=False).
FLAG_SELECT_STAR = "select_star"

#: Statements that are our own introspection, dbt's metadata, or DDL — advising on these
#: would be advising on our own noise.
_NOISE = re.compile(
    r"\b(pg_stat_statements|pg_stat_user_indexes|pg_stats|pg_class|pg_index|pg_namespace"
    r"|information_schema|svv_table_info|svl_statementtext|sys_query_history"
    r"|account_usage|create\s+index|drop\s+index|create\s+table|alter\s+table"
    r"|vacuum|analyze|begin|commit|rollback|set\s+)\b",
    re.IGNORECASE,
)


def is_noise(sql: str) -> bool:
    """True if a statement is introspection, session management, or DDL."""
    return _NOISE.search(sql) is not None


def literal_flags(tree: exp.Expression) -> frozenset[str]:
    """Signals that can only be read while literals are still present."""
    flags: set[str] = set()
    for node in tree.find_all(exp.Like, exp.ILike):
        pattern = node.args.get("expression")
        if isinstance(pattern, exp.Literal) and pattern.is_string and pattern.this.startswith("%"):
            flags.add(FLAG_LEADING_WILDCARD_LIKE)
            break
    if has_select_star(tree):
        flags.add(FLAG_SELECT_STAR)
    return frozenset(flags)


def redact_tree(tree: exp.Expression) -> exp.Expression:
    """Return a copy of ``tree`` with every literal replaced by a bind placeholder."""
    copy = tree.copy()
    for literal in list(copy.find_all(exp.Literal)):
        literal.replace(exp.Placeholder())
    return copy


def ingest(fetch: WorkloadFetch, dialect: str, *, keep_literals: bool = False) -> Workload:
    """Parse, flag, redact, fingerprint and group raw history rows into a Workload.

    Unparseable and noise statements are counted, never raised and never silently
    dropped. Raw rows are not retained past this function.
    """
    calls: dict[str, int] = defaultdict(int)
    cost: dict[str, float] = defaultdict(float)
    bytes_scanned: dict[str, int] = defaultdict(int)
    flags: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    skipped_unparseable = 0
    skipped_noise = 0

    for row in fetch.rows:
        if is_noise(row.sql):
            skipped_noise += 1
            continue
        try:
            tree = parse(row.sql, dialect)
        except SqlParseError:
            skipped_unparseable += 1
            continue

        row_flags = literal_flags(tree)
        analyzed = tree if keep_literals else redact_tree(tree)
        # identify=True quotes every identifier, giving a stable canonical key regardless
        # of how the engine happened to spell the query.
        key = analyzed.sql(dialect, identify=True)
        calls[key] += row.calls
        cost[key] += row.total_time_ms
        if row.bytes_scanned is not None:
            bytes_scanned[key] += row.bytes_scanned
        flags[key] |= set(row_flags)
        display.setdefault(key, analyzed.sql(dialect))

    stats = tuple(
        sorted(
            (
                QueryStat(
                    fingerprint=key,
                    sql=display[key],
                    calls=calls[key],
                    total_time_ms=cost[key],
                    bytes_scanned=bytes_scanned.get(key) or None,
                    flags=frozenset(flags[key]),
                )
                for key in calls
            ),
            key=lambda s: s.total_time_ms,
            reverse=True,
        )
    )
    return Workload(
        stats=stats,
        window_description=fetch.window_description,
        skipped_unparseable=skipped_unparseable,
        skipped_noise=skipped_noise,
    )
