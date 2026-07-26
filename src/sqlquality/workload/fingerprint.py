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

#: Statement-leading keywords marking session control, DDL, or maintenance — never user
#: workload. Anchored at the start deliberately: an unanchored `set\s+` also matches
#: `UPDATE ... SET`, and an unanchored `commit` matches a literal like `action = 'commit'`.
_LEADING_NOISE = re.compile(
    r"^\s*(?:set|reset|begin|start|commit|rollback|savepoint|discard|vacuum|analyze"
    r"|create|drop|alter|truncate|grant|revoke|comment|reindex|cluster|explain|copy"
    r"|prepare|deallocate|declare|fetch|close|listen|unlisten|notify|show|call|do)\b",
    re.IGNORECASE,
)
#: Relations only our own introspection (or dbt's metadata) reads. Matched anywhere, since
#: a statement can reference them in any position.
_INTROSPECTION = re.compile(
    r"\b(?:pg_stat_statements|pg_stat_database|pg_stat_user_indexes|pg_stats|pg_class"
    r"|pg_index|pg_namespace|pg_attribute|pg_database|pg_locks|information_schema"
    r"|svv_table_info|svv_redshift_columns|svv_alter_table_recommendations"
    r"|svl_statementtext|svl_query_summary|stl_query|sys_query_history"
    r"|account_usage)\b",
    re.IGNORECASE,
)


def is_noise(sql: str) -> bool:
    """True for session control, DDL, maintenance, and introspection statements.

    `SELECT`, `INSERT`, `UPDATE` and `DELETE` are all user workload and are always kept.
    A write's `WHERE` clause benefits from an index exactly as a read's does, and write
    volume is precisely what makes an index expensive to maintain — so dropping DML would
    both hide index candidates and bias the cost picture toward reads.
    """
    return bool(_LEADING_NOISE.match(sql) or _INTROSPECTION.search(sql))


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
