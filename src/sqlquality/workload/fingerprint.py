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


#: `DECLARE <name> [BINARY] [ASENSITIVE|INSENSITIVE] [[NO] SCROLL] CURSOR
#:  [WITH|WITHOUT HOLD] FOR <query>` — the full PostgreSQL grammar for the statement every
#: psycopg2 server-side cursor emits.
#:
#: Anchored on `CURSOR ... FOR` rather than on the first `FOR`, because a cursor name is an
#: identifier and a quoted one may contain the word: `DECLARE "for sale" CURSOR FOR ...`
#: would otherwise be cut at the wrong place and yield unparseable text. The name alternative
#: matches a quoted identifier (with doubled quotes escaped) before an unquoted one for the
#: same reason.
_DECLARE_CURSOR = re.compile(
    r"^\s*DECLARE\s+"
    r'(?:"(?:[^"]|"")*"|[A-Za-z_]\w*)\s+'
    r"(?:BINARY\s+)?"
    r"(?:ASENSITIVE\s+|INSENSITIVE\s+)?"
    r"(?:NO\s+SCROLL\s+|SCROLL\s+)?"
    r"CURSOR\s+"
    r"(?:WITH\s+HOLD\s+|WITHOUT\s+HOLD\s+)?"
    r"FOR\s+(?P<query>\S.*)$",
    re.IGNORECASE | re.DOTALL,
)

#: `COPY ( <query> ) TO ...` — the only COPY form carrying predicates worth analysing.
#: `COPY <table> TO` is a whole-relation dump with no predicates, and `COPY ... FROM` is a
#: write; both stay noise. The capture is greedy to the last `)` so a query containing
#: parentheses survives; the result is validated by the caller's parse, so a mis-cut yields
#: an unparseable count rather than a wrong analysis.
_COPY_QUERY = re.compile(
    r"^\s*COPY\s*\(\s*(?P<query>.*)\s*\)\s*TO\b",
    re.IGNORECASE | re.DOTALL,
)


def unwrap(sql: str) -> str:
    """The inner query of a cursor declaration or `COPY (...) TO`, else ``sql`` unchanged.

    `DECLARE ... CURSOR FOR SELECT ...` and `COPY (SELECT ...) TO ...` are ordinary reads
    with real predicates, but both begin with a keyword the noise filter drops — so on any
    workload using server-side cursors (every psycopg2 `cursor(name=...)`, which is what
    Django and SQLAlchemy emit for large result sets) the hottest reads were counted as
    "filtered" and thrown away.

    What this actually restores differs by form, measured on PostgreSQL 16. `COPY
    (SELECT ...) TO ...` attributes correctly: `pg_stat_statements` charges the whole
    execution's time and row count to the `COPY` statement itself, so cost-weighted rules
    see the real number.

    `DECLARE ... CURSOR FOR` does **not**: Postgres attributes the work of actually reading
    rows to the `FETCH` statements that follow, which `is_noise` still filters (a `FETCH`
    carries no query text, so it has no predicates to attribute and unfiltering it would
    only inflate the denominator with uncounted cost). The `DECLARE` itself is recorded
    with near-zero calls, time and rows — opening a cursor does no scanning. So a cursor
    read recovered here contributes its predicate columns to `aggregate` and can still join
    an index candidate, but it cannot earn a proposal on cost alone: the default
    `--min-cost-share` can suppress it outright, and `WITH HOLD` does not change this.

    Text surgery rather than AST surgery, for a reason that is not a preference: sqlglot
    cannot parse `DECLARE` at all — it falls back to `exp.Command` and leaves the entire
    tail as a single string literal, so there is no inner tree to lift. `COPY` *does* parse
    (to `exp.Copy` with a `Subquery`), but doing both here keeps one code path and, more
    importantly, lets `is_noise` run on the *unwrapped* text — which is what stops a
    `DECLARE c CURSOR FOR SELECT * FROM pg_stat_statements` from smuggling our own
    introspection into the analysed workload.

    Returns the input unchanged when nothing matches. The caller parses the result, so a
    partial or malformed wrapper degrades to the pre-existing behaviour — counted
    unparseable or filtered — rather than producing a wrong analysis.
    """
    for pattern in (_DECLARE_CURSOR, _COPY_QUERY):
        match = pattern.match(sql)
        if match is not None:
            return match.group("query").strip()
    return sql


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


def _inside_placeholder(node: exp.Expression) -> bool:
    """True if ``node`` sits inside a `$N` parameter marker.

    `pg_stat_statements` has already replaced the literal that was there; the integer left
    behind is Postgres's own index, not user data, and rewriting it corrupts the statement.
    """
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Parameter):
            return True
        parent = parent.parent
    return False


def redact_tree(tree: exp.Expression) -> exp.Expression:
    """Return a copy of ``tree`` with every literal replaced by a bind placeholder."""
    copy = tree.copy()
    for literal in list(copy.find_all(exp.Literal)):
        if _inside_placeholder(literal):
            continue
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
        # Unwrap *before* the noise test, so a cursor declaration is judged on the query it
        # declares. Judging the wrapper drops the read; judging the inner query keeps a real
        # read and still filters an inner introspection query.
        sql = unwrap(row.sql)
        if is_noise(sql):
            skipped_noise += 1
            continue
        try:
            tree = parse(sql, dialect)
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
            key=lambda s: (-s.total_time_ms, s.fingerprint),
        )
    )
    return Workload(
        stats=stats,
        window_description=fetch.window_description,
        skipped_unparseable=skipped_unparseable,
        skipped_noise=skipped_noise,
    )
