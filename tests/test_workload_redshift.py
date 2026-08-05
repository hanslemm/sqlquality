import re
import sys
import types
from datetime import timedelta
from pathlib import Path

import sqlglot
import pytest

from sqlquality.models import ConnectionParams, Relation
from sqlquality.workload import get_workload_adapter
from sqlquality.workload.base import MAX_TIMEOUT_S
from sqlquality.workload.fingerprint import ingest
from sqlquality.workload.redshift import (
    CAP_ADVISOR,
    CAP_SCHEMA,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    DEGRADATION_READ_ONLY,
    RedshiftWorkloadAdapter,
)
from sqlquality.workload.session import READ_ONLY_SQL

EXPECTED_CAPABILITIES = {CAP_WORKLOAD, CAP_SCHEMA, CAP_TABLE_FACTS, CAP_ADVISOR}


def test_registry_returns_the_redshift_adapter():
    adapter = get_workload_adapter("redshift")
    assert adapter.engine == "redshift"


def test_every_capability_has_a_statement_and_a_hint():
    statements = RedshiftWorkloadAdapter().introspection_sql()
    assert {s.capability for s in statements} == EXPECTED_CAPABILITIES
    for statement in statements:
        assert statement.sql.strip()
        assert statement.privilege_hint.strip()


@pytest.mark.parametrize("capability", sorted(EXPECTED_CAPABILITIES))
def test_every_statement_parses_as_redshift_sql(capability):
    """Syntax validation is the one correctness check available without a cluster.

    The catalog SQL in this adapter cannot be executed during development — there is no
    Redshift container and `svv_*`/`sys_*` do not exist in Postgres. Parsing each statement
    with sqlglot's redshift dialect cannot catch a wrong column name, but it catches a
    malformed statement, which would otherwise be invisible until a user ran it.
    """
    sql = RedshiftWorkloadAdapter.SQL[capability]
    # `%s` placeholders are libpq's, not SQL — sqlglot cannot parse them, so they become
    # bind markers for the purposes of this check.
    parsed = sqlglot.parse_one(sql.replace("%s", "?"), dialect="redshift")
    assert parsed is not None


@pytest.mark.parametrize("capability", sorted(EXPECTED_CAPABILITIES))
def test_no_statement_writes(capability):
    """Same guard the Postgres adapter carries, for the same reason."""
    forbidden = (
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "truncate",
        "grant",
        "revoke",
        "vacuum",
        "analyze",
    )
    lowered = RedshiftWorkloadAdapter.SQL[capability].lower()
    found = {verb for verb in forbidden if re.search(rf"\b{verb}\b", lowered)}
    assert not found, f"{capability} contains write verb(s): {sorted(found)}"


def _normalized(sql: str) -> str:
    """`sql` with every run of whitespace collapsed, so a reformat of the statement cannot
    break a predicate assertion that is really about the predicate."""
    return " ".join(sql.lower().split())


def test_workload_statement_is_scoped_to_the_current_database():
    """The same guard `tests/test_workload_postgres.py` carries under this exact name, for
    the same claim — it was not carried across when this engine was added, and deleting the
    predicate left the whole suite green.

    Without the scope, an `advise` run against a shared cluster ingests *other databases'*
    query history: their relation names are attributed to the connected database's schema
    map, and proposals — including full-table rewrites — are generated for tables the
    session was never pointed at.
    """
    sql = _normalized(RedshiftWorkloadAdapter.SQL[CAP_WORKLOAD])
    assert "sys_query_history" in sql
    assert "database_name = current_database()" in sql


def test_workload_statement_counts_only_successful_executions():
    """`sys_query_history` records failed and cancelled executions alongside successful
    ones, and unlike `pg_stat_statements` it is per-execution rather than pre-aggregated.
    Without this filter, work that never completed lands in the `cost_share` denominator
    and dilutes every proposal's share — and an aborted statement's elapsed time measures
    how long it ran before dying, not the cost of the query it was trying to be.
    """
    sql = _normalized(RedshiftWorkloadAdapter.SQL[CAP_WORKLOAD])
    assert "status = 'success'" in sql


@pytest.mark.parametrize("capability", sorted(EXPECTED_CAPABILITIES))
def test_a_denied_capability_records_its_privilege_hint_in_degraded(capability):
    """The recorded degradation must carry the capability's privilege hint, not only the
    driver's message.

    These four hint strings are this adapter's stated mitigation for column names it cannot
    verify against a cluster ("a wrong name costs one capability, recorded in `degraded`
    naming the statement"), and they are what someone hands their DBA. Dropping the hint
    from the message left 904 tests green, so the mitigation could be disconnected in
    silence. The expected text is read back out of `introspection_sql()` rather than
    duplicated here, so this cannot drift from the hint a `--dry-run` prints.
    """
    hints = {s.capability: s.privilege_hint for s in RedshiftWorkloadAdapter().introspection_sql()}
    marker = RedshiftWorkloadAdapter.SQL[capability]
    adapter = RedshiftWorkloadAdapter(querier=FakeQuerier({}, fail_markers=(marker,)))
    adapter._run(capability, ())
    assert len(adapter.degraded) == 1
    recorded_capability, reason = adapter.degraded[0]
    assert recorded_capability == capability
    assert "permission denied" in reason, "the driver's own message must survive too"
    assert hints[capability] in reason


def test_the_acquired_redshift_limitations_are_documented_for_users():
    """Three limitations this engine *acquires* must be documented where a user reads, not
    only in a docstring or a privilege hint.

    The first is the sharp one: `_HINTS[CAP_WORKLOAD]` describes the `SYSLOG ACCESS
    UNRESTRICTED` partial-workload trap precisely, and names it as the dangerous class
    *because* it produces no error — but that string only ever reaches a user through
    `--dry-run` or a failure, and this failure never happens. A hint that can only be
    delivered by an event that cannot occur is not disclosure. The other two are the
    per-execution meaning of `--limit` and the fingerprint split, whose siblings ("`cost_share`
    is not a partition", the PL/pgSQL double-count) are already in README `## Limitations`.
    Each claim is asserted separately, so documenting one and omitting another cannot pass.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "SYSLOG ACCESS UNRESTRICTED" in readme
    assert "only the connecting user's own queries" in readme
    assert "counts executions on Redshift and query groups on Postgres" in readme
    assert "Identifier case and attached comments can split one Redshift statement" in readme


def test_there_is_no_ndv_or_index_capability():
    """Redshift exposes no `pg_stats.n_distinct` equivalent and has no indexes.

    Declaring either capability would invite a rule to assume evidence that cannot exist.
    """
    capabilities = {s.capability for s in RedshiftWorkloadAdapter().introspection_sql()}
    assert not any("ndv" in c or "index" in c for c in capabilities)


# Every `WorkloadAdapter` method this feature once left deliberately unbuilt — including
# `render_ddl`, the last one — is now implemented. See git history for the
# `UNIMPLEMENTED`/`test_unimplemented_methods_say_so_rather_than_returning_empty` machinery
# this section used to carry: it named each unbuilt method individually (rather than
# discovering them by reflection) precisely so implementing the last one forced a visible,
# reviewable deletion here rather than an empty parametrisation quietly passing forever.


class FakeQuerier:
    """Returns canned rows per capability, keyed by a distinctive SQL substring.

    Mirrors `tests/test_workload_postgres.py`'s `FakeQuerier` exactly — same dispatch, same
    shape — so the two adapters' fetch tests read the same way.
    """

    def __init__(self, rows_by_marker, fail_markers=()):
        self.rows_by_marker = rows_by_marker
        self.fail_markers = fail_markers
        self.calls = []

    def __call__(self, sql, params):
        self.calls.append((sql, params))
        for marker in self.fail_markers:
            if marker in sql:
                raise RuntimeError(f"permission denied for {marker}")
        for marker, rows in self.rows_by_marker.items():
            if marker in sql:
                return rows
        return []


def _canned(rows_by_capability):
    """A FakeQuerier addressed by capability constant rather than a raw SQL substring."""
    return FakeQuerier(
        {
            RedshiftWorkloadAdapter.SQL[capability]: rows
            for capability, rows in rows_by_capability.items()
        }
    )


def test_fetch_workload_maps_rows_and_reports_no_filter_applied():
    querier = _canned({CAP_WORKLOAD: [("select id from orders where status = 'x'", 25_000)]})
    fetch = RedshiftWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert fetch.rows[0].sql == "select id from orders where status = 'x'"
    # One row per execution, not pre-aggregated — see the aggregation test below.
    assert fetch.rows[0].calls == 1
    # elapsed_time is documented in microseconds; total_time_ms wants milliseconds.
    assert fetch.rows[0].total_time_ms == pytest.approx(25.0)
    assert "no --since filter" in fetch.window_description
    assert "sys_query_history" in fetch.window_description
    # The statement is ORDER BY ... LIMIT n, so what was actually fetched is a truncated
    # top-n, not "everything" — the window text must say so, not merely name the source
    # view. See test_fetch_workload_window_names_the_truncation below for the full pin.
    assert "500" in fetch.window_description


def test_fetch_workload_window_is_honest_that_since_is_honoured():
    """The opposite discipline from Postgres's identically-named test: `sys_query_history`
    *does* carry a per-execution timestamp, so `--since` genuinely can be honoured, and the
    window text must say so rather than staying silent or (worse) copying Postgres's
    disclaimer that it cannot be.
    """
    querier = _canned({CAP_WORKLOAD: []})
    fetch = RedshiftWorkloadAdapter(querier=querier).fetch_workload(timedelta(days=7), 500)
    lowered = fetch.window_description.lower()
    assert "since" in lowered
    assert "honoured" in lowered
    assert "not supported" not in lowered


def test_fetch_workload_window_names_the_truncation_not_full_coverage():
    """`ORDER BY elapsed_time DESC LIMIT n` combined with a `--since` filter means the
    window actually covers is "the n most expensive queries since T", not "everything
    since T" — and `cost_share` denominators throughout the rest of the run are computed
    over exactly that truncated set. A window sentence that only names the cutoff (and
    not the limit) reads as full coverage, which overstates what was actually analysed.
    """
    querier = _canned({CAP_WORKLOAD: []})
    fetch = RedshiftWorkloadAdapter(querier=querier).fetch_workload(timedelta(days=7), 42)
    assert "42" in fetch.window_description
    assert "most expensive" in fetch.window_description.lower()


def test_redshift_reports_the_since_cutoff_it_actually_bound():
    """Unlike Postgres, `sys_query_history` carries timestamps, so `--since` is real and the
    window is comparable by construction — which is what lets `verify` grade it HIGH."""
    adapter = RedshiftWorkloadAdapter(querier=_canned({CAP_WORKLOAD: []}))
    adapter.fetch_workload(timedelta(days=7), 500)
    facts = adapter.window_facts()
    assert facts["since"] is not None
    assert facts["limit"] == 500
    assert facts["stats_reset_at"] is None, "Redshift has no cumulative-counter reset"


def test_redshift_window_facts_report_no_since_when_not_requested():
    adapter = RedshiftWorkloadAdapter(querier=_canned({CAP_WORKLOAD: []}))
    adapter.fetch_workload(None, 500)
    facts = adapter.window_facts()
    assert facts["since"] is None
    assert facts["limit"] == 500


def test_redshift_window_facts_do_not_issue_sql():
    """See the identical Postgres test's docstring: `window_facts()` must only read what
    `fetch_workload` already recorded on the instance.

    Recording the call rather than raising from the stub: `_run` swallows any exception
    into `degraded` and returns `[]`, so a stub that raises would be silently absorbed and
    this test would pass whether or not `window_facts()` actually queried.
    """
    adapter = RedshiftWorkloadAdapter(querier=_canned({CAP_WORKLOAD: []}))
    adapter.fetch_workload(timedelta(days=7), 500)
    calls = []
    adapter._query = lambda sql, params: calls.append((sql, params)) or []
    adapter.window_facts()
    assert calls == [], "window_facts() must not issue any SQL"


def test_fetch_workload_since_is_actually_bound_into_the_statement():
    """Guards the claim the test above makes in prose: `--since` must change what the
    statement is run with, not just what the sentence says. Without this, a
    `window_description` claiming the window was honoured while the query ran with no
    cutoff at all would still pass every other test here.
    """
    querier = FakeQuerier({RedshiftWorkloadAdapter.SQL[CAP_WORKLOAD]: []})
    RedshiftWorkloadAdapter(querier=querier).fetch_workload(timedelta(days=1), 10)
    assert len(querier.calls) == 1
    _sql, params = querier.calls[0]
    cutoff, cutoff_again, limit = params
    assert cutoff is not None
    assert cutoff == cutoff_again
    assert limit == 10


def test_fetch_workload_without_since_binds_no_cutoff():
    """The control for the test above: no `--since` must mean no cutoff bound in either
    placeholder, not merely a friendlier sentence with a filter silently applied anyway.
    """
    querier = FakeQuerier({RedshiftWorkloadAdapter.SQL[CAP_WORKLOAD]: []})
    RedshiftWorkloadAdapter(querier=querier).fetch_workload(None, 10)
    _sql, params = querier.calls[0]
    cutoff, cutoff_again, limit = params
    assert cutoff is None
    assert cutoff_again is None
    assert limit == 10


def test_two_executions_of_the_same_statement_collapse_to_one_query_stat_via_ingest():
    """Task 3's central aggregation claim, confirmed rather than assumed: unlike
    `pg_stat_statements`, `sys_query_history` returns one row per *execution*, so
    `fetch_workload` emits `calls=1` on every row (pinned below). The collapse into one
    `QueryStat` per fingerprint, with `calls` and `total_time_ms` summed, is not this
    adapter's job at all — it happens in the engine-agnostic `ingest()` — and this test is
    what actually proves that happens for Redshift's rows, rather than assuming `ingest()`'s
    Postgres behaviour carries over unchanged.
    """
    querier = _canned(
        {
            CAP_WORKLOAD: [
                ("select id from orders where status = 'a'", 120_000),
                ("select id from orders where status = 'b'", 80_000),
            ]
        }
    )
    fetch = RedshiftWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert len(fetch.rows) == 2
    assert all(row.calls == 1 for row in fetch.rows)

    workload = ingest(fetch, "redshift")
    assert len(workload.stats) == 1
    stat = workload.stats[0]
    assert stat.calls == 2
    assert stat.total_time_ms == pytest.approx(200.0)


def test_identifier_case_and_comments_can_still_split_one_statement_into_two_stats():
    """A deliberate, documented exposure rather than a silent bug: `sys_query_history`
    stores the *verbatim* text a client sent — unlike `pg_stat_statements`, which Postgres
    has already parsed and re-serialised (identifiers folded to lowercase) before storing.
    Two executions that are semantically one statement, differing only in identifier case
    or in an attached comment, therefore fingerprint as two separate `QueryStat`s in
    `ingest()`. This is not "fixed" here — a general identifier case-fold cannot tell a
    case-insensitive unquoted identifier apart from a deliberately-quoted, case-sensitive
    one without risking folding away a real distinction — so this test pins the current,
    accepted behaviour rather than a silently-changed one. See `fetch_workload`'s docstring
    for the `cost_share`/`--min-cost-share` consequence this has.
    """
    querier = _canned(
        {
            CAP_WORKLOAD: [
                ("select id from orders where status = 'a'", 100_000),
                ("SELECT ID FROM ORDERS WHERE STATUS = 'b'", 100_000),
                ("/* app=foo */ select id from orders where status = 'c'", 100_000),
            ]
        }
    )
    fetch = RedshiftWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    workload = ingest(fetch, "redshift")
    # Three distinct QueryStats, not one — the collapse test above pins the case ingest()
    # *does* unify; this pins the case it deliberately does not.
    assert len(workload.stats) == 3


def test_a_null_elapsed_time_does_not_crash_the_whole_run():
    """A malformed or NULL `elapsed_time` must cost this run nothing more than one row,
    never the whole `fetch_workload` call — the same "one missing grant, one capability"
    guarantee `_run` gives a denied statement, which is worthless if a single bad row can
    still take the run down one level higher up. Coerced defensively the way
    `postgres.py`'s `_as_float` call sites are, rather than raising a bare `TypeError` out
    of a generator expression no caller wraps in a try/except.
    """
    querier = _canned(
        {
            CAP_WORKLOAD: [
                ("select id from orders where status = 'a'", None),
                ("select id from customers where status = 'b'", 50_000),
            ]
        }
    )
    fetch = RedshiftWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert len(fetch.rows) == 2
    by_sql = {row.sql: row for row in fetch.rows}
    assert by_sql["select id from orders where status = 'a'"].total_time_ms == 0.0
    assert by_sql["select id from customers where status = 'b'"].total_time_ms == pytest.approx(
        50.0
    )


def test_fetch_schema_builds_a_sqlglot_schema_mapping():
    querier = _canned(
        {
            CAP_SCHEMA: [
                ("public", "orders", "id", "integer"),
                ("public", "orders", "status", "character varying"),
                ("public", "customers", "id", "integer"),
            ]
        }
    )
    schema = RedshiftWorkloadAdapter(querier=querier).fetch_schema(("public",))
    assert schema == {
        "public": {
            "orders": {"id": "integer", "status": "character varying"},
            "customers": {"id": "integer"},
        },
    }


def test_fetch_schema_is_nested_by_schema():
    rows = {
        CAP_SCHEMA: [
            ("sales", "orders", "id", "integer"),
            ("sales", "orders", "status", "text"),
            ("staging", "orders", "id", "integer"),
        ]
    }
    adapter = RedshiftWorkloadAdapter(querier=_canned(rows))
    assert adapter.fetch_schema(("sales", "staging")) == {
        "sales": {"orders": {"id": "integer", "status": "text"}},
        "staging": {"orders": {"id": "integer"}},
    }


def test_schema_rows_are_fetched_at_most_once_per_schema_tuple():
    """`_schema_cache`'s whole justification: `fetch_schema` and `fetch_table_facts` both
    need CAP_SCHEMA rows, and querying twice for the same `schemas` tuple would do twice
    the catalog work and — worse — record a denied grant in `degraded` twice for the same
    missing privilege. Pins the call count directly rather than trusting the docstring.
    """
    rows = {
        CAP_SCHEMA: [("public", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("public", "orders", 10, 1, 0.0, 0.0, "EVEN", "id", 0.0)],
    }
    querier = _canned(rows)
    adapter = RedshiftWorkloadAdapter(querier=querier)
    adapter.fetch_schema(("public",))
    adapter.fetch_table_facts(("public",), frozenset({Relation("public", "orders")}))
    schema_calls = [
        call for call in querier.calls if call[0] == RedshiftWorkloadAdapter.SQL[CAP_SCHEMA]
    ]
    assert len(schema_calls) == 1, "CAP_SCHEMA was queried more than once for the same schemas"

    # A second, distinct schema tuple is a real cache miss, not suppressed entirely.
    adapter.fetch_schema(("staging",))
    schema_calls = [
        call for call in querier.calls if call[0] == RedshiftWorkloadAdapter.SQL[CAP_SCHEMA]
    ]
    assert len(schema_calls) == 2


def test_table_facts_do_not_alias_across_schemas():
    rows = {
        CAP_SCHEMA: [("sales", "orders", "id", "integer"), ("staging", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [
            ("sales", "orders", 50_000, 1024, 5.0, 0.0, "EVEN", "id", 0.1),
            ("staging", "orders", 7, 1, 0.0, 0.0, "KEY(id)", "id", 0.0),
        ],
    }
    adapter = RedshiftWorkloadAdapter(querier=_canned(rows))
    facts = adapter.fetch_table_facts(
        ("sales", "staging"),
        frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    assert facts[Relation("sales", "orders")].row_estimate == 50_000
    assert facts[Relation("staging", "orders")].row_estimate == 7
    assert facts[Relation("sales", "orders")].size_bytes == 1024 * 1024 * 1024
    assert facts[Relation("staging", "orders")].size_bytes == 1 * 1024 * 1024


def test_fetch_table_facts_does_not_leak_a_same_named_table_from_another_schema():
    """Same over-fetch guard `PostgresWorkloadAdapter.fetch_table_facts` documents: the
    table parameter is bare names, so a same-named table in a schema that was requested but
    is not itself in `relations` can come back too. It must not appear in the result.
    """
    rows = {
        CAP_SCHEMA: [("sales", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [
            ("sales", "orders", 100, 1, 0.0, 0.0, "EVEN", "id", 0.0),
            ("staging", "orders", 9_999, 50, 0.0, 0.0, "EVEN", "id", 0.0),
        ],
    }
    adapter = RedshiftWorkloadAdapter(querier=_canned(rows))
    facts = adapter.fetch_table_facts(
        ("sales", "staging"), frozenset({Relation("sales", "orders")})
    )
    assert list(facts) == [Relation("sales", "orders")]


def test_a_null_tbl_rows_reads_as_an_unknown_row_estimate():
    """Sentinel 1: `tbl_rows` itself coming back SQL NULL — the row was never populated at
    all — must read as unknown, not as zero."""
    rows = {
        CAP_SCHEMA: [("public", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("public", "orders", None, 100, 0.0, 0.0, "EVEN", "id", 0.0)],
    }
    facts = RedshiftWorkloadAdapter(querier=_canned(rows)).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate is None
    # size is a separate sentinel (see below) and must be unaffected by this one.
    assert facts[Relation("public", "orders")].size_bytes == 100 * 1024 * 1024


def test_a_null_size_reads_as_an_unknown_size_bytes():
    """Sentinel 2: `size` coming back SQL NULL must read as unknown, independently of
    `tbl_rows`."""
    rows = {
        CAP_SCHEMA: [("public", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("public", "orders", 500, None, 0.0, 0.0, "EVEN", "id", 0.0)],
    }
    facts = RedshiftWorkloadAdapter(querier=_canned(rows)).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].size_bytes is None
    assert facts[Relation("public", "orders")].row_estimate == 500


def test_stats_off_100_does_not_suppress_a_real_row_estimate_or_size():
    """The inverted-premise bug a review caught in this module's first version: `stats_off`
    is a 0-100 *staleness percentage* for planner statistics, not a "never analyzed" flag —
    AWS documents `tbl_rows` and `size` as physical facts about the table, not values
    ANALYZE produces. Gating them on `stats_off = 100` discarded accurate facts for a
    merely-stale table and then claimed the row count "could not be checked" when it
    plainly could — the opposite of Redshift's answer to Postgres's genuine
    `pg_class.reltuples = -1` sentinel. `stats_off` is still real evidence (see
    `test_physical_facts_are_stashed_on_the_adapter_keyed_by_relation`), just not a reason
    to null out these two columns.
    """
    rows = {
        CAP_SCHEMA: [("public", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("public", "orders", 3, 1, 0.0, 100.0, "EVEN", "id", 0.0)],
    }
    facts = RedshiftWorkloadAdapter(querier=_canned(rows)).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate == 3
    assert facts[Relation("public", "orders")].size_bytes == 1 * 1024 * 1024


def test_a_null_stats_off_does_not_suppress_a_real_row_estimate():
    """`stats_off` itself coming back NULL — its staleness is simply unknown — must not
    suppress `tbl_rows`/`size` either, for the same reason a *known* `stats_off` no longer
    does: those two columns were never derived from ANALYZE in the first place, so nothing
    about `stats_off` — known, unknown, or maximally stale — is a reason to null them out.
    """
    rows = {
        CAP_SCHEMA: [("public", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("public", "orders", 42, 7, None, None, None, None, None)],
    }
    facts = RedshiftWorkloadAdapter(querier=_canned(rows)).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate == 42
    assert facts[Relation("public", "orders")].size_bytes == 7 * 1024 * 1024


def test_physical_facts_are_stashed_on_the_adapter_keyed_by_relation():
    rows = {
        CAP_SCHEMA: [("public", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [
            ("public", "orders", 1000, 10, 12.5, 3.0, "KEY(customer_id)", "created_at", 0.4)
        ],
    }
    adapter = RedshiftWorkloadAdapter(querier=_canned(rows))
    adapter.fetch_table_facts(("public",), frozenset({Relation("public", "orders")}))
    physical = adapter.physical_facts[Relation("public", "orders")]
    assert physical.unsorted == 12.5
    assert physical.stats_off == 3.0
    assert physical.diststyle == "KEY(customer_id)"
    assert physical.sortkey1 == "created_at"
    assert physical.skew_rows == 0.4


def test_a_relation_absent_from_svv_table_info_is_absent_from_the_facts_dict():
    """`svv_columns` carries external (Spectrum) tables; `svv_table_info` does not. A
    Spectrum relation must be simply missing from both results — not present with every
    field forced to `None` — so a later task's SORTKEY/DISTKEY rules can tell
    "structurally cannot have one" apart from "not analysed yet." See
    `fetch_table_facts`'s docstring.
    """
    rows = {
        CAP_SCHEMA: [
            ("public", "orders", "id", "integer"),
            ("spectrum", "events", "id", "integer"),
        ],
        CAP_TABLE_FACTS: [("public", "orders", 1000, 10, 0.0, 0.0, "EVEN", "id", 0.0)],
    }
    adapter = RedshiftWorkloadAdapter(querier=_canned(rows))
    facts = adapter.fetch_table_facts(
        ("public", "spectrum"),
        frozenset({Relation("public", "orders"), Relation("spectrum", "events")}),
    )
    assert Relation("public", "orders") in facts
    assert Relation("spectrum", "events") not in facts
    assert Relation("spectrum", "events") not in adapter.physical_facts
    # The columns are still there for qualification purposes (see fetch_schema).
    schema = adapter.fetch_schema(("public", "spectrum"))
    assert "events" in schema["spectrum"]


def _select_list(sql: str) -> str:
    """The text between `SELECT` and the first `FROM`. See the identical helper in
    `tests/test_workload_postgres.py`."""
    match = re.search(r"select\s+(.*?)\s+from\b", sql, re.IGNORECASE | re.DOTALL)
    assert match, f"no SELECT ... FROM found in statement: {sql!r}"
    return match.group(1)


def _select_list_columns(sql: str) -> list[str]:
    """The SELECT list's column names, in the order the *statement* lists them.

    Double quotes are stripped: `svv_table_info`'s `"schema"` and `"table"` are reserved
    words that must stay quoted in the SQL, but the name being pinned is the same either
    way. A comma nested inside parentheses does not split a column — none of today's
    select lists has one, but a naive `split(",")` would silently miscount if one ever
    appeared.
    """
    depth = 0
    names = [""]
    for ch in _select_list(sql):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            names.append("")
        else:
            names[-1] += ch
    return [name.strip().strip('"') for name in names]


#: One canned value per SELECT-list column *name*, per capability, deliberately
#: distinguishable from every other value in the same row — including from the ones of the
#: same SQL type, which is the whole point (see
#: `test_select_list_columns_land_in_the_field_their_consumer_reads`). `tuple(range(width))`
#: was positionally indistinguishable, so it pinned the column *count* and not their
#: *positions*, and swapping two same-typed columns in the SQL text left the suite green.
_CANNED_COLUMN_VALUES: dict[str, dict[str, object]] = {
    CAP_WORKLOAD: {
        "query_text": "select id from stand_in where status = 'x'",
        "elapsed_time": 7_000,
    },
    CAP_SCHEMA: {
        "schema_name": "sch",
        "table_name": "tbl",
        "column_name": "col",
        "data_type": "integer",
    },
    CAP_TABLE_FACTS: {
        "schema": "sch",
        "table": "tbl",
        "tbl_rows": 33,
        "size": 44,
        "unsorted": 11.0,
        "stats_off": 22.0,
        "diststyle": "KEY(dk_col)",
        "sortkey1": "sk_col",
        "skew_rows": 55.0,
    },
    CAP_ADVISOR: {
        "database_name": "db",
        "schema_name": "sch",
        "table_name": "tbl",
        "type": "sort key",
        "current_ddl": "CURRENT: ALTER TABLE x;",
        "recommended_ddl": "RECOMMENDED: ALTER TABLE y;",
    },
}

_CANNED_RELATION = Relation(schema="sch", table="tbl")


def _positional_row(capability: str) -> tuple[object, ...]:
    """One row for `capability`, ordered by the *statement's own* SELECT list.

    Values are keyed by column name and then placed in the order the SQL text puts those
    names in, which is what makes a swap of two columns in the statement change what the
    consumer receives: the value for `stats_off` moves to `unsorted`'s position, and the
    assertions below then read `unsorted == 22.0` instead of `11.0`.
    """
    values = _CANNED_COLUMN_VALUES[capability]
    columns = _select_list_columns(RedshiftWorkloadAdapter.SQL[capability])
    assert sorted(columns) == sorted(values), (
        f"{capability}'s SELECT list is {columns}, which no longer matches the columns this "
        f"test pins ({sorted(values)}) — add the new column and assert where it lands, "
        "rather than widening the fixture until it stops complaining"
    )
    return tuple(values[column] for column in columns)


def _check_workload_columns(adapter: RedshiftWorkloadAdapter) -> None:
    (row,) = adapter.fetch_workload(None, 10).rows
    assert row.sql == "select id from stand_in where status = 'x'"
    # elapsed_time is microseconds; total_time_ms is milliseconds.
    assert row.total_time_ms == pytest.approx(7.0)


def _check_schema_columns(adapter: RedshiftWorkloadAdapter) -> None:
    assert adapter.fetch_schema(("sch",)) == {"sch": {"tbl": {"col": "integer"}}}


def _check_table_facts_columns(adapter: RedshiftWorkloadAdapter) -> None:
    facts = adapter.fetch_table_facts(("sch",), frozenset({_CANNED_RELATION}))
    assert facts[_CANNED_RELATION].row_estimate == 33
    assert facts[_CANNED_RELATION].size_bytes == 44 * 1024 * 1024
    physical = adapter.physical_facts[_CANNED_RELATION]
    # `unsorted` and `stats_off` are both 0-100 float percentages, so nothing but this
    # assertion can tell them apart — and swapping them inverts ADV104's remediation
    # (VACUUM where ANALYZE was needed) at the one confidence rung this adapter reaches
    # HIGH on. Same for the `diststyle`/`sortkey1` pair, which are both text and gate
    # ADV101's and ADV102's suppression.
    assert physical.unsorted == 11.0
    assert physical.stats_off == 22.0
    assert physical.diststyle == "KEY(dk_col)"
    assert physical.sortkey1 == "sk_col"
    assert physical.skew_rows == 55.0


def _check_advisor_columns(adapter: RedshiftWorkloadAdapter) -> None:
    (row,) = adapter._advisor_rows(("sch",), frozenset({_CANNED_RELATION}))
    assert row.relation == _CANNED_RELATION
    assert row.rec_type == "sort key"
    assert row.current_ddl == "CURRENT: ALTER TABLE x;"
    assert row.recommended_ddl == "RECOMMENDED: ALTER TABLE y;"


@pytest.mark.parametrize(
    ("capability", "check"),
    [
        (CAP_WORKLOAD, _check_workload_columns),
        (CAP_SCHEMA, _check_schema_columns),
        (CAP_TABLE_FACTS, _check_table_facts_columns),
        (CAP_ADVISOR, _check_advisor_columns),
    ],
    ids=["workload", "schema", "table_facts", "advisor"],
)
def test_select_list_columns_land_in_the_field_their_consumer_reads(capability, check):
    """Every statement's SELECT list is pinned by *position*, not merely by arity.

    Batch 2 shipped a column-count mismatch between a statement's SELECT list and its
    Python unpacking that no fixture caught, and the guard added for it derived only the row
    *width* from the SQL text (`tuple(range(width))`) — so it pinned the count while leaving
    every column's position unverified. Swapping two same-typed columns in the statement
    left the whole suite green: `unsorted`/`stats_off` inverts ADV104's remediation for
    every table on the cluster, `diststyle`/`sortkey1` inverts ADV101's and ADV102's
    suppression gates, `table_name`/`column_name` mis-keys the schema map, and
    `query_text`/`elapsed_time` mis-reads the whole workload.

    The fixture row is built by looking each canned value up *by column name* and then
    placing it at the position the SQL text gives that name — so a swap in the SQL moves
    the values with it and the assertions below read the wrong field. Arity is still pinned
    too, and by name rather than by number: `_positional_row` requires the statement's
    SELECT list to be exactly the set of columns this test knows where to expect.

    One parametrized case per capability, including `CAP_ADVISOR` — whose consumer is
    `_advisor_rows` — so a mismatch in one cannot hide behind the other three passing.
    """
    querier = _canned({capability: [_positional_row(capability)]})
    check(RedshiftWorkloadAdapter(querier=querier))


class _FakeCursor:
    """Enough of a psycopg cursor for connect()'s session-setup statements.

    `fail_on` lets a test make exactly one statement (identified by its SQL text) raise,
    which is how the read-only degradation path is exercised without a live cluster: a
    real Redshift refusing `SET default_transaction_read_only` looks, from here, like a
    cursor.execute() that raises on that one statement and succeeds on every other.
    """

    def __init__(
        self,
        log: list[tuple] | None = None,
        *,
        fail_on: frozenset[str] = frozenset(),
        fail_message: str = "ERROR: {sql!r} is not supported on this cluster",
    ):
        self.executed: list[tuple] = []
        self._log = log if log is not None else []
        self._fail_on = fail_on
        self._fail_message = fail_message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self._log.append((sql, params))
        if sql in self._fail_on:
            raise RuntimeError(self._fail_message.format(sql=sql))

    def fetchall(self):
        return []


class _FakeConnection:
    def __init__(
        self,
        *,
        fail_on: frozenset[str] = frozenset(),
        fail_message: str = "ERROR: {sql!r} is not supported on this cluster",
    ) -> None:
        self.cursors: list[_FakeCursor] = []
        self.log: list[tuple] = []
        self._fail_on = fail_on
        self._fail_message = fail_message

    def cursor(self):
        cursor = _FakeCursor(self.log, fail_on=self._fail_on, fail_message=self._fail_message)
        self.cursors.append(cursor)
        return cursor


def _install_fake_psycopg(
    monkeypatch,
    seen: dict,
    *,
    fail_on: frozenset[str] = frozenset(),
    fail_message: str = "ERROR: {sql!r} is not supported on this cluster",
):
    """A psycopg that records the conninfo it was handed and connects successfully."""

    module = types.ModuleType("psycopg")

    def connect(conninfo, **kwargs):
        seen["conninfo"] = conninfo
        seen["connection"] = _FakeConnection(fail_on=fail_on, fail_message=fail_message)
        return seen["connection"]

    module.connect = connect  # type: ignore[attr-defined]
    module.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return module


def test_connect_without_psycopg_installed_raises_a_helpful_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    adapter = RedshiftWorkloadAdapter()
    params = ConnectionParams(
        engine="redshift", dsn="postgresql://u@h/db", fields={}, source="--dsn"
    )
    with pytest.raises(ImportError) as exc:
        adapter.connect(params, 30)
    assert "sqlquality[warehouse]" in str(exc.value)
    # Names the calling engine, not merely the extra: `import_psycopg("Redshift", ...)`
    # could be miscopied to `import_psycopg("Postgres", ...)` at the Redshift call site
    # and every existing assertion here would still pass.
    assert "Redshift" in str(exc.value)
    assert "Postgres" not in str(exc.value)


def test_connect_arms_a_statement_timeout_before_the_querier_is_usable(monkeypatch):
    """Mirrors the Postgres unit test of the same shape: session setup must precede any
    later query, and the relative order is only observable on the connection-wide log."""
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    adapter = RedshiftWorkloadAdapter()
    adapter.connect(
        ConnectionParams(engine="redshift", dsn="postgresql:///x", fields={}, source="--dsn"), 30
    )

    setup = seen["connection"].log[:]
    assert setup == [
        (READ_ONLY_SQL, None),
        ("SELECT set_config('statement_timeout', %s, false)", ("30000ms",)),
    ], setup
    assert adapter.degraded == []

    adapter._query("SELECT 1", ())
    assert seen["connection"].log[:2] == setup
    assert seen["connection"].log[2] == ("SELECT 1", ())


def test_an_out_of_range_timeout_is_clamped_before_it_reaches_the_session(monkeypatch):
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    RedshiftWorkloadAdapter().connect(
        ConnectionParams(engine="redshift", dsn="postgresql:///x", fields={}, source="--dsn"), 7200
    )
    assert seen["connection"].log[1][1] == (f"{MAX_TIMEOUT_S * 1000}ms",)


def test_a_refused_read_only_statement_degrades_rather_than_aborts(monkeypatch):
    """The Redshift-specific difference from Postgres: `SET default_transaction_read_only`
    is not accepted in every configuration. A refusal must not be silently treated as
    success — the whole "sqlquality never writes" promise rests on the operator being
    told, not on the tool assuming the best."""
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen, fail_on=frozenset({READ_ONLY_SQL}))
    adapter = RedshiftWorkloadAdapter()
    adapter.connect(
        ConnectionParams(engine="redshift", dsn="postgresql:///x", fields={}, source="--dsn"), 30
    )

    # The connection still succeeds and the statement timeout is still armed —
    # a refused read-only guard is not a reason to abandon the rest of setup.
    assert adapter._query is not None
    assert seen["connection"].log[1] == (
        "SELECT set_config('statement_timeout', %s, false)",
        ("30000ms",),
    )

    assert len(adapter.degraded) == 1
    capability, reason = adapter.degraded[0]
    assert capability == DEGRADATION_READ_ONLY
    assert "could not be proven read-only" in reason
    # Not "we might write" — the four SELECT-only statements this adapter issues are a
    # separate, already-pinned guarantee (test_no_statement_writes). What is missing is
    # the extra belt-and-braces defense, and the message must say which.
    assert "belt-and-braces" in reason.lower()


def test_the_read_only_degradation_survives_past_fetch_workload(monkeypatch):
    """A carried-forward item from Task 2: the read-only degradation above was recorded
    correctly, but it could never reach a user, because `cli.py` calls `fetch_workload()`
    immediately after `connect()` and that call raised `NotImplementedError` — an unhandled
    exception that crashed the whole run before it ever reached the loop that prints
    `adapter.degraded` to stderr. `fetch_workload` is now a real method, so that call
    succeeds instead of raising, and `degraded` survives to be printed later.

    This does not exercise `cli.py` end to end — `propose()` is still `NotImplementedError`
    until Task 5/6 builds it, so a full `advise` run cannot complete yet — but it proves
    the specific failure this task closes: the read-only warning is no longer lost between
    `connect()` and the rest of the run.
    """
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen, fail_on=frozenset({READ_ONLY_SQL}))
    adapter = RedshiftWorkloadAdapter()
    adapter.connect(
        ConnectionParams(engine="redshift", dsn="postgresql:///x", fields={}, source="--dsn"), 30
    )
    assert len(adapter.degraded) == 1  # the read-only degradation recorded by connect()

    fetch = adapter.fetch_workload(None, 10)  # must not raise, and must not touch degraded
    assert fetch.rows == ()
    assert len(adapter.degraded) == 1
    assert adapter.degraded[0][0] == DEGRADATION_READ_ONLY


def test_the_read_only_degradation_message_is_scrubbed(monkeypatch):
    """The one path that puts raw driver text into user-facing output.

    `self.degraded` is exactly what `cli.py` prints to stderr and embeds in the JSON and
    markdown reports, so a secret reaching this message is a real leak, not a theoretical
    one — unlike the connect-failure path, which is at least caught by a `ConnectionError`
    the caller might choose not to print. The fake driver's refusal is made to quote the
    password verbatim, the way a permission-denied message naming the failed session
    setting sometimes echoes surrounding context; `scrub()` must still remove it.
    """
    seen: dict = {}
    _install_fake_psycopg(
        monkeypatch,
        seen,
        fail_on=frozenset({READ_ONLY_SQL}),
        fail_message="ERROR: {sql!r} refused for connection password=hunter2",
    )
    adapter = RedshiftWorkloadAdapter()
    adapter.connect(
        ConnectionParams(
            engine="redshift",
            dsn=None,
            fields={"host": "db", "user": "hans", "password": "hunter2"},
            source="profiles.yml",
        ),
        30,
    )
    assert len(adapter.degraded) == 1
    _capability, reason = adapter.degraded[0]
    assert "hunter2" not in reason
    assert "***" in reason


def test_a_successful_read_only_statement_reports_no_degradation(monkeypatch):
    """Guards the guard above: without a forced failure, the same setup reports nothing,
    so the degradation in the previous test is attributable to the refusal, not to
    `connect()` always degrading regardless of what the driver does."""
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    adapter = RedshiftWorkloadAdapter()
    adapter.connect(
        ConnectionParams(engine="redshift", dsn="postgresql:///x", fields={}, source="--dsn"), 30
    )
    assert adapter.degraded == []


def test_connect_scrubs_a_password_from_a_driver_failure(monkeypatch):
    """psycopg is not believed to echo a password, but the auth-failure path cannot be
    exercised without a live server, so the secret is scrubbed rather than trusted —
    the same reasoning `test_workload_postgres.py`'s identical test gives."""
    fake_psycopg = types.ModuleType("psycopg")

    def explode(conninfo, **kwargs):
        raise RuntimeError(f"connection failed for conninfo {conninfo}")

    fake_psycopg.connect = explode  # type: ignore[attr-defined]
    fake_psycopg.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    params = ConnectionParams(
        engine="redshift",
        dsn=None,
        fields={"host": "db", "user": "hans", "password": "hunter2"},
        source="profiles.yml",
    )
    with pytest.raises(ConnectionError) as exc:
        RedshiftWorkloadAdapter().connect(params, 30)
    assert "hunter2" not in str(exc.value)
    assert "***" in str(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


def test_a_conninfo_build_failure_is_scrubbed_like_a_connect_failure(monkeypatch):
    """make_conninfo runs inside the scrubbing envelope: it can itself raise on an
    unusable keyword, and that message can quote the offending value."""

    module = types.ModuleType("psycopg")

    def never_called(conninfo, **kwargs):
        raise AssertionError("must not reach connect() when the conninfo cannot be built")

    def explode(**kwargs):
        raise RuntimeError(f"invalid connection option: {sorted(kwargs.items())}")

    module.connect = never_called  # type: ignore[attr-defined]
    module.conninfo = types.SimpleNamespace(make_conninfo=explode)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", module)

    params = ConnectionParams(
        engine="redshift",
        dsn=None,
        fields={"host": "db", "user": "hans", "password": "hunter2"},
        source="profiles.yml",
    )
    with pytest.raises(ConnectionError) as exc:
        RedshiftWorkloadAdapter().connect(params, 30)
    assert "hunter2" not in str(exc.value)
    assert "***" in str(exc.value)
    assert exc.value.__context__ is None


def test_dropped_profile_keys_are_named_on_stderr(monkeypatch, capsys):
    """A key we cannot forward must be reported, not discarded in silence."""
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="redshift",
        dsn=None,
        fields={
            "host": "db",
            "user": "hans",
            "password": "hunter2",
            "cluster_identifier": "my-cluster",
            "iam": "true",
        },
        source="profiles.yml",
    )
    RedshiftWorkloadAdapter().connect(params, 30)
    warning = capsys.readouterr().err
    assert "cluster_identifier" in warning
    assert "iam" in warning
    # Key names only — never a value, since one of them could be a secret.
    assert "hunter2" not in warning
    assert "my-cluster" not in warning


def test_forwarded_and_mapped_keys_are_not_reported_as_dropped(monkeypatch, capsys):
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="redshift",
        dsn=None,
        fields={"host": "db", "dbname": "x", "user": "u", "password": "p", "sslmode": "require"},
        source="profiles.yml",
    )
    RedshiftWorkloadAdapter().connect(params, 30)
    assert capsys.readouterr().err == ""


def test_profile_fields_are_translated_and_forwarded_to_the_driver(monkeypatch):
    """Pins the actual conninfo content, not just the dropped-keys warning.

    A previous version of this suite recorded `seen["conninfo"]` and never asserted on
    it, so scrambling the field map's targets, dropping the `database`/`username`
    aliases, or cutting the TLS passthrough set down to just `sslmode` all left the whole
    suite green. This test mirrors Postgres's
    `test_profile_tls_settings_are_forwarded_to_the_driver` for exactly that reason: the
    TLS group is not cosmetic — a profile saying `sslmode: verify-full` that silently
    connects under libpq's default `prefer` performs no certificate verification at all,
    and the user is never told.
    """
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="redshift",
        dsn=None,
        fields={
            "host": "db",
            "database": "mydb",  # alias for dbname
            "username": "hans",  # alias for user
            "password": "hunter2",
            "sslmode": "verify-full",
            "sslrootcert": "/etc/ssl/ca.crt",
            "sslcert": "/etc/ssl/client.crt",
            "sslkey": "/etc/ssl/client.key",
            "connect_timeout": "10",
        },
        source="profiles.yml",
    )
    RedshiftWorkloadAdapter().connect(params, 30)
    conninfo = seen["conninfo"]
    assert "host=db" in conninfo
    assert "dbname=mydb" in conninfo
    assert "user=hans" in conninfo
    assert "password=hunter2" in conninfo
    assert "sslmode=verify-full" in conninfo
    assert "sslrootcert=/etc/ssl/ca.crt" in conninfo
    assert "sslcert=/etc/ssl/client.crt" in conninfo
    assert "sslkey=/etc/ssl/client.key" in conninfo
    assert "connect_timeout=10" in conninfo
