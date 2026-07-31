import re
import sys
import types
from datetime import timedelta

import sqlglot
import pytest

from sqlquality.models import Aggregation, ConnectionParams, Relation, Workload
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


def test_there_is_no_ndv_or_index_capability():
    """Redshift exposes no `pg_stats.n_distinct` equivalent and has no indexes.

    Declaring either capability would invite a rule to assume evidence that cannot exist.
    """
    capabilities = {s.capability for s in RedshiftWorkloadAdapter().introspection_sql()}
    assert not any("ndv" in c or "index" in c for c in capabilities)


#: Every `WorkloadAdapter` method this task deliberately leaves unbuilt, with a call that
#: reaches it. Named individually rather than discovered by reflection: a later task that
#: implements one of these must delete its entry here, which is a visible, reviewable edit —
#: whereas a reflective sweep would silently stop covering whatever got implemented.
#:
#: `connect` is deliberately absent: Task 2 implements it (see the tests below) because
#: Redshift speaks the same PostgreSQL wire protocol Postgres does, so it is the one
#: method genuinely exercisable without a live Redshift cluster.
UNIMPLEMENTED = {
    "propose": lambda a: a.propose(
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        {},
        Workload(stats=(), window_description="w"),
        min_cost_share=0.01,
    ),
    "render_ddl": lambda a: a.render_ddl([]),
}


@pytest.mark.parametrize("method", sorted(UNIMPLEMENTED))
def test_unimplemented_methods_say_so_rather_than_returning_empty(method):
    """A half-built adapter that returns nothing looks exactly like a healthy cluster with
    no workload, which is the worst possible failure mode for this command.

    Every unbuilt method is covered, not just one. Task 1 originally pinned `fetch_schema`
    alone, which would have let a later task implement `fetch_workload` and silently leave
    `fetch_table_facts` returning `[]` — the run would then report a healthy cluster with no
    catalog facts rather than an unfinished adapter.
    """
    adapter = RedshiftWorkloadAdapter()
    with pytest.raises(NotImplementedError):
        UNIMPLEMENTED[method](adapter)


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


def test_stats_off_100_reads_tbl_rows_and_size_as_unknown_despite_present_values():
    """Sentinel 3, the central lesson this task exists to apply: Redshift's own version of
    `pg_class.reltuples = -1`. At `stats_off = 100` — statistics never refreshed by
    ANALYZE — `tbl_rows`/`size` are non-NULL but meaningless; reading them as facts is
    exactly what silently suppressed every Postgres proposal for a freshly-loaded table.
    """
    rows = {
        CAP_SCHEMA: [("public", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("public", "orders", 3, 1, 0.0, 100.0, "EVEN", "id", 0.0)],
    }
    facts = RedshiftWorkloadAdapter(querier=_canned(rows)).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate is None
    assert facts[Relation("public", "orders")].size_bytes is None


def test_a_null_stats_off_does_not_suppress_a_real_row_estimate():
    """The control for sentinel 3: `stats_off` itself coming back NULL is unknown
    staleness, not proven-stale, and must not be treated as "never analyzed" — otherwise
    every table whose staleness this adapter cannot see would silently lose its row
    estimate and size too.
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


def _select_list_arity(sql: str) -> int:
    """Number of columns in a statement's SELECT list, ignoring a comma nested inside
    parentheses (none of today's statements have one in the select list, but a naive comma
    count would silently miscount one if it ever did)."""
    depth = 0
    arity = 1
    for ch in _select_list(sql):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            arity += 1
    return arity


def _dummy_row(width: int) -> tuple:
    """A row exactly as wide as the statement's own SELECT list, so unpacking it exercises
    the real arity rather than a fixture written to match the unpacking — see the module
    docstring's provenance warning."""
    return tuple(range(width))


@pytest.mark.parametrize(
    ("capability", "fetch"),
    [
        (CAP_WORKLOAD, lambda a: a.fetch_workload(None, 10)),
        (CAP_SCHEMA, lambda a: a.fetch_schema(("public",))),
        (CAP_TABLE_FACTS, lambda a: a.fetch_table_facts(("public",), frozenset())),
    ],
    ids=["workload", "schema", "table_facts"],
)
def test_select_list_arity_matches_its_consumers_unpacking(capability, fetch):
    """Batch 2 shipped a column-count mismatch between a statement's SELECT list and its
    Python unpacking that no fixture caught, because the fixture was written to match the
    unpacking rather than the statement. This derives the row width from the SQL text
    itself and feeds it through the real consumer method, so a future edit to either side
    that the other does not follow raises `ValueError` here — one parametrized case per
    capability, so a mismatch in one does not hide behind the other two passing.
    """
    width = _select_list_arity(RedshiftWorkloadAdapter.SQL[capability])
    querier = _canned({capability: [_dummy_row(width)]})
    fetch(RedshiftWorkloadAdapter(querier=querier))  # must not raise ValueError


def test_advisor_select_list_arity_is_pinned_for_its_future_consumer():
    """`CAP_ADVISOR` has no consumer yet — `propose()` is Task 5/6's job — so there is no
    unpacking to compare against today. This pins the SELECT list's current arity so
    whoever builds that consumer inherits a known, deliberate number instead of discovering
    a drift between the statement and their own unpacking after the fact.
    """
    assert _select_list_arity(RedshiftWorkloadAdapter.SQL[CAP_ADVISOR]) == 6


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
