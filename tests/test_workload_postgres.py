import re
from datetime import timedelta

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload import get_workload_adapter
from sqlquality.workload.postgres import (
    CAP_INDEXES,
    CAP_NDV,
    CAP_SCHEMA,
    CAP_STATS_RESET,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    PostgresWorkloadAdapter,
)

EXPECTED_CAPABILITIES = {
    CAP_WORKLOAD,
    CAP_STATS_RESET,
    CAP_SCHEMA,
    CAP_TABLE_FACTS,
    CAP_NDV,
    CAP_INDEXES,
}


def test_registry_returns_postgres_adapter():
    adapter = get_workload_adapter("postgres")
    assert adapter.engine == "postgres"


def test_registry_rejects_unsupported_engine_with_a_useful_message():
    with pytest.raises(ValueError) as exc:
        get_workload_adapter("duckdb")
    assert "duckdb" in str(exc.value)


def test_introspection_statements_are_named_and_carry_privilege_hints():
    statements = get_workload_adapter("postgres").introspection_sql()
    assert statements, "adapter must declare its introspection statements for --dry-run"
    capabilities = {s.capability for s in statements}
    assert "workload" in capabilities
    assert all(s.privilege_hint for s in statements)


def test_every_capability_has_a_statement_and_a_hint():
    statements = PostgresWorkloadAdapter().introspection_sql()
    assert {s.capability for s in statements} == EXPECTED_CAPABILITIES
    for statement in statements:
        assert statement.sql.strip()
        assert statement.privilege_hint.strip()


#: Write verbs that must never appear in an introspection statement.
FORBIDDEN_VERBS = (
    "insert",
    "update",
    "delete",
    "create",
    "drop",
    "alter",
    "truncate",
    "grant",
    "revoke",
)


def _write_verbs_in(sql: str) -> set[str]:
    """Write verbs appearing as whole words in ``sql``.

    Word-boundary matching, not whitespace tokenizing. Two earlier attempts at this guard
    each left a hole: ``f" {verb} " in f" {sql.lower()} "`` misses a verb after a newline
    or at the statement start, and collapsing whitespace first still misses one glued to
    punctuation (``";drop table foo"``, ``"(delete from t)"``). ``\\b`` covers every
    position while still leaving ``created_at``, ``deleted`` and ``pg_stat_user_indexes``
    alone, since a following word character means no boundary.
    """
    lowered = sql.lower()
    return {verb for verb in FORBIDDEN_VERBS if re.search(rf"\b{verb}\b", lowered)}


def test_the_write_verb_detector_catches_every_adjacency():
    """Guard the guard. Each case below defeated an earlier version of this detector."""
    # Whitespace-separated — caught by every version.
    assert _write_verbs_in("select 1 from t; drop table foo") == {"drop"}
    # Newline, tab, and leading position — defeated the space-padded version.
    assert _write_verbs_in("select 1 from t limit 1;\ndrop table foo") == {"drop"}
    assert _write_verbs_in("select 1\n\tgrant select on x to y") == {"grant"}
    assert _write_verbs_in("delete from t") == {"delete"}
    # Punctuation-adjacent — defeated the whitespace-collapsing version too.
    assert _write_verbs_in("select 1;drop table foo") == {"drop"}
    assert _write_verbs_in("select(delete from t)") == {"delete"}


def test_the_write_verb_detector_does_not_fire_on_ordinary_identifiers():
    """A guard that flags `created_at` gets weakened to shut it up, so it must not."""
    assert _write_verbs_in("select created_at, updated_at from t") == set()
    assert _write_verbs_in("select deleted, insertion_id from pg_stat_user_indexes") == set()
    assert _write_verbs_in("select 1 from t where a = 2") == set()


def test_no_introspection_statement_writes():
    for statement in PostgresWorkloadAdapter().introspection_sql():
        found = _write_verbs_in(statement.sql)
        assert not found, f"{statement.capability} contains write verb(s): {sorted(found)}"


def test_workload_statement_is_scoped_to_the_current_database():
    sql = PostgresWorkloadAdapter().SQL[CAP_WORKLOAD].lower()
    assert "pg_stat_statements" in sql
    assert "current_database()" in sql
    assert "order by" in sql and "limit" in sql


class FakeQuerier:
    """Returns canned rows per capability, keyed by a distinctive SQL substring."""

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


def test_fetch_workload_maps_rows_and_reports_the_window():
    querier = FakeQuerier(
        {
            "pg_stat_statements": [("select id from orders where status = $1", 10, 250.0, 100)],
            "pg_stat_database": [("2026-07-01 00:00:00",)],
        }
    )
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert fetch.rows[0].sql == "select id from orders where status = $1"
    assert fetch.rows[0].calls == 10
    assert fetch.rows[0].total_time_ms == 250.0
    assert "2026-07-01" in fetch.window_description


def test_fetch_workload_window_is_honest_that_since_is_not_supported():
    querier = FakeQuerier(
        {
            "pg_stat_statements": [],
            "pg_stat_database": [("2026-07-01 00:00:00",)],
        }
    )
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(timedelta(days=7), 500)
    assert "since stats reset" in fetch.window_description.lower()


def test_fetch_schema_builds_a_sqlglot_schema_mapping():
    querier = FakeQuerier(
        {
            "information_schema.columns": [
                ("orders", "id", "integer"),
                ("orders", "status", "text"),
                ("customers", "id", "integer"),
            ]
        }
    )
    schema = PostgresWorkloadAdapter(querier=querier).fetch_schema(("public",))
    assert schema == {
        "orders": {"id": "integer", "status": "text"},
        "customers": {"id": "integer"},
    }


def test_fetch_table_facts_resolves_negative_n_distinct_as_a_row_fraction():
    querier = FakeQuerier(
        {
            "pg_total_relation_size": [("orders", 1000, 8192)],
            "information_schema.columns": [("orders", "id", "integer"), ("orders", "s", "text")],
            "pg_stats": [("orders", "id", 500.0), ("orders", "s", -0.25)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].row_estimate == 1000
    assert facts["orders"].ndv["id"] == 500.0
    # -0.25 means "a quarter of the rows are distinct"
    assert facts["orders"].ndv["s"] == 250.0


def test_negative_n_distinct_without_a_row_count_is_omitted_not_zeroed():
    """A negative n_distinct is a *fraction*, so it needs the row count to mean anything.

    The two facts come from different statements, so different privileges can hide one and
    not the other. Defaulting the missing row estimate to 0 would fabricate "zero distinct
    values" — a confident, wrong LOW-confidence signal for every proposal on the table.
    """
    querier = FakeQuerier(
        {
            "information_schema.columns": [("orders", "id", "integer")],
            "pg_stats": [("orders", "id", -0.25)],
            # No pg_total_relation_size rows: the row count is unknown.
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].row_estimate is None
    assert "id" not in facts["orders"].ndv


def test_absolute_n_distinct_survives_a_missing_row_count():
    """A positive n_distinct is an absolute count and needs no row estimate."""
    querier = FakeQuerier(
        {
            "information_schema.columns": [("orders", "id", "integer")],
            "pg_stats": [("orders", "id", 500.0)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].ndv["id"] == 500.0


def test_fetch_indexes_restores_column_order_from_ordinality():
    """Rows arriving out of order must still yield the right composite order.

    The statement does ORDER BY k.ordinality, but composite column order decides whether a
    proposal is correct, and a fixture that pre-sorts its rows cannot catch a regression
    here. Feeding them backwards is the only way to test the property.
    """
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "orders",
                    "idx_status_created",
                    "created_at",
                    2,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
                (
                    "orders",
                    "idx_status_created",
                    "status",
                    1,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )
    assert indexes["orders"][0].columns == ("status", "created_at")


def test_connect_scrubs_a_password_from_a_driver_failure(monkeypatch):
    """A driver exception is where this class of leak hides — see Task 6.

    psycopg is not believed to echo a password, but the auth-failure path cannot be
    exercised without a live server, so the secret is scrubbed rather than trusted.
    """
    import sys
    import types

    fake_psycopg = types.ModuleType("psycopg")

    def explode(conninfo, **kwargs):
        raise RuntimeError(f"connection failed for conninfo {conninfo}")

    fake_psycopg.connect = explode  # type: ignore[attr-defined]
    fake_psycopg.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={"host": "db", "user": "hans", "password": "hunter2"},
        source="profiles.yml",
    )
    with pytest.raises(ConnectionError) as exc:
        PostgresWorkloadAdapter().connect(params, 30)
    assert "hunter2" not in str(exc.value)
    assert "***" in str(exc.value)
    # And the unscrubbed original must not be reachable through the chain.
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


def test_fetch_indexes_groups_columns_in_ordinal_order():
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "orders",
                    "orders_pkey",
                    "id",
                    1,
                    True,
                    True,
                    900,
                    4096,
                    False,
                    None,
                    False,
                    "CREATE UNIQUE INDEX orders_pkey ON orders (id)",
                ),
                (
                    "orders",
                    "idx_status_created",
                    "status",
                    1,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
                (
                    "orders",
                    "idx_status_created",
                    "created_at",
                    2,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )
    by_name = {i.name: i for i in indexes["orders"]}
    assert by_name["idx_status_created"].columns == ("status", "created_at")
    assert by_name["orders_pkey"].is_primary is True
    assert by_name["idx_status_created"].scans == 0


def test_a_denied_statement_degrades_and_names_the_privilege():
    """The fixture supplies real pg_stats rows *and* denies them.

    Both halves are necessary. `assert facts == {} or facts["orders"].ndv == {}` proved
    nothing before: `wanted` always populates facts["orders"], and with no pg_stats rows in
    the fixture `ndv == {}` held whether or not the statement was denied — removing
    `fail_markers` left the test passing. With rows present, the empty ndv is caused by the
    denial and by nothing else.
    """
    querier = FakeQuerier(
        {
            "information_schema.columns": [("orders", "id", "integer")],
            "pg_stats": [("orders", "id", 500.0)],
        },
        fail_markers=("pg_stats",),
    )
    adapter = PostgresWorkloadAdapter(querier=querier)
    facts = adapter.fetch_table_facts(("public",), frozenset({"orders"}))
    assert facts["orders"].ndv == {}
    assert any(cap == CAP_NDV for cap, _ in adapter.degraded)
    assert any("pg_stats" in reason for _, reason in adapter.degraded)


def test_the_denial_fixture_would_otherwise_have_returned_statistics():
    """Guards the guard above: without the denial the same fixture yields a non-empty ndv,
    so the emptiness there is attributable to the denial rather than to an empty fixture."""
    querier = FakeQuerier(
        {
            "information_schema.columns": [("orders", "id", "integer")],
            "pg_stats": [("orders", "id", 500.0)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].ndv == {"id": 500.0}


class _FakeCursor:
    """Enough of a psycopg cursor for connect()'s two session-setup statements."""

    def __init__(self) -> None:
        self.executed: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return []


class _FakeConnection:
    def __init__(self) -> None:
        self.cursors: list[_FakeCursor] = []

    def cursor(self):
        cursor = _FakeCursor()
        self.cursors.append(cursor)
        return cursor


def _install_fake_psycopg(monkeypatch, seen: dict):
    """A psycopg that records the conninfo it was handed and connects successfully."""
    import sys
    import types

    module = types.ModuleType("psycopg")

    def connect(conninfo, **kwargs):
        seen["conninfo"] = conninfo
        return _FakeConnection()

    module.connect = connect  # type: ignore[attr-defined]
    module.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return module


def test_profile_tls_settings_are_forwarded_to_the_driver(monkeypatch):
    """A profile asking for verify-full must not connect under libpq's default `prefer`.

    Silently downgrading certificate verification for a tool pitched as safe to point at
    production is the wrong default, and the user was never told.
    """
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={
            "host": "db",
            "user": "hans",
            "password": "hunter2",
            "sslmode": "verify-full",
            "sslrootcert": "/etc/ssl/ca.crt",
            "sslcert": "/etc/ssl/client.crt",
            "sslkey": "/etc/ssl/client.key",
            "connect_timeout": "10",
        },
        source="profiles.yml",
    )
    PostgresWorkloadAdapter().connect(params, 30)
    conninfo = seen["conninfo"]
    assert "sslmode=verify-full" in conninfo
    assert "sslrootcert=/etc/ssl/ca.crt" in conninfo
    assert "sslcert=/etc/ssl/client.crt" in conninfo
    assert "sslkey=/etc/ssl/client.key" in conninfo
    assert "connect_timeout=10" in conninfo


def test_dropped_profile_keys_are_named_on_stderr(monkeypatch, capsys):
    """A key we cannot forward must be reported, not discarded in silence."""
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={
            "host": "db",
            "user": "hans",
            "password": "hunter2",
            "search_path": "app",
            "threads": "4",
        },
        source="profiles.yml",
    )
    PostgresWorkloadAdapter().connect(params, 30)
    warning = capsys.readouterr().err
    assert "search_path" in warning
    assert "threads" in warning
    # Key names only — never a value, since one of them could be a secret.
    assert "hunter2" not in warning
    assert "app" not in warning


def test_forwarded_and_mapped_keys_are_not_reported_as_dropped(monkeypatch, capsys):
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={"host": "db", "dbname": "x", "user": "u", "password": "p", "sslmode": "require"},
        source="profiles.yml",
    )
    PostgresWorkloadAdapter().connect(params, 30)
    assert capsys.readouterr().err == ""


def test_a_conninfo_build_failure_is_scrubbed_like_a_connect_failure(monkeypatch):
    """make_conninfo ran outside the scrubbing envelope, so its message was unfiltered.

    psycopg raises from make_conninfo on an unusable keyword, and that message can quote
    the offending value — which for a `password` keyword is the password.
    """
    import sys
    import types

    module = types.ModuleType("psycopg")

    def never_called(conninfo, **kwargs):
        raise AssertionError("must not reach connect() when the conninfo cannot be built")

    def explode(**kwargs):
        raise RuntimeError(f"invalid connection option: {sorted(kwargs.items())}")

    module.connect = never_called  # type: ignore[attr-defined]
    module.conninfo = types.SimpleNamespace(make_conninfo=explode)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", module)

    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={"host": "db", "user": "hans", "password": "hunter2"},
        source="profiles.yml",
    )
    with pytest.raises(ConnectionError) as exc:
        PostgresWorkloadAdapter().connect(params, 30)
    assert "hunter2" not in str(exc.value)
    assert "***" in str(exc.value)
    assert exc.value.__context__ is None


def test_connect_without_psycopg_installed_raises_a_helpful_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    adapter = PostgresWorkloadAdapter()
    params = ConnectionParams(
        engine="postgres", dsn="postgresql://u@h/db", fields={}, source="--dsn"
    )
    with pytest.raises(ImportError) as exc:
        adapter.connect(params, 30)
    assert "sqlquality[postgres]" in str(exc.value)


def test_the_ranking_key_ignores_a_boolean_cost_share():
    """The proposal sort had the same missing bool guard as the two renderers.

    `-float(True)` is -1.0, so a proposal carrying a stray True would sort ahead of a
    genuinely hot one at the same confidence — the ordering the CLI presents as "read this
    first".
    """
    from sqlquality.models import Confidence, Proposal

    stray = Proposal(
        code="ADV001",
        title="t",
        rationale="r",
        evidence={"cost_share": True},
        confidence=Confidence.HIGH,
    )
    hot = Proposal(
        code="ADV001",
        title="t",
        rationale="r",
        evidence={"cost_share": 0.5},
        confidence=Confidence.HIGH,
    )
    key = PostgresWorkloadAdapter._ranking_key
    assert key(hot) < key(stray)


def test_the_schema_statement_runs_once_per_run():
    """CAP_SCHEMA was executed by both fetch_schema and fetch_table_facts.

    Twice the catalog work, and — worse — two identical `degraded` entries when it is
    denied, so the user is told the same thing twice.
    """
    querier = FakeQuerier({"information_schema.columns": [("orders", "id", "integer")]})
    adapter = PostgresWorkloadAdapter(querier=querier)
    adapter.fetch_schema(("public",))
    adapter.fetch_table_facts(("public",), frozenset({"orders"}))
    schema_calls = [sql for sql, _ in querier.calls if "information_schema.columns" in sql]
    assert len(schema_calls) == 1


def test_a_denied_schema_statement_is_reported_once_not_twice():
    querier = FakeQuerier({}, fail_markers=("information_schema.columns",))
    adapter = PostgresWorkloadAdapter(querier=querier)
    adapter.fetch_schema(("public",))
    adapter.fetch_table_facts(("public",), frozenset({"orders"}))
    assert [cap for cap, _ in adapter.degraded].count(CAP_SCHEMA) == 1


def test_the_timeout_bounds_have_a_single_definition():
    """Two independent constant pairs drift: the CLI would reject what the adapter accepts,
    or the adapter would silently clamp past the range the CLI's error message promises."""
    import inspect
    from pathlib import Path

    from sqlquality import cli
    from sqlquality.workload import base

    assert cli.MIN_TIMEOUT_S is base.MIN_TIMEOUT_S
    assert cli.MAX_TIMEOUT_S is base.MAX_TIMEOUT_S
    from sqlquality.workload import postgres

    assert postgres.MAX_TIMEOUT_S is base.MAX_TIMEOUT_S
    # And imported, not re-typed — equal literals in two files are still two definitions.
    for module in (cli, postgres):
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
        assert str(base.MAX_TIMEOUT_S) not in source, (
            f"{module.__name__} restates the --timeout ceiling as a literal"
        )


def test_fetch_indexes_records_an_expression_index_rather_than_dropping_it():
    """`indkey` holds 0 for an expression column and no pg_attribute row has attnum 0.

    The old inner join therefore discarded those rows, so an index on `lower(status)`
    arrived with an empty column tuple and could not be reasoned about at all.
    """
    querier = FakeQuerier(
        {
            "pg_index": [
                # attname is NULL for the expression column, as a LEFT JOIN yields.
                (
                    "orders",
                    "idx_lower_status",
                    None,
                    1,
                    False,
                    False,
                    3,
                    8192,
                    False,
                    None,
                    True,
                    "CREATE INDEX idx_lower_status ON orders (lower(status))",
                ),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )
    index = indexes["orders"][0]
    assert index.has_expressions is True
    assert index.columns == ()
    assert "lower(status)" in (index.definition or "")


def test_fetch_indexes_records_a_partial_index_predicate():
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "orders",
                    "idx_open",
                    "status",
                    1,
                    False,
                    False,
                    7,
                    4096,
                    True,
                    "(shipped_at IS NULL)",
                    False,
                    "CREATE INDEX idx_open ON orders (status) WHERE shipped_at IS NULL",
                ),
            ]
        }
    )
    index = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )["orders"][0]
    assert index.is_partial is True
    assert index.predicate == "(shipped_at IS NULL)"
    assert index.columns == ("status",)


def test_fetch_indexes_leaves_a_plain_index_unmarked():
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "orders",
                    "idx_status",
                    "status",
                    1,
                    False,
                    False,
                    12,
                    4096,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status ON orders (status)",
                ),
            ]
        }
    )
    index = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )["orders"][0]
    assert (index.is_partial, index.predicate, index.has_expressions) == (False, None, False)


def test_the_indexes_statement_reads_predicate_and_expression_metadata():
    sql = PostgresWorkloadAdapter().SQL[CAP_INDEXES].lower()
    assert "indpred" in sql, "the partial-index predicate must be selected"
    assert "indexprs" in sql, "expression presence must be selected"
    assert "left join pg_attribute" in sql, (
        "an inner join drops expression columns, whose indkey entry is 0"
    )
