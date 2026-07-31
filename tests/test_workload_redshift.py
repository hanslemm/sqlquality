import re
import sys
import types

import sqlglot
import pytest

from sqlquality.models import Aggregation, ConnectionParams, Workload
from sqlquality.workload import get_workload_adapter
from sqlquality.workload.base import MAX_TIMEOUT_S
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
    "fetch_workload": lambda a: a.fetch_workload(None, 10),
    "fetch_schema": lambda a: a.fetch_schema(("public",)),
    "fetch_table_facts": lambda a: a.fetch_table_facts(("public",), frozenset()),
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
