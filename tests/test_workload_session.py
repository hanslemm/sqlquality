"""Unit tests for `workload/session.py`'s `open_session`, the mechanism both the Postgres
and Redshift adapters' `connect()` share.

`test_workload_postgres.py` and `test_workload_redshift.py` each exercise this through
their own adapter and are not to be modified for this extraction, but neither happens to
construct a fake driver where the read-only statement itself fails *and* the caller asked
for `read_only_required=True` — Postgres's fake cursor never fails on that statement, and
Redshift's tests only exercise `read_only_required=False`. That combination is exactly the
Postgres invariant this module must keep ("a refused read-only statement aborts the
connection like any other setup failure"), so it is pinned here directly against the
shared helper rather than left to be an accident of which adapter happens to test it.
"""

from __future__ import annotations

import sys
import types

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload.postgres import PostgresWorkloadAdapter
from sqlquality.workload.session import READ_ONLY_SQL, open_session


class _FakeCursor:
    def __init__(self, log: list[tuple], *, fail_on: frozenset[str] = frozenset()) -> None:
        self._log = log
        self._fail_on = fail_on

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._log.append((sql, params))
        if sql in self._fail_on:
            raise RuntimeError(f"ERROR: {sql!r} refused")

    def fetchall(self):
        return []


class _FakeConnection:
    def __init__(self, *, fail_on: frozenset[str] = frozenset()) -> None:
        self.log: list[tuple] = []
        self._fail_on = fail_on

    def cursor(self):
        return _FakeCursor(self.log, fail_on=self._fail_on)


def _fake_psycopg(seen: dict, *, fail_on: frozenset[str] = frozenset()):
    module = types.ModuleType("psycopg")

    def connect(conninfo, **kwargs):
        seen["conninfo"] = conninfo
        seen["connection"] = _FakeConnection(fail_on=fail_on)
        return seen["connection"]

    module.connect = connect  # type: ignore[attr-defined]
    return module


def test_a_required_read_only_statement_that_fails_aborts_the_connection():
    """The Postgres invariant: `read_only_required=True` means a refusal is a setup
    failure like any other, not a degradation. Neither adapter's own test suite happens
    to force the read-only statement itself to fail under `required=True` — Postgres's
    fake cursor always lets it through, and Redshift's tests only exercise
    `required=False` — so this is the one place that combination is pinned.
    """
    seen: dict = {}
    psycopg = _fake_psycopg(seen, fail_on=frozenset({READ_ONLY_SQL}))
    with pytest.raises(ConnectionError) as exc:
        open_session(
            psycopg=psycopg,
            conninfo_factory=lambda: "postgresql:///x",
            secrets=(),
            timeout_s=30,
            min_timeout_s=1,
            max_timeout_s=3600,
            read_only_sql=READ_ONLY_SQL,
            read_only_required=True,
        )
    assert "refused" in str(exc.value)
    # The statement_timeout statement must never run once the required setup step failed.
    assert seen["connection"].log == [(READ_ONLY_SQL, None)]


def test_a_non_required_read_only_statement_that_fails_degrades_instead():
    """The Redshift invariant, direct on the helper: the same failure, under
    `read_only_required=False`, is caught, scrubbed, and returned — not raised — and
    setup continues to arm the statement timeout regardless."""
    seen: dict = {}
    psycopg = _fake_psycopg(seen, fail_on=frozenset({READ_ONLY_SQL}))
    query, degradation = open_session(
        psycopg=psycopg,
        conninfo_factory=lambda: "postgresql:///x",
        secrets=(),
        timeout_s=30,
        min_timeout_s=1,
        max_timeout_s=3600,
        read_only_sql=READ_ONLY_SQL,
        read_only_required=False,
    )
    assert degradation is not None
    assert "could not be proven read-only" in degradation
    assert seen["connection"].log == [
        (READ_ONLY_SQL, None),
        ("SELECT set_config('statement_timeout', %s, false)", ("30000ms",)),
    ]
    assert query is not None


def test_a_required_read_only_statement_that_succeeds_reports_no_degradation():
    seen: dict = {}
    psycopg = _fake_psycopg(seen)
    query, degradation = open_session(
        psycopg=psycopg,
        conninfo_factory=lambda: "postgresql:///x",
        secrets=(),
        timeout_s=30,
        min_timeout_s=1,
        max_timeout_s=3600,
        read_only_sql=READ_ONLY_SQL,
        read_only_required=True,
    )
    assert degradation is None
    assert query is not None


def test_conninfo_factory_runs_inside_the_scrubbing_envelope():
    """A `conninfo_factory` that raises must be scrubbed exactly like a connect failure —
    proving it runs *inside* `open_session`'s try, not before the call."""
    module = types.ModuleType("psycopg")
    module.connect = lambda *a, **k: (_ for _ in ()).throw(  # pragma: no cover - unreachable
        AssertionError("must not reach psycopg.connect")
    )

    def exploding_factory() -> str:
        raise RuntimeError("invalid option: password=hunter2")

    with pytest.raises(ConnectionError) as exc:
        open_session(
            psycopg=module,
            conninfo_factory=exploding_factory,
            secrets=("hunter2",),
            timeout_s=30,
            min_timeout_s=1,
            max_timeout_s=3600,
            read_only_sql=READ_ONLY_SQL,
            read_only_required=True,
        )
    assert "hunter2" not in str(exc.value)
    assert "***" in str(exc.value)
    assert exc.value.__context__ is None


def test_postgres_adapter_aborts_when_its_read_only_statement_is_refused(monkeypatch):
    """Pins `postgres.py`'s own call site, not just the shared helper.

    `postgres.py`'s `connect()` passes `read_only_required=True` as a literal at its one
    call to `open_session`. Nothing forces that literal to stay `True`: flip it to
    `False` and every existing Postgres unit test still passes, because none of them
    makes the read-only statement itself fail — the fake cursor those tests use always
    lets it through. Under that flip, Postgres would continue *silently* on a refused
    read-only statement, discarding the returned degradation, which is exactly the "we
    never write" promise on the engine that actually ships. This goes through
    `PostgresWorkloadAdapter.connect()` itself (via a `psycopg` planted in `sys.modules`,
    since that is how the adapter imports it) rather than through `open_session`
    directly, so a regression at the call site — not just in the helper — fails it.
    """

    class _FailingCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            if sql == READ_ONLY_SQL:
                raise RuntimeError("ERROR: read-only not supported in this configuration")

        def fetchall(self):
            return []

    class _FailingConnection:
        def cursor(self):
            return _FailingCursor()

    module = types.ModuleType("psycopg")
    module.connect = lambda conninfo, **kwargs: _FailingConnection()  # type: ignore[attr-defined]
    module.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", module)

    adapter = PostgresWorkloadAdapter()
    params = ConnectionParams(engine="postgres", dsn="postgresql:///x", fields={}, source="--dsn")
    with pytest.raises(ConnectionError) as exc:
        adapter.connect(params, 30)
    assert "read-only" in str(exc.value)
    # Not silently continued: no querier was ever installed.
    assert adapter._query is None


def test_the_timeout_is_clamped_with_the_caller_supplied_bounds():
    seen: dict = {}
    psycopg = _fake_psycopg(seen)
    open_session(
        psycopg=psycopg,
        conninfo_factory=lambda: "postgresql:///x",
        secrets=(),
        timeout_s=99999,
        min_timeout_s=1,
        max_timeout_s=120,
        read_only_sql=READ_ONLY_SQL,
        read_only_required=True,
    )
    assert seen["connection"].log[1] == (
        "SELECT set_config('statement_timeout', %s, false)",
        ("120000ms",),
    )
