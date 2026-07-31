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

import types

import pytest

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
