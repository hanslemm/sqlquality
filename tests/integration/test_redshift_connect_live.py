"""Redshift's connect(), exercised against a real server.

Redshift speaks the PostgreSQL wire protocol, so `connect()` — the session setup, the
clamped statement timeout, and secret scrubbing on a bad password — is genuinely
testable against the same `postgres:16` container the rest of this package uses. The
catalog SQL (`svv_*`/`sys_*`) is not: those views do not exist in Postgres, so nothing
here calls `fetch_workload`, `fetch_schema`, `fetch_table_facts`, or `propose`.
"""

from __future__ import annotations

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload.redshift import RedshiftWorkloadAdapter


@pytest.mark.integration
def test_redshift_adapter_connects_over_the_postgres_wire_protocol(live_dsn):
    """Redshift speaks the PostgreSQL protocol, so the session setup is genuinely testable.

    The catalog statements are not — `svv_*` does not exist here — so this test deliberately
    covers connect() only, and asserts nothing about introspection.
    """
    adapter = RedshiftWorkloadAdapter()
    params = ConnectionParams(engine="redshift", dsn=live_dsn, fields={}, source="test")
    adapter.connect(params, timeout_s=30)
    assert adapter._query is not None
    rows = adapter._query("SELECT 1", ())
    assert rows == [(1,)]


@pytest.mark.integration
def test_a_wrong_password_leaks_nothing(live_dsn):
    adapter = RedshiftWorkloadAdapter()
    bad = live_dsn.replace(":sqlquality@", ":wr0ng-p4ss@")
    params = ConnectionParams(engine="redshift", dsn=bad, fields={}, source="test")
    with pytest.raises(ConnectionError) as exc:
        adapter.connect(params, timeout_s=5)
    assert "wr0ng-p4ss" not in str(exc.value)
    assert "wr0ng-p4ss" not in repr(exc.value)


@pytest.mark.integration
def test_a_real_postgres_accepts_the_read_only_statement_without_degradation(live_dsn):
    """Against the real `postgres:16` container `SET default_transaction_read_only` always
    succeeds, so the belt-and-braces guard reports no degradation. The refusal path itself
    (`test_a_refused_read_only_statement_degrades_rather_than_aborts` in
    `tests/test_workload_redshift.py`) cannot be exercised live — Postgres always accepts
    this statement — so it is covered at unit level with a fake driver that refuses it,
    which is the situation this adapter is actually built to handle on a real cluster.
    """
    adapter = RedshiftWorkloadAdapter()
    params = ConnectionParams(engine="redshift", dsn=live_dsn, fields={}, source="test")
    adapter.connect(params, timeout_s=30)
    assert adapter.degraded == []
