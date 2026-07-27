from __future__ import annotations

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload.secrets import (
    MIN_SCRUBBABLE_SECRET,
    SECRET_FIELDS,
    WITHHELD,
    clamp_timeout_ms,
    scrub,
    secrets_for,
)


def _params(**kwargs) -> ConnectionParams:
    base = {"engine": "postgres", "dsn": None, "fields": {}, "source": "--dsn"}
    base.update(kwargs)
    return ConnectionParams(**base)  # type: ignore[arg-type]


def test_secrets_for_collects_password_fields():
    assert secrets_for(_params(fields={"host": "db", "password": "hunter2"})) == ("hunter2",)


def test_secrets_for_covers_both_forms_of_a_dsn_password():
    """urlparse leaves the password encoded; libpq decodes it before authenticating."""
    got = secrets_for(_params(dsn="postgresql://u:p%40ss@h/db"))
    assert "p%40ss" in got
    assert "p@ss" in got


def test_secrets_for_tolerates_a_dsn_with_no_password_or_a_malformed_one():
    for dsn in ("postgresql://u@h/db", "not a valid dsn :: at all ///"):
        assert secrets_for(_params(dsn=dsn)) == (dsn,)


def test_scrub_redacts_a_present_secret():
    assert scrub('failed for user "u" (hunter2)', ("hunter2",)) == 'failed for user "u" (***)'


def test_scrub_withholds_rather_than_mangles_an_unredactable_secret():
    assert scrub("a database has an admin", ("a",)) == WITHHELD
    assert scrub("connection refused", ("a",)) == "connection refused"


def test_min_scrubbable_secret_is_the_documented_floor():
    assert MIN_SCRUBBABLE_SECRET == 4
    assert "password" in SECRET_FIELDS


@pytest.mark.parametrize(
    ("given", "expected_ms"),
    [(0, 1_000), (-5, 1_000), (30, 30_000), (99_999, 3_600_000)],
)
def test_clamp_timeout_ms_bounds_and_converts(given, expected_ms):
    assert clamp_timeout_ms(given, minimum=1, maximum=3600) == expected_ms
