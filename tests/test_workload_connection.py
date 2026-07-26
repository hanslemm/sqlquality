from pathlib import Path

import pytest

from sqlquality.workload.connection import (
    ConnectionResolutionError,
    read_profile,
    resolve_connection,
)


def test_dsn_wins_over_env_and_profiles():
    params = resolve_connection(
        dsn="postgresql://u@h/db",
        engine=None,
        profile=None,
        target=None,
        profiles_dir=None,
        env={"SQLQUALITY_DSN": "postgresql://other@h/db"},
    )
    assert params.dsn == "postgresql://u@h/db"
    assert params.source == "--dsn"


def test_engine_inferred_from_dsn_scheme():
    for scheme, engine in [
        ("postgresql", "postgres"),
        ("postgres", "postgres"),
        ("redshift", "redshift"),
        ("snowflake", "snowflake"),
    ]:
        params = resolve_connection(
            dsn=f"{scheme}://u@h/db",
            engine=None,
            profile=None,
            target=None,
            profiles_dir=None,
            env={},
        )
        assert params.engine == engine


def test_explicit_engine_overrides_the_scheme():
    params = resolve_connection(
        dsn="postgresql://u@h/db",
        engine="redshift",
        profile=None,
        target=None,
        profiles_dir=None,
        env={},
    )
    assert params.engine == "redshift"


def test_env_dsn_used_when_no_flag():
    params = resolve_connection(
        dsn=None,
        engine=None,
        profile=None,
        target=None,
        profiles_dir=None,
        env={"SQLQUALITY_DSN": "postgresql://u@h/db"},
    )
    assert params.source == "env"
    assert params.engine == "postgres"


def test_unknown_dsn_scheme_is_an_error():
    with pytest.raises(ConnectionResolutionError) as exc:
        resolve_connection(
            dsn="mysql://u@h/db", engine=None, profile=None, target=None, profiles_dir=None, env={}
        )
    assert "mysql" in str(exc.value)


def test_nothing_supplied_is_an_error_naming_all_three_options():
    with pytest.raises(ConnectionResolutionError) as exc:
        resolve_connection(
            dsn=None, engine=None, profile=None, target=None, profiles_dir=None, env={}
        )
    message = str(exc.value)
    assert "--dsn" in message and "SQLQUALITY_DSN" in message and "--profile" in message


def test_read_profile_resolves_target_and_env_var(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        """
jaffle:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5432
      user: "{{ env_var('PGUSER') }}"
      dbname: analytics
      schema: public
"""
    )
    engine, fields = read_profile(tmp_path, "jaffle", None, {"PGUSER": "hans"})
    assert engine == "postgres"
    assert fields["user"] == "hans"
    assert fields["dbname"] == "analytics"


def test_read_profile_rejects_unknown_profile(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "nope", None, {})
    assert "nope" in str(exc.value)


def test_read_profile_missing_env_var_reports_the_variable_name(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        "      user: \"{{ env_var('PGUSER') }}\"\n"
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    assert "PGUSER" in str(exc.value)


def test_profiles_path_used_when_no_dsn(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n      dbname: a\n"
    )
    params = resolve_connection(
        dsn=None,
        engine=None,
        profile="jaffle",
        target=None,
        profiles_dir=Path(tmp_path),
        env={},
    )
    assert params.source == "profiles.yml"
    assert params.engine == "postgres"
    assert params.dsn is None
