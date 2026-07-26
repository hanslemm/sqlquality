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


def test_read_profile_uses_the_env_var_default_when_the_variable_is_unset(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        "      host: \"{{ env_var('PGHOST', 'localhost') }}\"\n"
    )
    _engine, fields = read_profile(tmp_path, "jaffle", None, {})
    assert fields["host"] == "localhost"


def test_read_profile_prefers_the_environment_over_the_default(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        "      host: \"{{ env_var('PGHOST', 'localhost') }}\"\n"
    )
    _engine, fields = read_profile(tmp_path, "jaffle", None, {"PGHOST": "db.internal"})
    assert fields["host"] == "db.internal"


def test_read_profile_rejects_jinja_it_cannot_resolve(tmp_path):
    """An unresolved `{{ ... }}` reaching a driver as a hostname is a baffling failure."""
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        '      host: "{{ my_custom_macro() }}"\n'
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    assert "host" in str(exc.value)
    assert "--dsn" in str(exc.value)


def test_profile_errors_never_leak_a_secret_value(tmp_path):
    """Field values can be passwords; only names may appear in an error message."""
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        '      password: "hunter2{{ my_custom_macro() }}"\n'
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    assert "hunter2" not in str(exc.value)


def test_malformed_yaml_error_never_echoes_a_secret(tmp_path):
    """PyYAML's message quotes the offending source line verbatim.

    A stray tab on a `password:` line is an ordinary hand-editing mistake, and
    interpolating `str(exc)` would put the secret into the error — and into CI logs.
    """
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        "      password: hunter2\tSUPERSECRET\n"
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    message = str(exc.value)
    assert "SUPERSECRET" not in message
    assert "hunter2" not in message
    # Still actionable: the position survives even though the content does not.
    assert "line 6" in message


def test_malformed_yaml_error_suppresses_the_leaky_cause(tmp_path):
    """`from None` is the other half of the fix.

    A chained cause would let the traceback print PyYAML's snippet even though our own
    message is clean, so the suppression is load-bearing rather than stylistic.
    """
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  outputs:\n    dev:\n      password: hunter2\tSUPERSECRET\n"
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__ is True


def test_profile_with_no_default_target_says_so(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  outputs:\n    dev:\n      type: postgres\n      dbname: a\n"
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    message = str(exc.value)
    assert "--target" in message
    assert "'None'" not in message


def test_dsn_means_profiles_yml_is_never_read(tmp_path):
    """Precedence must short-circuit, not merely be preferred.

    profiles.yml here is malformed, so any attempt to read it raises. A passing test
    proves the DSN path returns before the file is touched.
    """
    (tmp_path / "profiles.yml").write_text("{{{ not valid yaml at all")
    params = resolve_connection(
        dsn="postgresql://u@h/db",
        engine=None,
        profile="jaffle",
        target=None,
        profiles_dir=Path(tmp_path),
        env={},
    )
    assert params.source == "--dsn"


def test_env_dsn_means_profiles_yml_is_never_read(tmp_path):
    (tmp_path / "profiles.yml").write_text("{{{ not valid yaml at all")
    params = resolve_connection(
        dsn=None,
        engine=None,
        profile="jaffle",
        target=None,
        profiles_dir=Path(tmp_path),
        env={"SQLQUALITY_DSN": "postgresql://u@h/db"},
    )
    assert params.source == "env"


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
