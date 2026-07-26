from pathlib import Path

import pytest

from sqlquality.workload.connection import (
    ConnectionResolutionError,
    read_profile,
    resolve_connection,
)
from sqlquality.workload.profiles import ProfileError, read_profiles_file


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


def test_malformed_yaml_error_is_fully_detached_from_the_leaky_original(tmp_path):
    """Walks the whole chain, the way an error reporter does.

    `raise ... from None` is *not* sufficient: raising inside an `except` block sets
    __context__ regardless of the `from` clause, so the raw YAMLError — whose text quotes
    the offending source line — stays reachable even though a printed traceback hides it.
    Both raise sites therefore build their message inside the handler and raise after it
    exits. This test fails if either one is moved back inside.
    """
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  outputs:\n    dev:\n      password: hunter2\tSUPERSECRET\n"
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})

    node: BaseException | None = exc.value
    depth = 0
    while node is not None and depth < 8:
        assert "SUPERSECRET" not in str(node), f"secret reachable at chain depth {depth}"
        assert "hunter2" not in str(node), f"secret reachable at chain depth {depth}"
        node = node.__cause__ or node.__context__
        depth += 1


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


def test_profiles_yml_is_decoded_as_utf8_regardless_of_locale(tmp_path, monkeypatch):
    """`read_text()` with no encoding uses the *platform's* preferred encoding.

    On a cp1252 machine a UTF-8 profiles.yml is mojibake, and a non-ASCII password
    corrupts into a baffling authentication failure with nothing pointing at the cause.
    """
    recorded: dict = {}
    real_read_text = Path.read_text

    def spy(self, *args, **kwargs):
        recorded["encoding"] = kwargs.get("encoding")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        "      password: pässwörd\n",
        encoding="utf-8",
    )
    read_profile(tmp_path, "jaffle", None, {})
    assert recorded["encoding"] == "utf-8"


#: A latin-1 profiles.yml whose password contains a byte utf-8 cannot decode.
UNDECODABLE_PROFILE = "jaffle:\n  outputs:\n    dev:\n      password: SUPERSECRÈT\n".encode(
    "latin-1"
)


def test_a_non_utf8_profiles_file_names_the_file(tmp_path):
    """UnicodeDecodeError already exits 2 (it is a ValueError), so this is message quality:
    the raw codec error named a byte offset and nothing a user could act on."""
    (tmp_path / "profiles.yml").write_bytes(UNDECODABLE_PROFILE)
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    message = str(exc.value)
    assert "profiles.yml" in message
    assert "UTF-8" in message
    assert "SUPERSECR" not in message


def test_the_undecodable_file_is_not_reachable_from_the_profile_error(tmp_path):
    """Walks the chain from `read_profiles_file`, and inspects `.object`, not just `str()`.

    Two things had to be right here, and an earlier version of this test had neither.

    *Where it walks.* Walking from `read_profile` proves nothing about this module: that
    boundary re-raises outside its own handler, so the chain it hands back is one exception
    long no matter what `profiles.py` does. A re-chain inside `read_profiles_file` was
    invisible from there. The walk has to start at the layer that catches the decode error.

    *What it inspects.* `str(UnicodeDecodeError)` is "'utf-8' codec can't decode byte 0xc8
    in position 53 ..." — it never quotes the content, so a str-only assertion could not
    fail. The bytes are in `.object`, which holds the *entire* file, `password:` line
    included. That is the leak, and it is why this error is recorded and re-raised after the
    handler exits rather than chained.
    """
    (tmp_path / "profiles.yml").write_bytes(UNDECODABLE_PROFILE)
    with pytest.raises(ProfileError) as exc:
        read_profiles_file(tmp_path)

    node: BaseException | None = exc.value
    depth = 0
    while node is not None and depth < 8:
        where = f"chain depth {depth} ({type(node).__name__})"
        assert "SUPERSECR" not in str(node), f"file content reachable in the message at {where}"
        payload = getattr(node, "object", b"")
        if isinstance(payload, (bytes, bytearray)):
            assert b"SUPERSECR" not in payload, f"file content reachable via .object at {where}"
        node = node.__cause__ or node.__context__
        depth += 1
    # And the chain really is severed, not merely secret-free by luck.
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
