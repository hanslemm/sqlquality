"""The CI job that runs the live suite, pinned structurally.

This cannot prove the job works — only a run on a PR can, and until then it is unverified. What
it *can* do is pin every piece of wiring whose silent breakage would turn the live suite back
into something that never runs, which is the state this job was added to end:

* the job existing at all;
* the step that starts the server publishing the port and creating the database the fixture
  actually looks for (a mismatch makes every test skip, and a skip exits 0);
* the two `pg_stat_statements` server flags, without which the suite connects and then fails on
  every workload read;
* the explicit readiness wait — `docker run` has no health check to gate the job's steps — and
  the final step that refuses a run which skipped or executed nothing.

Each of those is a one-line edit away from being silently wrong, and none of them is visible in
a green run. `tests/integration/conftest.py`'s own constants are the source of truth here, so
the port and database name cannot drift between the compose file, the fixture and CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:  # pragma: no cover - pytest normally does this itself
    sys.path.insert(0, str(_TESTS_DIR))

from integration.conftest import DEFAULT_PORT, EXPECTED_DATABASE  # noqa: E402

_CI = _TESTS_DIR.parent / ".github" / "workflows" / "ci.yml"
_JOB = "integration"


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Also the test that ci.yml is valid YAML at all — a workflow that does not parse is
    silently not run by GitHub, with no failing check to notice."""
    return yaml.safe_load(_CI.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def job(workflow: dict) -> dict:
    assert _JOB in workflow["jobs"], (
        f"ci.yml has no {_JOB!r} job, so the 23 live tests run on nobody's machine but the "
        f"author's. Jobs present: {sorted(workflow['jobs'])}"
    )
    return workflow["jobs"][_JOB]


def _steps_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


@pytest.fixture(scope="module")
def server_step(job: dict) -> str:
    """The step that starts Postgres, as shell text.

    Deliberately a step and not a `services:` container, and the history is worth keeping
    because the obvious approach fails on a real runner. A service container cannot be given
    a command — GitHub's schema has no `command` key, and `options` is passed to
    `docker create` *before* the image — so the `-c shared_preload_libraries=…` flags the
    compose file uses cannot be expressed there. The first version worked around that with
    `ALTER SYSTEM` plus a restart and failed in CI: `ALTER SYSTEM SET
    pg_stat_statements.track` is rejected outright with "unrecognized configuration
    parameter", because that setting does not exist until the library is loaded — which is
    what the restart was supposed to accomplish. Getting the order right would need two
    restarts.

    `docker run` takes the same flags as `tests/integration/docker-compose.yml`, so the
    server CI tests against is described once rather than twice.
    """
    steps = [
        str(step.get("run", ""))
        for step in job["steps"]
        if "docker run" in str(step.get("run", ""))
    ]
    assert len(steps) == 1, (
        "expected exactly one step starting the server; found "
        f"{len(steps)}. Without it the live suite skips itself and the job exits 0."
    )
    return steps[0]


def test_the_server_is_postgres_16(server_step: str):
    """The live suite reads `pg_stat_statements` columns and `reltuples` semantics that are
    version-dependent; the compose file pins 16 and CI must not silently test another major."""
    assert "postgres:16" in server_step


def test_the_server_publishes_the_port_the_fixture_connects_to(server_step: str):
    """A port that drifts from the fixture's makes every test skip — and pytest exits 0 on a
    skip, so the job would pass having run nothing. The final step catches that; this catches
    it earlier and says which side is wrong."""
    assert f"-p {DEFAULT_PORT}:5432" in server_step


def test_the_server_creates_the_database_the_fixture_verifies(server_step: str):
    """`live_dsn` fails when `current_database()` is not this name, so a change here would turn
    the whole job red with a port-collision message that is not the real cause."""
    assert f"POSTGRES_DB={EXPECTED_DATABASE}" in server_step
    assert "POSTGRES_PASSWORD=" in server_step, "the fixture's DSN authenticates with a password"


def test_ci_waits_for_the_server_before_running_anything(server_step: str):
    """`docker run` has no health check to gate the job's steps, so the wait is explicit.
    Without it the suite races startup, every test skips itself, and the job exits 0 — the
    exact silent no-op this job was added to end. The final `pg_isready` outside the retry
    loop is what turns a server that never came up into a failure rather than a skip."""
    assert "pg_isready" in server_step
    assert "seq 1" in server_step, "a bounded retry loop, not a single optimistic check"


def test_ci_passes_both_pg_stat_statements_settings_as_server_flags(server_step: str):
    """Neither can be reached by `CREATE EXTENSION`: `shared_preload_libraries` is loaded at
    postmaster start, and `track = all` is what makes nested statements (the DECLARE ... CURSOR
    case the live suite unwraps) appear at all.

    Asserted as **command flags**, which is the only form that works. Applying them with
    `ALTER SYSTEM` and a restart was tried and failed in CI: `ALTER SYSTEM SET
    pg_stat_statements.track` is rejected with "unrecognized configuration parameter", because
    the setting does not exist until the library is loaded. These same two flags appear in
    `tests/integration/docker-compose.yml`; a drift between the two would mean local and CI
    runs test differently configured servers.
    """
    assert "-c shared_preload_libraries=pg_stat_statements" in server_step
    assert "-c pg_stat_statements.track=all" in server_step


def test_ci_verifies_the_preload_took_effect_before_running_any_test(job: dict):
    """Otherwise a failed ALTER SYSTEM surfaces as an error deep inside a seeding fixture,
    which says nothing about the real cause."""
    steps = _steps_text(job)
    assert "current_setting('shared_preload_libraries')" in steps
    assert "current_setting('pg_stat_statements.track')" in steps


def test_ci_points_the_fixture_at_the_service_it_started(job: dict):
    """The fixture reads `SQLQUALITY_TEST_DSN`. Set explicitly rather than relying on the
    fixture's default, so this file and the fixture cannot disagree silently."""
    dsn = next(
        step["env"]["SQLQUALITY_TEST_DSN"]
        for step in job["steps"]
        if "SQLQUALITY_TEST_DSN" in step.get("env", {})
    )
    assert f":{DEFAULT_PORT}/" in dsn
    assert dsn.endswith(f"/{EXPECTED_DATABASE}")


def test_ci_actually_selects_the_integration_marker(job: dict):
    """`pytest` with no `-m` runs the *default* suite, which deselects every one of these
    tests: the job would be green, fast, and a complete no-op."""
    steps = _steps_text(job)
    assert "-m integration" in steps
    assert "--strict-markers" in steps, "a typo'd marker would otherwise select nothing"


def test_ci_refuses_a_run_that_skipped_or_executed_nothing(job: dict):
    """The reason this job needs a guard at all: every test in the package skips itself when no
    Postgres answers, and pytest exits 0 on a skip. A service that never became ready, a wrong
    port, or a marker typo would all produce a passing job that ran nothing — the same failure
    the `no-extras` job refuses for the same reason.
    """
    steps = _steps_text(job)
    assert "--junitxml" in steps, (
        "the count has to come from pytest's own machine-readable report, not a regex over "
        "output written for humans"
    )
    assert "skipped" in steps
    assert "executed == 0" in steps
    guard = next(step for step in job["steps"] if "executed == 0" in str(step.get("run", "")))
    assert guard.get("if") == "always()", (
        "a failing suite must still report whether it skipped everything or genuinely failed"
    )


def _guard_script(job: dict) -> str:
    """The guard's Python body, exactly as the shell pipes it to the interpreter.

    Deliberately no `textwrap.dedent`: YAML has already stripped the block scalar's common
    indentation, so what CI feeds Python is this string verbatim. Dedenting here would hide a
    real indentation error in the file.
    """
    run = next(step["run"] for step in job["steps"] if "executed == 0" in str(step.get("run", "")))
    lines = run.splitlines()
    assert "PY" in lines, "the heredoc terminator must be a line of its own"
    assert lines[lines.index("PY")] == "PY", (
        "the terminator is indented, so `<<'PY'` never closes and the shell reads to EOF"
    )
    return run.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def test_the_guard_script_is_valid_python(job: dict):
    """It is a heredoc inside YAML inside a shell script, three layers away from anything that
    would catch a syntax error before CI runs it — and it only executes at the very end of the
    job, after everything expensive has already run."""
    import ast

    tree = ast.parse(_guard_script(job))
    assert len(tree.body) > 5, "a near-empty parse would make this test vacuous"


def _junit_for(source: str, tmp_path: Path, *, select: str = "") -> Path:
    """A **real** pytest JUnit report, not a hand-written one.

    The guard reads attributes off pytest's XML, so a fixture built from what those attributes
    are believed to be would prove only self-consistency. Generating the report with pytest
    itself is what makes the parse test meaningful.
    """
    import subprocess

    (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")
    xml = tmp_path / "integration.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"]
        + (["-m", select] if select else []),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert xml.exists(), "pytest wrote no report; the rest of this test would prove nothing"
    return xml


def _run_guard(job: dict, xml: Path) -> tuple[int, str]:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _guard_script(job)],
        cwd=xml.parent,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def test_the_guard_accepts_a_run_where_every_test_executed(job: dict, tmp_path):
    """The control. A guard that fails on everything would block every honest run, and the
    obvious way to write one — treating a nonzero `skipped` attribute as the only signal —
    reads `tests` and `skipped` off the wrong element and fails universally."""
    xml = _junit_for("def test_a(): pass\ndef test_b(): pass\n", tmp_path)
    code, output = _run_guard(job, xml)
    assert code == 0, output
    assert "executed=2" in output, output


def test_the_guard_fails_a_run_that_skipped(job: dict, tmp_path):
    """The failure mode this job exists to refuse. Every test in tests/integration/ skips itself
    when no Postgres answers, and pytest exits 0 on a skip — so a service container that never
    became ready would otherwise be indistinguishable from a passing run.
    """
    xml = _junit_for(
        "import pytest\ndef test_a(): pass\ndef test_b(): pytest.skip('no server')\n", tmp_path
    )
    code, output = _run_guard(job, xml)
    assert code != 0, output
    assert "skipped" in output
    assert "must not read as a pass" in output


def test_the_guard_explains_a_missing_report_instead_of_raising(job: dict, tmp_path):
    """Reachable because the step is `if: always()`: the pytest step can die before writing a
    report, and a traceback over `ET.parse` buries the real log."""
    code, output = _run_guard(job, tmp_path / "integration.xml")
    assert code != 0
    assert "never written" in output
    assert "Traceback" not in output


def test_the_guard_fails_a_run_that_collected_nothing(job: dict, tmp_path):
    """A marker typo, a renamed package, or a deselect-everything change: pytest reports zero
    tests, and without this the job's only other signal is an exit code the previous step
    already consumed."""
    xml = _junit_for(
        "import pytest\n@pytest.mark.other\ndef test_a(): pass\n", tmp_path, select="nothingmatches"
    )
    code, output = _run_guard(job, xml)
    assert code != 0, output
    assert "zero integration tests ran" in output
