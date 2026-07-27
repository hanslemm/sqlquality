import pytest
import sqlglot
from sqlglot import exp

from sqlquality.models import RawQueryRow, WorkloadFetch
from sqlquality.sqlast import parse
from sqlquality.workload.fingerprint import (
    FLAG_LEADING_WILDCARD_LIKE,
    FLAG_SELECT_STAR,
    ingest,
    is_noise,
    literal_flags,
    redact_tree,
    unwrap,
)


def _tree(sql):
    return sqlglot.parse_one(sql, dialect="postgres")


def test_redact_removes_every_literal():
    redacted = redact_tree(_tree("select id from t where email = 'a@b.de' and n > 42"))
    rendered = redacted.sql("postgres")
    assert "a@b.de" not in rendered
    assert "42" not in rendered


def test_redact_does_not_mutate_the_input_tree():
    tree = _tree("select id from t where email = 'a@b.de'")
    redact_tree(tree)
    assert "a@b.de" in tree.sql("postgres")


def test_literal_flags_capture_leading_wildcard_before_redaction():
    assert FLAG_LEADING_WILDCARD_LIKE in literal_flags(_tree("select 1 from t where a like '%x'"))
    assert FLAG_LEADING_WILDCARD_LIKE not in literal_flags(
        _tree("select 1 from t where a like 'x%'")
    )


def test_literal_flags_capture_select_star():
    assert FLAG_SELECT_STAR in literal_flags(_tree("select * from t"))
    assert FLAG_SELECT_STAR not in literal_flags(_tree("select id from t"))


def test_is_noise_filters_our_own_introspection_and_ddl():
    assert is_noise("SELECT * FROM pg_stat_statements")
    assert is_noise("select column_name from information_schema.columns")
    assert is_noise("CREATE INDEX idx ON t (a)")
    assert is_noise("VACUUM ANALYZE orders")
    assert not is_noise("select id from orders where status = $1")


def test_is_noise_still_discards_the_raw_wrapper_text():
    """`is_noise` itself is a statement-prefix filter and stays that way.

    A cursor declaration or a `COPY (...) TO` still starts with a keyword `_LEADING_NOISE`
    matches, so calling `is_noise` on the *raw* row still discards it whole. That is no
    longer a loss: `ingest` calls `unwrap()` first and tests `is_noise` on the inner query,
    so the real read survives — see `test_a_declared_cursor_is_analyzed_not_filtered` and
    `test_a_copy_subquery_is_analyzed_not_filtered` below.
    """
    assert is_noise("DECLARE cur CURSOR FOR SELECT id FROM orders WHERE status = $1")
    assert is_noise("COPY (SELECT id FROM orders WHERE status = $1) TO STDOUT")


def test_is_noise_filters_session_control_but_not_update_set():
    assert is_noise("SET search_path TO public")
    assert is_noise("set statement_timeout = '30s'")
    # The bug this guards: an unanchored `set\s+` also matches `UPDATE ... SET`, which
    # silently dropped every UPDATE in the workload.
    assert not is_noise("UPDATE users SET email = $1 WHERE id = $2")
    assert not is_noise("update orders set status = $1 where created_at < $2")


def test_is_noise_keeps_all_dml():
    assert not is_noise("insert into audit (id, note) values ($1, $2)")
    assert not is_noise("DELETE FROM sessions WHERE expires_at < $1")
    assert not is_noise("with recent as (select 1) select * from recent")


def test_is_noise_does_not_trip_on_a_keyword_inside_a_literal():
    assert not is_noise("select id from events where action = 'commit'")
    assert not is_noise("select id from t where label = 'vacuum the floor'")


def test_ingest_groups_by_fingerprint_and_sums_cost():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(sql="select id from t where a = 1", calls=2, total_time_ms=10.0),
            RawQueryRow(sql="select id from t where a = 999", calls=3, total_time_ms=20.0),
        ),
        window_description="since stats reset",
    )
    workload = ingest(fetch, "postgres")
    assert len(workload.stats) == 1
    assert workload.stats[0].calls == 5
    assert workload.stats[0].total_time_ms == 30.0
    assert "999" not in workload.stats[0].sql


def test_ingest_counts_unparseable_and_noise_without_raising():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(sql="select from where", calls=1, total_time_ms=1.0),
            RawQueryRow(sql="select * from pg_stat_statements", calls=1, total_time_ms=1.0),
            RawQueryRow(sql="select id from t", calls=1, total_time_ms=1.0),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_unparseable == 1
    assert workload.skipped_noise == 1
    assert len(workload.stats) == 1


def test_ingest_captures_literal_flags_before_redaction():
    """The load-bearing ordering guard.

    `like '%x'` becomes `like %s` once redacted, so if ingest ever computed the flags from
    the already-redacted tree, FLAG_LEADING_WILDCARD_LIKE would become permanently absent
    and every other test would still pass.
    """
    fetch = WorkloadFetch(
        rows=(RawQueryRow(sql="select id from t where a like '%x'", calls=1, total_time_ms=1.0),),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert FLAG_LEADING_WILDCARD_LIKE in workload.stats[0].flags
    assert "%x" not in workload.stats[0].sql


def test_ingest_captures_select_star_flag():
    fetch = WorkloadFetch(
        rows=(RawQueryRow(sql="select * from t", calls=1, total_time_ms=1.0),),
        window_description="w",
    )
    assert FLAG_SELECT_STAR in ingest(fetch, "postgres").stats[0].flags


def test_ingest_keep_literals_preserves_values():
    fetch = WorkloadFetch(
        rows=(RawQueryRow(sql="select id from t where a = 999", calls=1, total_time_ms=1.0),),
        window_description="w",
    )
    workload = ingest(fetch, "postgres", keep_literals=True)
    assert "999" in workload.stats[0].sql


def test_ingest_stats_are_sorted_by_cost_descending():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(sql="select a from t", calls=1, total_time_ms=1.0),
            RawQueryRow(sql="select b from t", calls=1, total_time_ms=99.0),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert [s.total_time_ms for s in workload.stats] == [99.0, 1.0]


def test_redaction_leaves_a_postgres_placeholder_intact():
    """`$N` parses as Parameter(this=Literal(N)), so a naive literal walk mangles it.

    pg_stat_statements hands us `$1`; redacting the `1` inside it produced `$%s`.
    """
    tree = parse("select id from orders where status = $1", "postgres")
    assert "$1" in redact_tree(tree).sql("postgres")


def test_redaction_does_not_dismember_a_normalised_interval():
    """The shape that was silently dropped, using the real text a live server produced.

    Redacting the literal inside `interval $2` left `INTERVAL` as a bare dangling token.
    Re-parsed, sqlglot read it as a *column* named INTERVAL, which failed to qualify — so
    the whole group vanished into skipped_unqualifiable. `created_at > now() - interval
    '1 day'` is the most ordinary filter shape there is.

    Note: sqlglot's postgres generator renders a unitless `INTERVAL $2` as `INTERVAL '2'`
    (it reads the parameter's `.name` into its single-string interval form) even for the
    *untouched* tree, before `redact_tree` ever runs — so the literal text `$2` does not
    survive rendering regardless of this fix. What the fix guarantees, and what actually
    caused the query group to vanish, is that the round trip stays parseable and
    `created_at` keeps being read as a column rather than growing a bogus `INTERVAL`
    sibling. That is what this test proves instead.
    """
    sql = "select id from orders where status = $1 and created_at > now() - interval $2"
    redacted = redact_tree(parse(sql, "postgres")).sql("postgres")
    assert not redacted.rstrip().endswith("INTERVAL")
    reparsed = parse(redacted, "postgres")
    columns = {c.name.upper() for c in reparsed.find_all(exp.Column)}
    assert "INTERVAL" not in columns
    assert "CREATED_AT" in columns


def test_redaction_still_erases_a_real_literal_beside_a_placeholder():
    """The control. Skipping placeholders must not smuggle a genuine literal through."""
    sql = "select id from orders where status = 'secret-value' and n > $1"
    redacted = redact_tree(parse(sql, "postgres")).sql("postgres")
    assert "secret-value" not in redacted
    assert "$1" in redacted


@pytest.mark.parametrize(
    "sql,expected",
    [
        (
            "DECLARE c CURSOR FOR SELECT id FROM orders WHERE status = 'x'",
            "SELECT id FROM orders WHERE status = 'x'",
        ),
        (
            "DECLARE c CURSOR WITH HOLD FOR SELECT id FROM orders",
            "SELECT id FROM orders",
        ),
        (
            "DECLARE c NO SCROLL CURSOR FOR SELECT id FROM orders",
            "SELECT id FROM orders",
        ),
        (
            "DECLARE c BINARY INSENSITIVE SCROLL CURSOR WITH HOLD FOR SELECT a FROM t",
            "SELECT a FROM t",
        ),
        (
            'DECLARE "my cursor" CURSOR FOR SELECT a FROM t',
            "SELECT a FROM t",
        ),
        (
            "COPY (SELECT id FROM orders WHERE status = 'x') TO STDOUT",
            "SELECT id FROM orders WHERE status = 'x'",
        ),
        ("copy (select 1) to stdout", "select 1"),
    ],
)
def test_unwrap_recovers_the_inner_query(sql, expected):
    assert unwrap(sql) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM orders",
        "COPY orders TO STDOUT",
        "COPY orders (id, status) FROM STDIN",
        "FETCH 100 FROM c",
        "CLOSE c",
        "DECLARE c CURSOR FOR",
        "DECLARE",
    ],
)
def test_unwrap_leaves_everything_else_alone(sql):
    """Anything without a recoverable inner query is returned unchanged, not mangled."""
    assert unwrap(sql) == sql


def test_a_declared_cursor_is_analyzed_not_filtered():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="DECLARE c CURSOR FOR SELECT id FROM orders WHERE status = 'x'",
                calls=3,
                total_time_ms=300.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_noise == 0
    assert len(workload.stats) == 1
    assert "DECLARE" not in workload.stats[0].sql.upper()
    assert workload.stats[0].calls == 3


def test_a_copy_subquery_is_analyzed_not_filtered():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="COPY (SELECT id FROM orders WHERE status = 'x') TO STDOUT",
                calls=1,
                total_time_ms=10.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_noise == 0
    assert len(workload.stats) == 1


def test_a_declared_cursor_over_introspection_is_still_filtered():
    """Unwrapping must not become a way to smuggle our own catalog reads into the workload."""
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="DECLARE c CURSOR FOR SELECT * FROM pg_stat_statements",
                calls=1,
                total_time_ms=1.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_noise == 1
    assert workload.stats == ()


def test_a_whole_table_copy_is_still_filtered():
    fetch = WorkloadFetch(
        rows=(RawQueryRow(sql="COPY orders TO STDOUT", calls=1, total_time_ms=1.0),),
        window_description="w",
    )
    assert ingest(fetch, "postgres").skipped_noise == 1


def test_the_unwrapped_query_is_still_redacted():
    """Redaction runs after unwrapping, so the inner literal must not survive."""
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="DECLARE c CURSOR FOR SELECT id FROM orders WHERE email = 'a@b.test'",
                calls=1,
                total_time_ms=1.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert "a@b.test" not in workload.stats[0].sql
    assert "a@b.test" not in workload.stats[0].fingerprint
