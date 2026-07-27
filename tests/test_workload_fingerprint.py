import sqlglot

from sqlquality.models import RawQueryRow, WorkloadFetch
from sqlquality.workload.fingerprint import (
    FLAG_LEADING_WILDCARD_LIKE,
    FLAG_SELECT_STAR,
    ingest,
    is_noise,
    literal_flags,
    redact_tree,
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


def test_is_noise_also_discards_predicate_bearing_declare_and_copy():
    """A documented loss, pinned so it cannot become an undocumented one.

    `_LEADING_NOISE` is a statement-prefix filter, so a cursor declaration or a COPY that
    wraps a real SELECT — with real predicates — is discarded whole. Django's
    `QuerySet.iterator()` emits the first form. Unwrapping to the inner SELECT is a
    follow-up; until then this is a README limitation and the skip counter says only
    "filtered", never "introspection/DDL".
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
