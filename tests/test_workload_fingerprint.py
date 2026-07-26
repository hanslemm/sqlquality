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
    assert not is_noise("select id from orders where status = $1")


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
