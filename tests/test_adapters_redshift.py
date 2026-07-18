from sqlquality.adapters import get_adapter
from sqlquality.adapters.redshift import RedshiftAdapter, parse_redshift_plan

BCAST = "XN Hash Join DS_BCAST_INNER  (cost=0.00..1.00)\n  XN Seq Scan on orders\n"
DISTBOTH = "XN Hash Join DS_DIST_BOTH  (cost=0.00..2.00)\n"
NESTED = "XN Nested Loop  (cost=0.00..9.00)\n  XN Seq Scan on a\n"
CLEAN = "XN Merge Join DS_DIST_NONE  (cost=0.00..1.00)\n"
ALLINNER = "XN Hash Join DS_DIST_ALL_INNER  (cost=0.00..3.00)\n"


def test_parse_redshift_markers():
    assert "RS010" in {f.code for f in parse_redshift_plan(BCAST)}
    assert "RS011" in {f.code for f in parse_redshift_plan(DISTBOTH)}
    assert "RS013" in {f.code for f in parse_redshift_plan(NESTED)}
    assert "RS012" in {f.code for f in parse_redshift_plan(ALLINNER)}


def test_parse_redshift_clean():
    assert parse_redshift_plan(CLEAN) == []


def test_get_adapter_redshift():
    adapter = get_adapter("redshift")
    assert isinstance(adapter, RedshiftAdapter)
    assert adapter.engine == "redshift"


def test_redshift_static_findings_include_dist_sort_and_antipatterns():
    codes = {
        f.code
        for f in get_adapter("redshift").static_findings(
            "select * from orders o join customers c on c.id = o.customer_id "
            "where o.created_at > '2024-01-01'"
        )
    }
    assert "SQ001" in codes  # SELECT * anti-pattern (dialect-agnostic)
    assert "RS001" in codes  # DISTKEY suggestion
    assert "RS002" in codes  # SORTKEY suggestion


def test_redshift_plan_findings_via_adapter():
    assert "RS010" in {f.code for f in get_adapter("redshift").plan_findings(BCAST)}
