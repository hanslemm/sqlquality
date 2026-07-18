import pytest

from sqlquality.adapters import get_adapter
from sqlquality.adapters.postgres import PostgresAdapter, parse_pg_plan

SEQ = [
    {
        "Plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "orders",
            "Total Cost": 1234.5,
            "Plan Rows": 50000,
            "Plans": [],
        }
    }
]
SPILL = [
    {
        "Plan": {
            "Node Type": "Sort",
            "Sort Method": "external merge Disk",
            "Plans": [{"Node Type": "Seq Scan", "Relation Name": "big", "Plans": []}],
        }
    }
]
INDEX = [
    {
        "Plan": {
            "Node Type": "Index Scan",
            "Relation Name": "orders",
            "Index Name": "orders_pk",
            "Plans": [],
        }
    }
]


def test_seq_scan_flagged():
    assert "PG001" in {f.code for f in parse_pg_plan(SEQ)}


def test_sort_spill_and_child_seq_scan():
    codes = {f.code for f in parse_pg_plan(SPILL)}
    assert "PG002" in codes  # the spill
    assert "PG001" in codes  # the nested seq scan on 'big'


def test_index_scan_clean():
    assert parse_pg_plan(INDEX) == []


def test_get_adapter_postgres():
    adapter = get_adapter("postgres")
    assert isinstance(adapter, PostgresAdapter)
    assert adapter.engine == "postgres"


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError):
        get_adapter("oracle")


def test_adapter_static_findings():
    codes = {f.code for f in get_adapter("postgres").static_findings("select * from a, b")}
    assert "SQ001" in codes and "SQ002" in codes
