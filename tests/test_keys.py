from sqlquality.keys import dist_sort_findings, filter_columns, join_key_columns


def test_join_key_columns():
    assert join_key_columns(
        "select * from orders o join customers c on c.id = o.customer_id", "redshift"
    ) == ["customer_id", "id"]
    assert join_key_columns(
        "select * from a join b on a.x = b.y and a.k = b.k", "redshift"
    ) == ["k", "x", "y"]
    assert join_key_columns("select * from a, b", "redshift") == []
    assert join_key_columns("select * from a join b on a.x > b.y", "redshift") == []


def test_filter_columns():
    assert filter_columns("select * from t where created_at > '2024-01-01'", "redshift") == [
        "created_at"
    ]
    assert filter_columns(
        "select * from t where status = 'x' and amount >= 100", "redshift"
    ) == ["amount", "status"]
    assert filter_columns(
        "select * from t where ts between '2024-01-01' and '2024-02-01'", "redshift"
    ) == ["ts"]
    assert filter_columns("select * from t where a = b", "redshift") == []
    assert filter_columns("select * from t", "redshift") == []


def test_dist_sort_findings():
    findings = dist_sort_findings(
        "select * from orders o join customers c on c.id = o.customer_id "
        "where o.created_at > '2024-01-01'",
        "redshift",
    )
    codes = {f.code for f in findings}
    assert codes == {"RS001", "RS002"}
    rs001 = next(f for f in findings if f.code == "RS001")
    assert "customer_id" in rs001.message


def test_dist_sort_no_findings_when_nothing():
    assert dist_sort_findings("select id from t", "redshift") == []


def test_dist_sort_parse_error_silent():
    assert dist_sort_findings("select from where", "redshift") == []
