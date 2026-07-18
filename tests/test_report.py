from sqlquality.delta import ModelDelta
from sqlquality.gate import GateReport
from sqlquality.report import gate_payload, render_html

REPORT = GateReport(
    deltas=[ModelDelta("model.demo.orders", 10.7, 11.1, 0.4, False)],
    regressions=[],
    passed=True,
)


def test_gate_payload_shape():
    payload = gate_payload(REPORT, neighbors=["model.demo.stg_orders"])
    assert payload["passed"] is True
    assert payload["regressions"] == []
    assert payload["neighbors"] == ["model.demo.stg_orders"]
    assert payload["models"][0] == {
        "unique_id": "model.demo.orders",
        "baseline": 10.7,
        "candidate": 11.1,
        "delta": 0.4,
        "is_new": False,
    }


def test_render_html_is_self_contained():
    html = render_html(REPORT)
    assert html.strip().startswith("<!doctype html>")
    assert "model.demo.orders" in html
    assert "http://" not in html and "https://" not in html  # no external assets
    assert "PASS" in html.upper()
