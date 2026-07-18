from sqlquality.delta import ModelDelta
from sqlquality.gate import GateReport
from sqlquality.report import render_markdown

PASS = GateReport(
    deltas=[ModelDelta("model.demo.orders", 10.7, 11.1, 0.4, False)],
    regressions=[],
    passed=True,
)
FAIL = GateReport(
    deltas=[ModelDelta("model.demo.orders", 5.0, 30.0, 25.0, False)],
    regressions=["model.demo.orders"],
    passed=False,
)


def test_markdown_pass():
    md = render_markdown(PASS)
    assert "# sqlquality" in md
    assert "PASS" in md
    assert "model.demo.orders" in md
    assert "| model |" in md  # a markdown table header


def test_markdown_fail_marks_regression():
    md = render_markdown(FAIL)
    assert "FAIL" in md
    assert "model.demo.orders" in md


def test_markdown_lists_skipped():
    md = render_markdown(PASS, skipped=[("seed.demo.raw", "no compiled SQL")])
    assert "seed.demo.raw" in md
    assert "no compiled SQL" in md
