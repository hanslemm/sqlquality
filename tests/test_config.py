from sqlquality.config import Config, GateConfig, load_config


def test_defaults_when_no_path():
    cfg = load_config(None)
    assert cfg.gate.mode == "warn"
    assert cfg.gate.max_complexity_increase == 10.0
    assert cfg.waivers == ()


def test_defaults_when_missing_file(tmp_path):
    assert load_config(tmp_path / "nope.yml") == Config()


def test_parses_yaml(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text(
        "gate:\n"
        "  mode: fail\n"
        "  max_complexity_increase: 5.0\n"
        "waivers:\n"
        "  - model.demo.orders\n"
    )
    cfg = load_config(p)
    assert cfg.gate == GateConfig(mode="fail", max_complexity_increase=5.0)
    assert cfg.waivers == ("model.demo.orders",)


def test_partial_yaml_falls_back(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  mode: fail\n")
    cfg = load_config(p)
    assert cfg.gate.mode == "fail"
    assert cfg.gate.max_complexity_increase == 10.0  # default
    assert cfg.waivers == ()
