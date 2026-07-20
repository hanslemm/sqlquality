import pytest

from sqlquality.config import Config, ConfigError, GateConfig, load_config


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
        "gate:\n  mode: fail\n  max_complexity_increase: 5.0\nwaivers:\n  - model.demo.orders\n"
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


def test_scalar_waivers_coerced_to_single_element(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("waivers: model.demo.orders\n")
    cfg = load_config(p)
    assert cfg.waivers == ("model.demo.orders",)


def test_null_gate_falls_back_to_defaults(tmp_path):
    # A fully-commented-out gate block yields `gate: null` -> defaults, no error.
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n")
    cfg = load_config(p)
    assert cfg.gate == GateConfig()


def test_scalar_gate_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate: fail\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_list_gate_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  - fail\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_mode_typo_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  mode: fial\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_non_numeric_threshold_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  max_complexity_increase: soon\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_malformed_yaml_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate: [unterminated\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_unknown_keys_are_tolerated(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  mode: fail\n  future_flag: true\ntop_level: 1\n")
    cfg = load_config(p)
    assert cfg.gate.mode == "fail"


def test_top_level_list_doc_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(p)


def test_top_level_string_doc_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("just a string\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(p)


def test_empty_doc_falls_back_to_defaults(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("")
    assert load_config(p) == Config()


def test_nan_threshold_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  max_complexity_increase: .nan\n")
    with pytest.raises(ConfigError, match="finite"):
        load_config(p)


def test_inf_threshold_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  max_complexity_increase: .inf\n")
    with pytest.raises(ConfigError, match="finite"):
        load_config(p)


def test_bool_threshold_raises(tmp_path):
    p = tmp_path / "sqlquality.yml"
    p.write_text("gate:\n  max_complexity_increase: true\n")
    with pytest.raises(ConfigError):
        load_config(p)
