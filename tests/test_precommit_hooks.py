from pathlib import Path

import yaml

HOOKS = Path(__file__).parent.parent / ".pre-commit-hooks.yaml"


def test_precommit_hooks_defines_lint():
    data = yaml.safe_load(HOOKS.read_text())
    hooks = {h["id"]: h for h in data}
    assert "sqlquality-lint" in hooks
    hook = hooks["sqlquality-lint"]
    assert hook["entry"] == "sqlquality lint"
    assert hook["language"] == "python"
    assert hook["files"] == r"\.sql$"
