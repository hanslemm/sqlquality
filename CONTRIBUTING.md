# Contributing to sqlquality

Thanks for contributing. This is a small, test-driven codebase; the bar is that all
checks pass and behavior changes are pinned by tests.

## Setup

```bash
uv sync
uv run sqlquality --version
```

`uv sync` installs the runtime dependencies plus the `dev` group (pytest, ruff, mypy).

## The four checks

Run all four before opening a PR. CI runs the same set across Python 3.11–3.14.

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/sqlquality
```

- `pytest` — the test suite (208 tests at time of writing).
- `ruff check` — lint.
- `ruff format --check` — formatting (run `uv run ruff format src tests` to fix).
- `mypy src/sqlquality` — static types (the package ships a `py.typed` marker).

## Tests

- **Tests mirror modules 1:1.** Each `src/sqlquality/<module>.py` has a matching
  `tests/test_<module>.py`. New modules get their own test file; CLI behavior lives in
  `tests/test_cli.py`, `tests/test_lint_cli.py`, `tests/test_perf_cli.py`, etc.
- **Pin behavior with exact assertions.** Assert on concrete values — exit codes,
  finding codes, composite scores, rendered strings — not just "it didn't crash". The
  gate's correctness depends on precise deltas and exit-code semantics, so tests should
  make those explicit.
- Small fixtures live under `tests/fixtures/` (e.g. `manifest_v12.json`).

## Pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
  and PR titles (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, …).
- All four checks must be green.
- Behavior changes need tests that cover the new behavior (and the failure paths).
- Any user-visible change (new flag, changed exit code, changed output, new config key)
  needs a `CHANGELOG.md` entry under the current unreleased version, in the appropriate
  `Added` / `Changed` / `Fixed` / `BREAKING` section.
