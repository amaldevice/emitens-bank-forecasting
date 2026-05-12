# Testing Patterns

## 1) Test Stack and Commands

- Primary test framework: `[TODO] none detected`
- Assertion/mocking tools: `[TODO] none detected`
- Commands:

```bash
[TODO] No test runner is configured in the repository
uv run python train_price_pytorch.py --bank BBCA --data-dir data --output-root results --seed 42
uv run python run_all_training.py --banks BBCA --models lstm --seeds 42 --dry-run
uv run python run_study_multiseed.py --banks BBCA --cases lstm_data --seeds 42
python paper/build.py
```

## 2) Test Layout

- Test file placement pattern: no `tests/`, `__tests__/`, or similarly named test tree was found
- Naming convention: `[TODO] none detected`
- Setup files and where they run: `[TODO] none detected`

## 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | No | N/A | No unit test files or test runner config were found |
| Integration | No | N/A | Validation is done by running the scripts directly |
| E2E | No | N/A | The nearest equivalent is full pipeline execution plus artifact inspection |

## 4) Mocking and Isolation Strategy

- Main mocking approach: none found
- Isolation guarantees: none found
- Common failure mode in tests: script failures surface at runtime because there is no test harness around them

## 5) Coverage and Quality Signals

- Coverage tool + threshold: `[TODO] none configured`
- Current reported coverage: `[TODO] none reported`
- Known gaps/flaky areas: no automated regression suite, no fixture-based data isolation, and no contract tests for the generated CSV schemas

## 6) Evidence

- `code/train_price_pytorch.py`
- `code/run_all_training.py`
- `code/run_study_multiseed.py`
- `code/run_shap_from_checkpoints.py`
- `paper/build.py`
- `code/pyproject.toml`
