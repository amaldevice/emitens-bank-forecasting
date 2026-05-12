# Coding Conventions

## 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files | Snake case for Python scripts and generated outputs | `train_price_pytorch.py`, `run_shap_from_checkpoints.py`, `forecast_pytorch.csv` | `code/train_price_pytorch.py`, `code/run_shap_from_checkpoints.py` |
| Functions/methods | Snake case | `build_dataloaders`, `run_epoch`, `forecast_future` | `code/train_price_pytorch.py` |
| Types/interfaces | PascalCase for classes and dataclasses | `TrainConfig`, `SotaConfig`, `PriceLSTM` | `code/train_price_pytorch.py`, `code/train_sota_common.py` |
| Constants/env vars | Upper snake case | `TARGET_COL`, `FEATURE_COLS`, `MACRO_FEATURE_COLS` | `code/train_price_pytorch.py` |

## 2) Formatting and Linting

- Formatter: `[TODO] No formatter config file was detected in the scan output`
- Linter: `[TODO] No linter config file was detected in the scan output`
- Most relevant enforced rules: explicit type hints in the newer scripts, direct exception raising on missing inputs, and print-based progress output
- Run commands: `[TODO] Not configured in repo; use direct script execution with `python` or `uv run python``

## 3) Import and Module Conventions

- Import grouping/order: standard library first, then third-party packages, then local sibling imports
- Alias vs relative import policy: sibling modules are imported directly by name, not via a package prefix or relative import
- Public exports/barrel policy: no barrel exports or package `__init__.py` aggregation pattern is present in the scanned code

## 4) Error and Logging Conventions

- Error strategy by layer: file and configuration problems raise `FileNotFoundError` or `ValueError`; subprocess failures bubble through `check=True`
- Logging style and required context fields: `print()` is used for progress, metrics, and artifact paths; there is no structured logging library
- Sensitive-data redaction rules: `[TODO] no secrets, auth, or PII handling pattern was found in source`

## 5) Testing Conventions

- Test file naming/location rule: `[TODO] No dedicated test directory or test file pattern was found`
- Mocking strategy norm: `[TODO] No mocking framework or test harness was found`
- Coverage expectation: `[TODO] No coverage threshold is configured`

## 6) Evidence

- `code/train_price_pytorch.py`
- `code/train_sota_common.py`
- `code/run_shap_from_checkpoints.py`
- `code/pyproject.toml`
- `code/train_price.py`
