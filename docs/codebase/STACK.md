# Technology Stack

## 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python | `code/pyproject.toml`, `code/.python-version` |
| Runtime + version | Python 3.12+ (`.python-version` pins `3.12`; `pyproject.toml` requires `>=3.12`) | `code/.python-version`, `code/pyproject.toml` |
| Package manager | `uv` | `code/uv.lock`, `code/pyproject.toml` |
| Module/build system | `pyproject.toml` plus `uv.lock`; scripts are run directly from `code/` | `code/pyproject.toml`, `code/uv.lock`, `code/run_study_multiseed.py` |

## 2) Production Dependencies and Roles

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| `numpy` | `2.3.0` | Array math, sequence handling, metrics support | `code/pyproject.toml` |
| `pandas` | `3.0.0` | CSV loading, preprocessing, result export | `code/pyproject.toml`, `code/train_price_pytorch.py` |
| `scikit-learn` | `1.7.0` | Scaling and error metrics | `code/pyproject.toml`, `code/train_price_pytorch.py` |
| `torch` | `2.8.0` | Main recurrent model training and inference | `code/pyproject.toml`, `code/train_price_pytorch.py` |
| `tensorflow` | `2.20.0` | Legacy LSTM pipeline | `code/pyproject.toml`, `code/train_price.py` |
| `lightgbm` | `4.6.0` | SOTA baseline in `train_sota_common.py` | `code/pyproject.toml`, `code/train_sota_common.py` |
| `xgboost` | `3.0.0` | SOTA baseline in `train_sota_common.py` | `code/pyproject.toml`, `code/train_sota_common.py` |
| `shap` | `0.49.1` | Explainability for model checkpoints | `code/pyproject.toml`, `code/train_price_pytorch.py` |

## 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| `ipykernel` | Interactive notebook/kernel support | `code/pyproject.toml` |
| `tqdm` | Progress bars in study and analysis scripts; not declared in `pyproject.toml` | `code/run_study_multiseed.py`, `code/analyze_study_results.py` |
| `matplotlib` | SHAP beeswarm plot generation; not declared in `pyproject.toml` | `code/make_shap_beeswarm.py` |
| `scipy` | Statistical tests in analysis; not declared in `pyproject.toml` | `code/analyze_study_results.py` |

## 4) Key Commands

```bash
uv sync
uv run python train_price_pytorch.py --bank BBCA --data-dir data --output-root results --run-shap
uv run python run_study_multiseed.py --banks BBCA,BBNI --cases lstm_data,bilstm_data
python paper/build.py
```

## 5) Environment and Config

- Config sources: `code/pyproject.toml`, `code/uv.lock`, `code/.python-version`, `code/data_macro/README.md`, `paper/main.tex`
- Required env vars: `[TODO] None found in the scan output or source files`
- Deployment/runtime constraints: local CSV files are the primary data source; paper build depends on a local LaTeX toolchain (`pdflatex`, `bibtex`)

## 6) Evidence

- `code/pyproject.toml`
- `code/.python-version`
- `code/uv.lock`
- `code/train_price_pytorch.py`
- `code/train_sota_common.py`
- `paper/build.py`
