# Repository Guidelines

## Project Structure & Module Organization

This repository combines experiment code and an IEEE-style LaTeX manuscript.

Use `TRANSFORMER_VS_RECURRENT_RESULTS.md` as the current root-level reference
for Transformer versus LSTM/BiLSTM experiment outcomes. It summarizes the raw
280-run Transformer study, the existing 280-run recurrent study, and the
best-per-bank comparison metrics.

- `code/` contains Python research scripts, CSV datasets, macro data snapshots, and generated experiment outputs.
- `code/train_price_pytorch.py` is the main LSTM/BiLSTM training pipeline.
- `code/train_price_transformers.py` adds Informer/Autoformer and Lite transformer baselines.
- `code/train_price_pytorch_norm.py`, `code/train_price_pytorch_cnn.py`, and `code/train_price_pytorch_attention.py` add recurrent architecture variants.
- `code/run_all_training.py`, `run_study_multiseed.py`, `run_transformer_training.py`, `run_ablation.py`, and `run_shap_from_checkpoints.py` orchestrate batch runs and analysis.
- `code/modal_train.py` submits neural training jobs to Modal GPU infrastructure.
- `code/train_sota_common.py`, `train_sota_data.py`, and `train_sota_macro.py` contain XGBoost/LightGBM baseline workflows.
- `paper/` contains `main.tex`, section files under `paper/sections/`, images, bibliography, and `paper/build.py`.
- `docs/codebase/` contains generated repository maps and current architecture notes.

There is no dedicated `tests/` directory at present.

## Build, Test, and Development Commands

Run Python commands from `code/` unless noted.

```bash
uv sync
```

Install the Python 3.12 environment from `code/pyproject.toml` and `uv.lock`.

```bash
uv run python train_price_pytorch.py --bank BBCA --data-dir data --output-root results --seed 42
```

Run a single PyTorch training pipeline for one bank.

```bash
uv run python run_all_training.py --banks BBCA --pytorch-cases lstm_data --transformer-cases informer_data --seeds 42 --dry-run
```

Preview generated batch commands without training.

```bash
uv run python run_all_training.py --backend modal --banks BBCA --pytorch-cases lstm_data --transformer-cases informer_data --seeds 42 --epochs 1
```

Submit LSTM/BiLSTM and Informer/Autoformer jobs to Modal. Requires `modal setup`.

## Modal Cost Control

Modal can launch many GPU jobs in parallel. A full 7-bank, 5-seed, 8-case
Transformer run creates 280 GPU jobs, and enabling SHAP for every run is
expensive. Default to this staged workflow:

1. Run a small smoke test first: one bank, one seed, one epoch, no SHAP.
2. Run the full training sweep without `--run-shap` to collect metrics cheaply.
3. Aggregate metrics and identify top cases per bank.
4. Run SHAP only for selected best cases, seeds, or checkpoints.
5. Avoid `--save-model` unless checkpoint analysis is required.

Use `--run-shap` on all runs only when the user explicitly accepts the cost.

```bash
python paper/build.py
```

Build `paper/main.pdf` with `pdflatex` and `bibtex` from the repository root.

## Coding Style & Naming Conventions

Use Python 3.12+. Follow existing script style: 4-space indentation, `snake_case` files/functions, `PascalCase` classes and dataclasses, and `UPPER_SNAKE_CASE` constants. Prefer typed function signatures in newer code. Keep scripts CLI-friendly with `argparse` and `if __name__ == "__main__":` entry points. No formatter or linter config is currently defined.

## Testing Guidelines

No automated test framework or coverage threshold is configured. For changes, run the smallest relevant smoke check: `--dry-run` for orchestration edits, one-bank training for model pipeline edits, `python -m py_compile code/*.py` for syntax checks, and `python paper/build.py` for manuscript changes. Avoid committing unchecked changes to generated result schemas.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries, sometimes with Conventional Commit prefixes such as `feat:` and `refactor:`. Prefer intent-focused messages, for example `feat: add SHAP batch workflow`.

Pull requests should describe the research or paper impact, list commands run, note changed output locations, and call out any result-regeneration requirements. Include screenshots or PDF excerpts when paper figures, tables, or layout change.

## Security & Configuration Tips

The project is local-file based and has no observed secrets workflow. Do not commit credentials or machine-specific paths. Treat `code/results_*`, large CSVs, and PDFs as high-churn artifacts; verify whether they are required before adding new generated files.
