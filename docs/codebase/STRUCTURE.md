# Codebase Structure

## 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `code/` | Python research scripts, CSV datasets, macro data snapshots, and generated experiment outputs | `code/train_price_pytorch.py`, `code/build_data_with_macro.py`, `code/README.md` |
| `paper/` | IEEE LaTeX manuscript and paper build helper | `paper/main.tex`, `paper/build.py` |
| `docs/` | Generated documentation, including this codebase map | `docs/codebase/STACK.md` |
| `README.md` | Root-level project overview | `README.md` |
| `LICENSE` | Repository license | `LICENSE` |
| `rebuttal_instructions.md`, `rebuttal_response.md` | Review/rebuttal notes for the paper process | `rebuttal_instructions.md`, `rebuttal_response.md` |

## 2) Entry Points

- Main runtime entry: `code/train_price_pytorch.py`
- Secondary entry points: `code/run_all_training.py`, `code/run_study_multiseed.py`, `code/run_ablation.py`, `code/run_shap_from_checkpoints.py`, `code/train_sota_data.py`, `code/train_sota_macro.py`, `code/train_price.py`
- How entry is selected: each script has its own `if __name__ == "__main__":` block or is launched by another script via `subprocess.run`

## 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|-----------------------|
| Recurrent training pipeline | `train_price_pytorch.py` model definition, training loop, SHAP, artifact writes | Multi-bank orchestration, paper compilation, macro ETL |
| Baseline SOTA pipeline | `train_sota_common.py`, `train_sota_data.py`, `train_sota_macro.py` | Recurrent model internals |
| Experiment orchestration | `run_study_multiseed.py`, `run_ablation.py`, `run_shap_from_checkpoints.py`, `analyze_study_results.py` | Model architecture changes |
| Macro data enrichment | `build_data_with_macro.py`, `code/data_macro/README.md` | Training loop or LaTeX build logic |
| Paper source | `paper/main.tex`, `paper/sections/introduction.tex`, `paper/sections/methodology.tex`, `paper/sections/results.tex` | Python data processing or experiment execution |

## 4) Naming and Organization Rules

- File naming pattern: snake_case Python scripts such as `train_price_pytorch.py`, `run_study_multiseed.py`, and `build_data_with_macro.py`
- Directory organization pattern: domain-by-artifact, not package layers; `code/results_study_full/`, `code/results_sota_data/`, and `code/data*` are grouped by experiment output or dataset role
- Import aliasing or path conventions: scripts import sibling modules directly without a package prefix, for example `from train_sota_common import parse_args, run_pipeline`

## 5) Evidence

- `code/train_price_pytorch.py`
- `code/run_study_multiseed.py`
- `code/run_all_training.py`
- `code/train_sota_common.py`
- `code/build_data_with_macro.py`
- `paper/main.tex`
- `code/README.md`
