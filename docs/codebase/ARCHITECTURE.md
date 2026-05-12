# Architecture

## 1) Architectural Style

- Primary style: script-driven experimental pipeline with shared helpers
- Why this classification: the repository is organized around CLI scripts that load CSVs, preprocess data, train models, write artifacts, and then aggregate results for the paper
- Primary constraints: chronological evaluation, repeated-seed reproducibility, and offline CSV-based data handling

## 2) System Flow

```text
code/data/*.csv or code/data_with_macro/*.csv
  -> train_price_pytorch.py loads, sorts, scales, and windows the data
  -> recurrent model trains or checkpoint is loaded for SHAP-only runs
  -> predictions, forecasts, training history, and summary JSON are written
  -> run_study_multiseed.py / run_ablation.py repeat the run across banks, seeds, and cases
  -> analyze_study_results.py and SHAP scripts aggregate results for paper tables and figures
```

1. `train_price_pytorch.py` reads a bank CSV, checks required columns, forward-fills missing values, and standard-scales features and target.
2. The script splits the time series by date, creates fixed-length sequences, and trains an `LSTM` or `BiLSTM` model.
3. After training, it evaluates on the held-out test span, generates a future forecast by rolling the last sequence, and writes CSV/JSON artifacts.
4. If SHAP is enabled, the same checkpoint is re-used to produce feature attribution output.
5. `run_all_training.py`, `run_study_multiseed.py`, and `run_ablation.py` call training scripts repeatedly with different banks, cases, models, and seeds, then summarize the results.
6. `build_data_with_macro.py` prepares macro-enriched bank CSVs before the recurrent pipeline runs.

## 3) Layer / Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `train_price_pytorch.py` | Data loading, preprocessing, `PriceLSTM`, training, test metrics, forecasting, SHAP, artifact export | Multi-bank orchestration or paper build | `code/train_price_pytorch.py` |
| `train_sota_common.py` | XGBoost/LightGBM baseline training and shared split logic | Recurrent model internals | `code/train_sota_common.py` |
| `run_all_training.py`, `run_study_multiseed.py` | Bank/case/model/seed orchestration and summary table generation | Model architecture or feature engineering | `code/run_all_training.py`, `code/run_study_multiseed.py` |
| `build_data_with_macro.py` | Macro feature enrichment and CSV materialization | Model training or evaluation | `code/build_data_with_macro.py` |
| `paper/build.py` | LaTeX compilation sequence | Data loading or ML logic | `paper/build.py` |

## 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| Shared dataclass config | `TrainConfig` in `train_price_pytorch.py`, `SotaConfig` in `train_sota_common.py` | Keeps CLI args, paths, and hyperparameters explicit |
| Shared date splitting and sequence windowing | `train_price_pytorch.py`, `train_sota_common.py`, `make_shap_beeswarm.py` | Keeps experiment splits and windows consistent |
| Subprocess orchestration | `run_study_multiseed.py`, `run_ablation.py`, `run_shap_from_checkpoints.py` | Lets one script fan out many controlled runs |
| Checkpoint-driven post-hoc explainability | `train_price_pytorch.py`, `run_shap_from_checkpoints.py`, `make_shap_beeswarm.py` | Reuses saved models without retraining |

## 5) Known Architectural Risks

- Feature scaling is fit before the chronological split in `train_price_pytorch.py` and `train_sota_common.py`, which can leak information from validation/test periods into training preprocessing.
- `forecast_future()` repeats the last observed feature vector for every forecast step, so future exogenous variables are not modeled.
- The repository keeps a legacy TensorFlow pipeline in `train_price.py` alongside the PyTorch pipeline, which increases drift risk.
- The paper text in `paper/sections/methodology.tex` describes GA-style optimization, but the current implementation in `train_price_pytorch.py` is lightweight random search over a smaller hyperparameter set.

## 6) Evidence

- `code/train_price_pytorch.py`
- `code/train_sota_common.py`
- `code/run_all_training.py`
- `code/run_study_multiseed.py`
- `code/build_data_with_macro.py`
- `code/make_shap_beeswarm.py`
- `paper/build.py`
