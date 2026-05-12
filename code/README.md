# Training Code Guide

This directory contains the experiment code for Indonesian banking stock-price forecasting. Commands below should be run from `code/`.

## Main Pipelines

- `train_price_pytorch.py`: single-run LSTM/BiLSTM training.
- `train_price_pytorch_norm.py`: single-run pre/post LayerNorm LSTM/BiLSTM variants.
- `train_price_pytorch_cnn.py`: single-run CNN-LSTM/CNN-BiLSTM variants.
- `train_price_pytorch_attention.py`: single-run LSTM/BiLSTM temporal-attention variants.
- `train_price_transformers.py`: single-run Informer/Autoformer and Lite transformer training.
- `run_study_multiseed.py`: multi-bank/multi-seed LSTM/BiLSTM runner.
- `run_transformer_training.py`: multi-bank/multi-seed Informer/Autoformer runner.
- `run_all_training.py`: full local or Modal-backed orchestration.
- `train_sota_data.py` and `train_sota_macro.py`: XGBoost/LightGBM baselines.
- `build_data_with_macro.py`: builds `data_with_macro/` from `data/` plus `data_macro/`.

## Model Families

`train_price_pytorch.py` supports:

- `--model-type lstm`
- `--model-type bilstm`

Additional PyTorch recurrent entrypoints support:

- `train_price_pytorch_norm.py`: `lstm_pre_norm`, `lstm_post_norm`, `lstm_pre_post_norm`, `bilstm_pre_norm`, `bilstm_post_norm`, `bilstm_pre_post_norm`.
- `train_price_pytorch_cnn.py`: `cnn_lstm`, `cnn_bilstm`.
- `train_price_pytorch_attention.py`: `lstm_attention`, `bilstm_attention`.

`train_price_transformers.py` supports:

- `--model-type informer`: Informer experiment path with ProbSparse self-attention and encoder distilling.
- `--model-type autoformer`: Autoformer experiment path with moving-average decomposition blocks and FFT-based Auto-Correlation.
- `--model-type informer_lite`: earlier Informer-style baseline with standard Transformer attention.
- `--model-type autoformer_lite`: earlier Autoformer-style baseline with standard Transformer attention.

The `informer` and `autoformer` paths now implement the core mechanisms from
the original papers while adapting them to this repository's next-day scalar
forecasting target. The `_lite` variants keep the earlier lightweight
Transformer-style baselines.

## Local Training Examples

Single bank, LSTM:

```bash
python train_price_pytorch.py --bank BBCA --model-type lstm --data-dir data --epochs 50
```

Single bank, Autoformer with macro features:

```bash
python train_price_transformers.py --bank BBCA --model-type autoformer --data-dir data_with_macro --use-macro --epochs 50
```

Dry-run the full suite without training:

```bash
python run_all_training.py --dry-run
```

Small smoke run:

```bash
python run_all_training.py --banks BBCA --pytorch-cases lstm_data --transformer-cases informer_data --seeds 42 --epochs 1 --sota-trials 1
```

## Full Orchestration Arguments

Common `run_all_training.py` arguments:

- `--banks BBCA,BBNI`: select banks.
- `--pytorch-cases lstm_data,bilstm_macro`: select LSTM/BiLSTM cases.
- `--transformer-cases informer_data,autoformer_macro`: select Informer/Autoformer cases.
- `--seeds 42,52,62`: select random seeds.
- `--epochs 50`: training epochs for neural models.
- `--sota-trials 50`: random-search trials for XGBoost/LightGBM.
- `--run-shap`: generate SHAP outputs.
- `--save-model`: persist model checkpoints.
- `--run-statistics`: run Wilcoxon and Diebold-Mariano summaries from generated run tables.
- `--transformer-hpo-method ga|random`: choose GA or random-search HPO for transformer `_ga` cases.
- `--transformer-hpo-generations 5`: GA generations for transformer HPO.
- `--skip-pytorch`, `--skip-transformer`, `--skip-sota`, `--skip-macro-data`: disable parts of the suite.

Available PyTorch cases:

```text
lstm_data,lstm_macro,bilstm_data,bilstm_macro,
lstm_data_ga,lstm_macro_ga,bilstm_data_ga,bilstm_macro_ga,
lstm_pre_norm_data,lstm_pre_norm_macro,lstm_pre_norm_data_ga,lstm_pre_norm_macro_ga,
lstm_post_norm_data,lstm_post_norm_macro,lstm_post_norm_data_ga,lstm_post_norm_macro_ga,
lstm_pre_post_norm_data,lstm_pre_post_norm_macro,lstm_pre_post_norm_data_ga,lstm_pre_post_norm_macro_ga,
bilstm_pre_norm_data,bilstm_pre_norm_macro,bilstm_pre_norm_data_ga,bilstm_pre_norm_macro_ga,
bilstm_post_norm_data,bilstm_post_norm_macro,bilstm_post_norm_data_ga,bilstm_post_norm_macro_ga,
bilstm_pre_post_norm_data,bilstm_pre_post_norm_macro,bilstm_pre_post_norm_data_ga,bilstm_pre_post_norm_macro_ga,
cnn_lstm_data,cnn_lstm_macro,cnn_lstm_data_ga,cnn_lstm_macro_ga,
cnn_bilstm_data,cnn_bilstm_macro,cnn_bilstm_data_ga,cnn_bilstm_macro_ga,
lstm_attention_data,lstm_attention_macro,lstm_attention_data_ga,lstm_attention_macro_ga,
bilstm_attention_data,bilstm_attention_macro,bilstm_attention_data_ga,bilstm_attention_macro_ga
```

For PyTorch recurrent models, `_random` is accepted as an alias for `_ga`
because the current recurrent optimizer is a lightweight random search.

Available transformer cases:

```text
informer_data,informer_macro,autoformer_data,autoformer_macro,
informer_data_random,informer_macro_random,autoformer_data_random,autoformer_macro_random,
informer_data_ga,informer_macro_ga,autoformer_data_ga,autoformer_macro_ga,
informer_lite_data,informer_lite_macro,autoformer_lite_data,autoformer_lite_macro,
informer_lite_data_random,informer_lite_macro_random,autoformer_lite_data_random,autoformer_lite_macro_random,
informer_lite_data_ga,informer_lite_macro_ga,autoformer_lite_data_ga,autoformer_lite_macro_ga
```

The `informer_*` and `autoformer_*` cases use the full paper-inspired
implementations. The earlier lightweight baselines remain available as
`informer_lite_*` and `autoformer_lite_*` cases.

Cases ending in `_random` enable fast budgeted random search over the same
configuration dimensions used by the LSTM/BiLSTM tuner: hidden size, learning
rate, batch size, and dropout. Cases ending in `_ga` enable genetic-algorithm
HPO over hidden size, learning rate, batch size, dropout, sequence length, and
encoder depth. Use `--transformer-hpo-method random` to force random search for
all optimized transformer cases.

Run transformer training with GA, SHAP, and statistical validation:

```bash
python run_all_training.py --skip-pytorch --skip-sota --transformer-cases informer_data,informer_data_ga,autoformer_data,autoformer_data_ga --run-shap --run-statistics
```

Run a faster initial transformer sweep:

```bash
python run_all_training.py --skip-pytorch --skip-sota --transformer-cases informer_data_random,autoformer_data_random --tune-candidates 6
```

## Modal GPU Backend

`run_all_training.py` defaults to local execution. Use Modal for GPU-backed neural training:

```bash
python run_all_training.py --backend modal --modal-gpu L4
```

Small Modal smoke run:

```bash
python run_all_training.py --backend modal --banks BBCA --pytorch-cases lstm_data --transformer-cases informer_data --seeds 42 --epochs 1 --modal-gpu L4
```

Modal transformer GA smoke run:

```bash
python run_all_training.py --backend modal --skip-pytorch --modal-skip-local-sota --banks BBCA --transformer-cases informer_data_ga,autoformer_data_ga --seeds 42 --epochs 1 --tune-candidates 4 --transformer-hpo-generations 2 --modal-gpu L4
```

Skip local SOTA jobs when using Modal:

```bash
python run_all_training.py --backend modal --modal-skip-local-sota
```

Modal-specific arguments:

- `--backend local|modal`: choose local subprocesses or Modal submission.
- `--modal-gpu L4`: GPU type requested by Modal. Examples: `T4`, `L4`, `A10`, `L40S`, `A100`, `H100`.
- `--modal-run-name smoke_test`: output namespace inside the Modal Volume.
- `--modal-skip-local-sota`: skip CPU SOTA jobs after Modal neural jobs are submitted.

Modal outputs are committed to the Volume:

```text
siml-stock-training-outputs
```

Before running Modal jobs, install and authenticate Modal locally:

```bash
pip install modal
modal setup
```

## Outputs

Local PyTorch outputs are written under the configured output root, usually `results_full_suite/`. Transformer outputs default to `results_transformer_study/`.

Single-run output files:

- `summary_pytorch.json` or `summary_transformer.json`
- `test_predictions_pytorch.csv` or `test_predictions_transformer.csv`
- `forecast_pytorch.csv` or `forecast_transformer.csv`
- `training_history_pytorch.csv` or `training_history_transformer.csv`
- optional checkpoint: `model.pt` or `model_transformer.pt`
- optional SHAP CSV

Transformer runs write both transformer-specific names and PyTorch-compatible
names, including `shap_kernel_feature_importance.csv`, so downstream scripts can
read paths shaped like `results_study_full/<BANK>/<case>/seed_<seed>/<BANK>/`.

## Notes

Neural models use CUDA automatically when PyTorch detects it:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

XGBoost/LightGBM baselines currently run as CPU workflows; GPU parameters are not configured in `train_sota_common.py`.
