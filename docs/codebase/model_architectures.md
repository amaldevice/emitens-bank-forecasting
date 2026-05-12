# Model Architectures and Experiment Roles

This note maps the forecasting models available in `code/` and clarifies which
ones are paper-era models, new experiment candidates, or auxiliary baselines.

## Main Paper-Era Neural Models

### LSTM

Implemented in `code/train_price_pytorch.py` with `--model-type lstm`.
It uses a recurrent encoder over fixed-length price windows and predicts the
next-day closing price from the final hidden state.

### BiLSTM

Implemented in `code/train_price_pytorch.py` with `--model-type bilstm`.
It keeps the same training pipeline as LSTM but uses bidirectional recurrent
encoding. Existing paper results are centered on LSTM/BiLSTM ablations over
price-only data, macro-enriched data, and tuned variants.

## New Recurrent Experiment Models

These variants share the same data split, metrics, SHAP output schema, and
orchestration path as `train_price_pytorch.py`.

### Norm LSTM/BiLSTM

Implemented in `code/train_price_pytorch_norm.py`. Supported model types are
`lstm_pre_norm`, `lstm_post_norm`, `lstm_pre_post_norm`,
`bilstm_pre_norm`, `bilstm_post_norm`, and `bilstm_pre_post_norm`.
Pre-norm applies `LayerNorm` to each input time step before the recurrent
encoder. Post-norm applies `LayerNorm` to the final recurrent representation
before the prediction head.

### CNN-LSTM and CNN-BiLSTM

Implemented in `code/train_price_pytorch_cnn.py` with `--model-type cnn_lstm`
or `--model-type cnn_bilstm`. A `Conv1d` block extracts local temporal-feature
patterns before the sequence is passed to LSTM/BiLSTM.

### Attention LSTM/BiLSTM

Implemented in `code/train_price_pytorch_attention.py` with
`--model-type lstm_attention` or `--model-type bilstm_attention`. The recurrent
encoder returns all time-step states, then a learned temporal attention layer
aggregates them before the prediction head.

## New Transformer Experiment Models

### Informer

Implemented in `code/train_price_transformers.py` with `--model-type informer`.
This is now the default Informer path for new transformer experiments. It uses
the original Informer core ideas: ProbSparse self-attention and encoder
distilling. The implementation is adapted to this repository's next-day scalar
forecasting task rather than the original long-horizon decoder benchmark setup.

Run example:

```bash
python train_price_transformers.py --bank BBCA --model-type informer --data-dir data
```

### Autoformer

Implemented in `code/train_price_transformers.py` with `--model-type autoformer`.
This is now the default Autoformer path for new transformer experiments. It
uses moving-average series decomposition blocks and an FFT-based
Auto-Correlation mechanism, adapted to next-day closing-price prediction.

Run example:

```bash
python train_price_transformers.py --bank BBCA --model-type autoformer --data-dir data_with_macro --use-macro
```

## Lightweight Transformer Baselines

### InformerLite

Implemented with `--model-type informer_lite`. This is the earlier
Informer-style baseline: standard Transformer encoder layers with positional
encoding and optional sequence distilling. It does not use ProbSparse attention.

### AutoformerLite

Implemented with `--model-type autoformer_lite`. This is the earlier
Autoformer-style baseline: moving-average decomposition plus a standard
Transformer encoder and trend/seasonal heads. It does not use the original
Auto-Correlation mechanism.

## Tree-Based Baselines Present in Code

### XGBoost and LightGBM

Implemented through `code/train_sota_common.py`, `code/train_sota_data.py`, and
`code/train_sota_macro.py`. These flatten time windows into tabular features,
run random-search tuning, and optionally compute TreeSHAP explanations. They
are useful as SOTA-style baselines but are separate from the neural model
pipeline used in the paper-era results.

## Orchestration

`code/run_all_training.py` runs recurrent, transformer, and tree baselines.
Recurrent variant cases are selected through `--pytorch-cases`, for example
`cnn_lstm_data`, `bilstm_attention_macro`, or `lstm_pre_post_norm_data_ga`.
Transformer cases `informer_*` and `autoformer_*` now refer to the full
Informer/Autoformer experiment paths. Use `informer_lite_*` or
`autoformer_lite_*` when the older lightweight baselines are desired.

Transformer cases ending in `_random` run a fast initial HPO pass using the same
search dimensions as the LSTM/BiLSTM tuner: hidden size, learning rate, batch
size, and dropout. Transformer cases ending in `_ga` run genetic-algorithm HPO
over hidden size, learning rate, batch size, dropout, sequence length, and
encoder depth. The global `--transformer-hpo-method random` option can force
random search for all optimized transformer cases.

Statistical validation uses `code/analyze_study_results.py`. The transformer
runner writes `transformer_study_runs.csv` with the same columns required by the
Wilcoxon and Diebold-Mariano analysis path: `bank`, `case`, `seed`, metrics, and
`test_predictions_path`. SHAP/XAI uses the same KernelExplainer helper as the
LSTM/BiLSTM pipeline and is enabled with `--run-shap`. Transformer runs write
both transformer-specific output names and PyTorch-compatible names such as
`shap_kernel_feature_importance.csv`.

Examples:

```bash
python run_all_training.py --pytorch-cases cnn_lstm_data,bilstm_attention_macro --skip-transformer --skip-sota
python run_all_training.py --run-transformer --transformer-cases informer_data,autoformer_data
python run_transformer_training.py --cases informer_lite_data,autoformer_lite_data
python run_all_training.py --skip-pytorch --skip-sota --transformer-cases informer_data_random,autoformer_data_random
python run_all_training.py --skip-pytorch --skip-sota --transformer-cases informer_data,informer_data_ga --run-shap --run-statistics
```
