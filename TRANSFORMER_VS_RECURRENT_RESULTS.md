# Transformer vs Recurrent Study Results

This document consolidates the available experiment outputs for three model
families: original recurrent models, Transformer models, and the newer
recurrent architecture variants. It is a code-result artifact, not a paper
revision. The paper files under `paper/` have not been changed.

## Source Outputs

- Original recurrent source: `code/results_study_full`
- Transformer source: `code/results_transformer_study`
- Recurrent variants source: `code/results_recurrent_variants_study`
- Original recurrent + Transformer table: `code/results_transformer_study/combined_recurrent_transformer_runs.csv`
- Three-family best-per-bank table: `code/results_recurrent_variants_study/best_three_family_by_bank.csv`

## Run Coverage

| Family | Runs | Banks | Seeds | Main cases | SHAP | Checkpoints |
|---|---:|---:|---:|---|---|---|
| Original recurrent | 280 | 7 | 5 | LSTM/BiLSTM data, macro, tuned | yes | yes |
| Transformer | 280 | 7 | 5 | Informer/Autoformer data, macro, random-HPO | yes | no |
| Recurrent variants | 1400 | 7 | 5 | Norm, CNN-RNN, attention data/macro/tuned | no | no |

The original recurrent and Transformer studies both include SHAP outputs.
Recurrent variants were intentionally run without SHAP and without `model.pt`
checkpoints to keep the 1400-run sweep cheaper and lighter.

## Model Families

Original recurrent models are the existing `lstm_*` and `bilstm_*` cases from
`code/train_price_pytorch.py`.

Transformer models are the `informer_*` and `autoformer_*` cases from
`code/train_price_transformers.py`. The strongest Transformer result comes from
Informer.

Recurrent variants include:

- Norm LSTM/BiLSTM: pre-norm, post-norm, and pre+post LayerNorm variants.
- CNN-LSTM and CNN-BiLSTM: a Conv1D block before recurrent encoding.
- LSTM/BiLSTM attention: temporal attention over recurrent hidden states.

## Global Best Case Comparison

| Family | Best case | Mean RMSE | Mean MAE | Mean MAPE |
|---|---|---:|---:|---:|
| Original recurrent | `bilstm_data_ga` | 680.83 | 591.04 | 10.65% |
| Recurrent variants | `cnn_bilstm_data_ga` | 754.96 | 668.21 | 12.35% |
| Transformer | `informer_data` | 401.02 | 325.11 | 5.94% |

Transformer is the strongest global family. The recurrent variants did not beat
the original recurrent best case globally; their best case is about 10.89%
worse than `bilstm_data_ga` by RMSE.

## Best Model Per Bank

| Bank | Original recurrent | RMSE | MAE | MAPE | Recurrent variant | RMSE | MAE | MAPE | Transformer | RMSE | MAE | MAPE |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| BBCA | `bilstm_data` | 1881.53 | 1721.55 | 18.36% | `bilstm_attention_data` | 2004.10 | 1830.50 | 19.51% | `autoformer_macro` | 545.93 | 456.43 | 5.07% |
| BBNI | `bilstm_data` | 335.85 | 252.66 | 4.90% | `cnn_bilstm_data` | 362.88 | 263.19 | 5.08% | `informer_data` | 219.66 | 145.55 | 2.80% |
| BBRI | `bilstm_data` | 525.70 | 435.44 | 8.34% | `bilstm_attention_data` | 580.65 | 454.46 | 8.59% | `informer_macro` | 379.21 | 296.39 | 5.69% |
| BBTN | `bilstm_data_ga` | 51.44 | 43.48 | 3.23% | `bilstm_attention_data` | 70.84 | 58.99 | 4.35% | `informer_data_random` | 43.01 | 35.32 | 2.63% |
| BDMN | `lstm_data` | 120.72 | 98.32 | 3.74% | `cnn_bilstm_data_ga` | 116.73 | 89.56 | 3.40% | `informer_data` | 76.82 | 55.19 | 2.06% |
| BMRI | `bilstm_macro_ga` | 1388.71 | 1165.79 | 19.09% | `cnn_bilstm_macro` | 1533.72 | 1299.75 | 21.29% | `informer_data_random` | 885.21 | 727.41 | 11.85% |
| BNGA | `bilstm_data` | 244.14 | 192.77 | 11.23% | `cnn_bilstm_data_ga` | 247.60 | 196.96 | 11.48% | `informer_data` | 132.56 | 101.88 | 6.19% |

Only BDMN shows a recurrent-variant improvement over the original recurrent
best. Transformer wins every bank-level comparison.

## Average Best-Per-Bank Improvement

Positive values mean the newer family improves over the comparison family.
Negative values mean it is worse.

| Comparison | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Transformer vs original recurrent | 38.31% | 42.17% | 41.95% |
| Recurrent variants vs original recurrent | -10.18% | -7.90% | -7.41% |
| Transformer vs recurrent variants | 44.16% | 46.48% | 46.02% |

The recurrent variants expand the architectural search space, but they do not
improve the current best recurrent baseline on average. The Transformer family
keeps a wide margin against both recurrent groups.

## Original Recurrent vs Transformer Detail

| Bank | RMSE improvement | MAE improvement | MAPE improvement |
|---|---:|---:|---:|
| BBCA | 70.98% | 73.49% | 72.42% |
| BBNI | 34.60% | 42.39% | 42.89% |
| BBRI | 27.87% | 31.93% | 31.77% |
| BBTN | 16.40% | 18.77% | 18.83% |
| BDMN | 36.37% | 43.86% | 44.99% |
| BMRI | 36.26% | 37.60% | 37.93% |
| BNGA | 45.70% | 47.15% | 44.84% |

The strongest Transformer case globally is `informer_data`. `informer_data`
wins on BBNI, BDMN, and BNGA. `informer_data_random` wins on BBTN and BMRI,
which suggests fast random HPO can help for some emitents. `autoformer_macro`
wins only on BBCA.

## Top Recurrent Variant Cases

| Rank | Case | Runs | Mean RMSE | Mean MAE | Mean MAPE |
|---:|---|---:|---:|---:|---:|
| 1 | `cnn_bilstm_data_ga` | 35 | 754.96 | 668.21 | 12.35% |
| 2 | `cnn_bilstm_data` | 35 | 758.48 | 662.05 | 12.48% |
| 3 | `cnn_lstm_data` | 35 | 760.28 | 659.10 | 12.17% |
| 4 | `cnn_bilstm_macro` | 35 | 773.34 | 674.43 | 12.76% |
| 5 | `cnn_lstm_data_ga` | 35 | 788.07 | 689.19 | 12.70% |

CNN-based recurrent variants are the strongest subgroup among the new recurrent
variants. Attention variants win several bank-level variant comparisons, but
they do not beat Transformer results.

## Analysis

The central result is stable across global and bank-level views: Transformer
models are the strongest family. The margin is large enough that the conclusion
does not depend on a single bank. Transformer models reduce average best-per-bank
RMSE by 38.31% versus the original recurrent family and by 44.16% versus the
new recurrent-variant family.

The recurrent variants are still useful scientifically because they test whether
the recurrent baseline can be improved through architecture changes without
moving to Transformer models. The answer from this sweep is mostly no. The only
bank-level exception is BDMN, where `cnn_bilstm_data_ga` slightly improves over
the original recurrent best. For the other six banks, the best original
LSTM/BiLSTM result remains stronger than the best recurrent variant.

Macro features remain bank-dependent. They are not uniformly beneficial across
families. The best recurrent variant globally is data-only, while BMRI favors a
macro CNN-BiLSTM case. Transformer winners are also mixed: BBCA favors
`autoformer_macro`, BBRI favors `informer_macro`, while most others favor
data-only Informer cases.

For XAI, the original recurrent and Transformer families already have SHAP
outputs. Recurrent variants currently do not. The cost-aware next step is to run
SHAP only for the seven best recurrent-variant cases listed in the per-bank
table, not for all 1400 variant runs.

## Practical Follow-Up

Use the Transformer family as the strongest experimental extension. Keep
recurrent variants as an ablation/robustness study rather than the primary
contribution. If more compute is available, prioritize:

1. SHAP for each bank's best recurrent-variant case.
2. Statistical tests comparing the best Transformer, original recurrent, and
   recurrent-variant predictions per bank.
3. A compact paper table that reports only best-per-family metrics per bank,
   with the full 1400-run variant study kept as supporting code output.
