# Rebuttal Response

We thank all reviewers for their constructive feedback.

## Response to Reviewer A

**Strengths acknowledged:** We appreciate the recognition of our rigorous experimental design, statistical validation, and interpretability analysis.

**Comment 1:** Limited methodological novelty, no comparison with Transformers.
**Response:** Our contribution lies in establishing a rigorous benchmarking framework rather than proposing new architectures. We provide systematic ablation analysis with statistical validation that is often missing in financial forecasting literature. This foundational work enables fair comparison of future architectures including Transformers.

**Comment 2:** GA optimization not described in sufficient detail.
**Response:** We acknowledge this limitation. The GA implementation uses fixed search budget with validation loss as objective (Section 4.2). In camera-ready, we will provide detailed hyperparameter search space and GA configuration for full reproducibility.

**Comment 3:** Analysis of macroeconomic features could be deeper.
**Response:** Table 7 provides systematic RMSE delta analysis showing macro features help only specific banks (BBTN, BMRI). SHAP analysis (Section 6.4) reveals market microstructure factors dominate over macro indicators. This empirical evidence contributes to understanding when macro features are beneficial vs. detrimental.

**Comment 4:** Dataset limited to Indonesian banking stocks.
**Response:** We acknowledge this limitation. The Indonesian banking sector provides a controlled experimental environment for establishing methodology. The framework is designed for extensibility to other markets and sectors, as outlined in our future work.

## Response to Reviewer B

**Comment 1:** GA not described in abstract, problem unclear.
**Response:** We will clarify the abstract to mention GA's role in hyperparameter optimization. The problem statement (forecasting bank stock prices in emerging markets) addresses a practical need for systematic evaluation methodology in financial time series.

**Comment 2:** Method is mixed, GA role not explained.
**Response:** The GA serves as budgeted hyperparameter optimization (Section 4.2) to test whether constrained tuning improves accuracy versus default settings. This ablation factor helps isolate optimization effects from architectural choices.

**Comment 3:** Paper too specific, needs broader appeal.
**Response:** While focused on Indonesian banks, our statistical validation framework (Wilcoxon, DM tests) and multi-seed protocol establish generalizable methodology for financial forecasting evaluation. The systematic approach benefits broader time series forecasting research.

## Response to Reviewer C

**Comment 1:** Elaborate GA hyperparameters and search space.
**Response:** We acknowledge this gap. The GA optimizes hidden dimensions, learning rates, and sequence parameters within fixed budget. We will provide complete search space specification in camera-ready for reproducibility.

**Comment 2:** Discuss why With_Macro caused error spikes in BDMN.
**Response:** Table 7 shows BDMN experienced +202.86% RMSE degradation with macro features. This suggests temporal misalignment between macro indicators and BDMN's price dynamics, consistent with our SHAP analysis showing microstructure factors dominate for this bank.

**Comment 3:** Increase seed count from 5 to 10-20.
**Response:** We acknowledge the statistical power limitation. Section 6.5 explicitly notes Wilcoxon tests are underpowered at n=5. The 5-seed constraint reflects computational resource limitations, but our framework supports scaling to higher seed counts for stronger statistical validation.

**Comment 4:** Add ablation on feature lag for temporal alignment.
**Response:** This is an excellent suggestion. Testing 7- and 30-day lags could reveal whether temporal misalignment explains macro feature failures. This represents valuable future work enabled by our established framework.

## Concluding Remarks

The reviewers' suggestions for enhanced GA documentation, increased seed counts, and feature lag analysis align with our commitment to rigorous methodology. This work establishes a reproducible benchmark that enables systematic evaluation of these improvements and comparison with advanced architectures.