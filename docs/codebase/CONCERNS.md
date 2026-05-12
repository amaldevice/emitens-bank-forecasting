# Codebase Concerns

## 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| High | Scaling is fit before the chronological split in the main training and baseline pipelines | `code/train_price_pytorch.py`, `code/train_sota_common.py` | Results can leak future information into preprocessing and overstate performance | Split first, fit scalers on train only, then transform validation/test |
| High | The paper describes GA optimization, but the current implementation is lightweight random search over a different hyperparameter set | `paper/sections/methodology.tex`, `code/train_price_pytorch.py`, `code/run_ablation.py` | Paper claims can diverge from executable behavior | Align the manuscript with the actual implementation or implement the documented GA |
| High | Several imported runtime libraries are not declared in `code/pyproject.toml` | `code/train_price_pytorch.py`, `code/run_study_multiseed.py`, `code/make_shap_beeswarm.py`, `code/analyze_study_results.py`, `code/pyproject.toml` | Clean installs may fail or be incomplete | Add missing dependencies or remove unused imports |

## 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| Legacy TensorFlow pipeline duplicates the newer PyTorch pipeline | Old revision is still kept in the repo | `code/train_price.py` | Behavior drift and maintenance overhead | Archive or remove after confirming it is no longer needed |
| Generated results live inside the source tree | Experiment outputs are committed or retained alongside code | `code/results_study_full/`, `code/results_sota_data/`, `code/results_sota_macro/` | Repo size grows quickly and diffs become noisy | Move large artifacts out of source control or add explicit ignore rules |
| Root and `code/` README files describe an older template state | Documentation was not updated after the pipeline changed | `README.md`, `code/README.md` | New contributors may follow stale instructions | Refresh docs to match the current scripts and outputs |

## 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|-------------------------------|----------|--------------------|-----|
| No secrets or auth layer was found | N/A | `code/data_macro/README.md`, `code/pyproject.toml` | Local-file workflow only | No explicit secret-management story |
| CLI path inputs are not strongly validated | A01/A05-adjacent, depending on deployment context | `code/train_price_pytorch.py`, `code/build_data_with_macro.py` | Scripts fail fast on missing files | No input sanitization or sandboxing beyond local execution |

## 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| SHAP KernelExplainer is expensive | `code/train_price_pytorch.py`, `code/run_shap_from_checkpoints.py`, `code/make_shap_beeswarm.py` | Long runtime for many checkpoints | Can dominate end-to-end study time | Cache or batch explainability outputs and use narrower default sampling |
| Future forecasts reuse the last observed feature vector | `code/train_price_pytorch.py` | Forecasts beyond the test window do not model exogenous dynamics | Longer horizons can become unrealistic | Forecast exogenous features separately or limit the horizon claim |
| Study and ablation runs are sequential subprocess loops | `code/run_study_multiseed.py`, `code/run_ablation.py` | Large wall-clock time for the full matrix | Runtime grows linearly with banks, cases, and seeds | Parallelize independent runs if compute budget allows |

## 5) Fragile / High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| `code/train_price_pytorch.py` | Central pipeline with data, model, metrics, SHAP, and artifact writes in one script | High churn in recent git history | Lock behavior with targeted smoke checks before edits |
| `paper/main.tex`, `paper/sections/methodology.tex`, `paper/sections/results.tex` | Paper claims must stay aligned with code behavior | Frequent updates in recent git history | Update manuscript only after code changes are confirmed |
| `code/run_study_multiseed.py` | Drives the full experiment matrix and downstream summaries | Recent churn in git history | Change one case at a time and validate output paths |

## 6) `[ASK USER]` Questions

1. [ASK USER] Do you want me to prioritize correctness cleanup first, starting with the leakage and dependency-drift issues?
2. [ASK USER] Should the generated result trees stay in the repository, or should I move them out and treat them as build artifacts?
3. [ASK USER] Should I update the paper and README files so they match the current PyTorch/random-search implementation instead of the older template text?

## 7) Evidence

- `code/train_price_pytorch.py`
- `code/train_sota_common.py`
- `code/run_ablation.py`
- `code/run_study_multiseed.py`
- `code/make_shap_beeswarm.py`
- `code/analyze_study_results.py`
- `paper/sections/methodology.tex`
- `README.md`
- `code/README.md`
