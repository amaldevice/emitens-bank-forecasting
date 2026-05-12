# External Integrations

## 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|------------------------|---------|------------|-------------|----------|
| World Bank macro CSV snapshots | File-based data source | Inflation and exchange-rate inputs | None observed | High | `code/data_macro/README.md`, `code/build_data_with_macro.py` |
| FRED BI rate CSV snapshot | File-based data source | Policy-rate input | None observed | High | `code/data_macro/README.md`, `code/build_data_with_macro.py` |
| Yahoo Finance IHSG snapshot | File-based data source | Market index input | None observed | Medium | `code/data_macro/README.md`, `code/build_data_with_macro.py` |
| Local LaTeX toolchain | Build integration | Compile the paper PDF | None | Medium | `paper/build.py`, `paper/main.tex` |

## 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| `code/data/` | Bank-level source CSVs | `train_price_pytorch.py`, `train_sota_common.py` | Input quality and schema drift | `code/train_price_pytorch.py`, `code/train_sota_common.py` |
| `code/data_macro/` | Macro source snapshots | `build_data_with_macro.py` | Manual refresh and lag alignment | `code/data_macro/README.md`, `code/build_data_with_macro.py` |
| `code/data_with_macro/` | Enriched training inputs | `build_data_with_macro.py`, training scripts | Look-ahead leakage if alignment is wrong | `code/build_data_with_macro.py`, `code/train_price_pytorch.py` |
| `code/results_study_full/`, `code/results_sota_data/`, `code/results_sota_macro/` | Model outputs and study artifacts | Training and analysis scripts | Repo bloat and churn | `code/run_study_multiseed.py`, `code/run_ablation.py` |

## 3) Secrets and Credentials Handling

- Credential sources: `[TODO] none found`
- Hardcoding checks: paths, bank symbols, and date boundaries are mostly hardcoded or CLI arguments
- Rotation or lifecycle notes: `[TODO] no secret lifecycle exists in the repository`

## 4) Reliability and Failure Behavior

- Retry/backoff behavior: none observed
- Timeout policy: none observed
- Circuit-breaker or fallback behavior: none observed

## 5) Observability for Integrations

- Logging around external calls: `print()` output only; no structured integration log
- Metrics/tracing coverage: none observed
- Missing visibility gaps: no request IDs, no integration health checks, no retry telemetry

## 6) Evidence

- `code/build_data_with_macro.py`
- `code/data_macro/README.md`
- `code/run_study_multiseed.py`
- `paper/build.py`
- `code/train_price_pytorch.py`
