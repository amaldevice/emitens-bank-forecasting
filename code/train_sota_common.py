from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


TARGET_COL = "Closing Price"
FEATURE_COLS = [
    "Ask Price",
    "Bid Price",
    "High Price",
    "Low Price",
    "Moving Average 20 Day",
    "Price to Sales Ratio",
    "Price to Book Ratio",
    "Price Earnings Ratio",
    "Dividend Indicated Yld - Gross",
    "Price Change 1 Day Percent",
]

MACRO_FEATURE_COLS = [
    "inflation_rate",
    "cpi_index",
    "bi_rate",
    "exchange_rate_usd_idr",
    "ihsg_close",
    "ihsg_return_1d",
]


@dataclass
class SotaConfig:
    bank_name: str
    use_macro: bool
    data_dir: Path
    output_root: Path
    time_steps: int = 60
    trials: int = 50
    seeds: tuple[int, ...] = (42, 52, 62, 72, 82)
    train_start: str = "2014-01-02"
    val_start: str = "2020-01-02"
    test_start: str = "2022-01-02"
    test_end: str = "2024-12-31"
    run_shap: bool = False
    shap_eval_size: int = 256


def parse_args(default_data_dir: str, default_output_root: str, use_macro: bool) -> SotaConfig:
    parser = argparse.ArgumentParser(description="Run SOTA baselines (XGBoost/LightGBM) for stock-price prediction.")
    parser.add_argument("--bank", type=str, required=True, help="Bank symbol, e.g. BBCA")
    parser.add_argument("--data-dir", type=str, default=default_data_dir)
    parser.add_argument("--output-root", type=str, default=default_output_root)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--trials", type=int, default=50, help="Random-search trials per model and seed.")
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82", help="Comma-separated seeds.")
    parser.add_argument("--train-start", type=str, default="2014-01-02")
    parser.add_argument("--val-start", type=str, default="2020-01-02")
    parser.add_argument("--test-start", type=str, default="2022-01-02")
    parser.add_argument("--test-end", type=str, default="2024-12-31")
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-eval-size", type=int, default=256)
    args = parser.parse_args()

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    if not seeds:
        raise ValueError("At least one seed is required.")

    return SotaConfig(
        bank_name=args.bank.strip().upper(),
        use_macro=use_macro,
        data_dir=Path(args.data_dir),
        output_root=Path(args.output_root),
        time_steps=args.time_steps,
        trials=max(1, int(args.trials)),
        seeds=seeds,
        train_start=args.train_start,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        run_shap=args.run_shap,
        shap_eval_size=max(1, int(args.shap_eval_size)),
    )


def split_dataset_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    return df[(df.index >= start_date) & (df.index < end_date)]


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_seq, y_seq = [], []
    features = df[feature_cols].values
    targets = df[target_col].values
    for i in range(time_steps, len(df)):
        x_seq.append(features[i - time_steps : i])
        y_seq.append(targets[i])
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def prepare_data(config: SotaConfig, feature_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    csv_path = config.data_dir / f"{config.bank_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in feature_cols + [TARGET_COL, "Date"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.ffill().dropna()

    selected = df[feature_cols + [TARGET_COL]].copy()
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    selected[feature_cols] = feature_scaler.fit_transform(selected[feature_cols].values)
    selected[TARGET_COL] = target_scaler.fit_transform(selected[[TARGET_COL]]).flatten()
    return selected, target_scaler


def to_flat_features(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae), "mape": mape}


def sample_xgb_params(rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "n_estimators": int(rng.integers(300, 2001)),
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.1])),
        "max_depth": int(rng.choice([3, 4, 5, 6, 8])),
        "min_child_weight": float(rng.choice([1, 2, 4, 6, 8])),
        "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 0.01, 0.1, 0.5, 1.0])),
        "reg_lambda": float(rng.choice([1.0, 2.0, 5.0, 10.0])),
        "gamma": float(rng.choice([0.0, 0.1, 0.2])),
    }


def sample_lgbm_params(rng: np.random.Generator) -> dict[str, float | int]:
    return {
        "n_estimators": int(rng.integers(300, 2001)),
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.1])),
        "num_leaves": int(rng.choice([31, 63, 127])),
        "max_depth": int(rng.choice([-1, 5, 8, 12])),
        "min_child_samples": int(rng.choice([10, 20, 40, 60])),
        "subsample": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 0.01, 0.1, 0.5, 1.0])),
        "reg_lambda": float(rng.choice([1.0, 2.0, 5.0, 10.0])),
    }


def fit_predict_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_target: np.ndarray,
    params: dict[str, float | int],
    seed: int,
) -> np.ndarray:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Install it with: uv pip install xgboost") from exc

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=100,
        **params,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model.predict(x_target)


def fit_xgb_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, float | int],
    seed: int,
):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Install it with: uv pip install xgboost") from exc

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=100,
        **params,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def fit_predict_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_target: np.ndarray,
    params: dict[str, float | int],
    seed: int,
) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is not installed. Install it with: uv pip install lightgbm") from exc

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return model.predict(x_target)


def fit_lgbm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, float | int],
    seed: int,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is not installed. Install it with: uv pip install lightgbm") from exc

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return model


def compute_tree_shap_importance(
    model_name: str,
    model,
    x_eval: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is not installed. Install it with: uv pip install shap") from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    importance = np.mean(np.abs(shap_values), axis=0)
    out = pd.DataFrame({"feature": feature_names, "importance": importance, "model": model_name})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def run_search(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_val_raw: np.ndarray,
    y_test_raw: np.ndarray,
    target_scaler: StandardScaler,
    trials: int,
    seed: int,
) -> tuple[dict[str, float | int], dict[str, float], np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed + (0 if model_name == "xgb" else 10000))
    best_mae = float("inf")
    best_params: dict[str, float | int] = {}
    best_pred_raw: np.ndarray | None = None
    rows: list[dict[str, object]] = []

    for trial in range(1, trials + 1):
        params = sample_xgb_params(rng) if model_name == "xgb" else sample_lgbm_params(rng)
        if model_name == "xgb":
            val_pred = fit_predict_xgb(x_train, y_train, x_val, y_val, x_val, params, seed)
            test_pred = fit_predict_xgb(x_train, y_train, x_val, y_val, x_test, params, seed)
        else:
            val_pred = fit_predict_lgbm(x_train, y_train, x_val, y_val, x_val, params, seed)
            test_pred = fit_predict_lgbm(x_train, y_train, x_val, y_val, x_test, params, seed)

        val_pred_raw = target_scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
        val_mae = float(mean_absolute_error(y_val_raw, val_pred_raw))
        rows.append({"trial": trial, "val_mae": val_mae, "params": json.dumps(params)})

        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params
            best_pred_raw = target_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

        print(f"[{model_name.upper()}] seed={seed} trial={trial:03d}/{trials} val_mae={val_mae:.4f}")

    if best_pred_raw is None:
        raise RuntimeError("No model candidate was successfully evaluated.")

    metrics = compute_metrics(y_test_raw, best_pred_raw)
    trial_df = pd.DataFrame(rows)
    return best_params, metrics, best_pred_raw, trial_df


def run_pipeline(config: SotaConfig) -> None:
    feature_cols = FEATURE_COLS + MACRO_FEATURE_COLS if config.use_macro else FEATURE_COLS
    print(f"Bank: {config.bank_name} | use_macro={config.use_macro}")
    print(f"Data dir: {config.data_dir} | Output root: {config.output_root}")
    print(f"Seeds: {config.seeds} | Trials/model/seed: {config.trials}")

    selected_data, target_scaler = prepare_data(config, feature_cols)
    train_set = split_dataset_by_date(selected_data, config.train_start, config.val_start)
    val_set = split_dataset_by_date(selected_data, config.val_start, config.test_start)
    test_set = split_dataset_by_date(selected_data, config.test_start, config.test_end)
    print(f"Split sizes | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    x_train_3d, y_train = create_sequences(train_set, feature_cols, TARGET_COL, config.time_steps)
    x_val_3d, y_val = create_sequences(val_set, feature_cols, TARGET_COL, config.time_steps)
    x_test_3d, y_test = create_sequences(test_set, feature_cols, TARGET_COL, config.time_steps)
    if len(x_train_3d) == 0 or len(x_val_3d) == 0 or len(x_test_3d) == 0:
        raise ValueError("Insufficient sequence samples. Reduce --time-steps or verify date ranges.")

    x_train = to_flat_features(x_train_3d)
    x_val = to_flat_features(x_val_3d)
    x_test = to_flat_features(x_test_3d)
    flat_feature_names = [f"t{t:02d}_{name}" for t in range(config.time_steps) for name in feature_cols]

    y_val_raw = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_test_raw = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_dates = test_set.index[config.time_steps:]

    all_rows: list[dict[str, object]] = []
    config.output_root.mkdir(parents=True, exist_ok=True)

    for seed in config.seeds:
        for model_name in ("xgb", "lgbm"):
            best_params, metrics, y_pred_raw, trial_df = run_search(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_val_raw=y_val_raw,
                y_test_raw=y_test_raw,
                target_scaler=target_scaler,
                trials=config.trials,
                seed=seed,
            )

            run_dir = config.output_root / config.bank_name / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            shap_top_feature = None
            if config.run_shap:
                eval_n = min(config.shap_eval_size, len(x_test))
                x_eval = x_test[:eval_n]
                if model_name == "xgb":
                    best_model = fit_xgb_model(x_train, y_train, x_val, y_val, best_params, seed)
                else:
                    best_model = fit_lgbm_model(x_train, y_train, x_val, y_val, best_params, seed)
                shap_df = compute_tree_shap_importance(model_name, best_model, x_eval, flat_feature_names)
                shap_df.to_csv(run_dir / "shap_tree_feature_importance.csv", index=False)
                shap_top_feature = str(shap_df.iloc[0]["feature"])

            pred_df = pd.DataFrame(
                {
                    "date": test_dates,
                    "actual": y_test_raw,
                    "predicted": y_pred_raw,
                    "abs_error": np.abs(y_test_raw - y_pred_raw),
                }
            )
            pred_df.to_csv(run_dir / "test_predictions_sota.csv", index=False)
            trial_df.to_csv(run_dir / "trials_sota.csv", index=False)

            config_dict = asdict(config)
            config_dict["data_dir"] = str(config_dict["data_dir"])
            config_dict["output_root"] = str(config_dict["output_root"])
            summary = {
                "config": config_dict,
                "model": model_name,
                "seed": seed,
                "metrics": metrics,
                "best_params": best_params,
                "feature_columns": feature_cols,
                "target_column": TARGET_COL,
                "n_train_sequences": int(len(x_train)),
                "n_val_sequences": int(len(x_val)),
                "n_test_sequences": int(len(x_test)),
            }
            if shap_top_feature is not None:
                summary["shap"] = {
                    "method": "tree_explainer",
                    "eval_size": int(min(config.shap_eval_size, len(x_test))),
                    "top_feature": shap_top_feature,
                }
            with open(run_dir / "summary_sota.json", "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)

            all_rows.append(
                {
                    "bank": config.bank_name,
                    "mode": "macro" if config.use_macro else "data",
                    "model": model_name,
                    "seed": seed,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "mse": metrics["mse"],
                    "summary_path": str(run_dir / "summary_sota.json"),
                    "predictions_path": str(run_dir / "test_predictions_sota.csv"),
                }
            )
            print(
                f"Finished {model_name.upper()} | seed={seed} | "
                f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} MAPE={metrics['mape']:.2f}%"
            )

    pd.DataFrame(all_rows).to_csv(config.output_root / "study_runs_sota.csv", index=False)
    print(f"Saved run table: {config.output_root / 'study_runs_sota.csv'}")
