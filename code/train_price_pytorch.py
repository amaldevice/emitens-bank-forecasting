import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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
class TrainConfig:
    bank_name: str
    forecast_days: int = 30
    data_dir: Path = Path("data")
    output_root: Path = Path("results")
    time_steps: int = 60
    epochs: int = 50
    batch_size: int = 32
    hidden_size: int = 32
    dropout: float = 0.3
    learning_rate: float = 1e-3
    optimize: bool = False
    use_macro: bool = False
    tune_epochs: int = 8
    tune_candidates: int = 6
    run_shap: bool = False
    shap_background_size: int = 64
    shap_eval_size: int = 32
    shap_nsamples: int = 100
    seed: int = 42


class PriceLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_data(
    config: TrainConfig, feature_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler, StandardScaler]:
    csv_path = config.data_dir / f"{config.bank_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in feature_cols + [TARGET_COL, "Date"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.ffill().dropna()

    selected = df[feature_cols + [TARGET_COL]].copy()
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    selected[feature_cols] = feature_scaler.fit_transform(selected[feature_cols].values)
    selected[TARGET_COL] = target_scaler.fit_transform(selected[[TARGET_COL]]).flatten()
    return selected, feature_scaler, target_scaler


def build_dataloaders(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
    feature_cols: list[str],
    time_steps: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = create_sequences(train_set, feature_cols, TARGET_COL, time_steps)
    x_val, y_val = create_sequences(val_set, feature_cols, TARGET_COL, time_steps)
    x_test, y_test = create_sequences(test_set, feature_cols, TARGET_COL, time_steps)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, x_train, y_train, x_test, y_test


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * x_batch.size(0)

    return running_loss / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_counter = 0
    patience = 20

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.6g}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x).to(device)
        pred = model(tensor).cpu().numpy()
    return pred


def evaluate_test(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: StandardScaler,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    y_pred_scaled = predict(model, x_test, device)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    metrics = {"mse": float(mse), "rmse": rmse, "mae": float(mae), "mape": mape}
    return metrics, y_true, y_pred


def forecast_future(
    model: nn.Module,
    last_sequence: np.ndarray,
    forecast_days: int,
    target_scaler: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    seq = last_sequence.copy()
    preds_scaled = []

    for _ in range(forecast_days):
        y_next = predict(model, seq[np.newaxis, :, :].astype(np.float32), device)[0]
        preds_scaled.append(y_next)

        next_step = seq[-1].copy()
        seq = np.vstack([seq[1:], next_step])

    preds = target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


def run_shap_kernel_explainer(
    model: nn.Module,
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_cols: list[str],
    time_steps: int,
    device: torch.device,
    background_size: int,
    eval_size: int,
    nsamples: int,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with: uv pip install shap"
        ) from exc

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("SHAP requires non-empty train and test sequences.")

    n_features = len(feature_cols)
    rng = np.random.default_rng(42)

    bg_n = min(background_size, len(x_train))
    eval_n = min(eval_size, len(x_test))
    bg_idx = rng.choice(len(x_train), size=bg_n, replace=False)
    eval_idx = rng.choice(len(x_test), size=eval_n, replace=False)

    background = x_train[bg_idx]
    explain_samples = x_test[eval_idx]
    background_flat = background.reshape(bg_n, -1)
    explain_flat = explain_samples.reshape(eval_n, -1)

    def model_predict(flat_input: np.ndarray) -> np.ndarray:
        seq_input = flat_input.reshape(-1, time_steps, n_features).astype(np.float32)
        return predict(model, seq_input, device)

    print(
        f"Running SHAP KernelExplainer | background={bg_n}, eval={eval_n}, nsamples={nsamples}"
    )
    explainer = shap.KernelExplainer(model_predict, background_flat)
    shap_values = explainer.shap_values(explain_flat, nsamples=nsamples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    shap_3d = shap_values.reshape(eval_n, time_steps, n_features)
    feature_importance = np.mean(np.abs(shap_3d), axis=(0, 1))
    out = pd.DataFrame({"feature": feature_cols, "importance": feature_importance})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def tune_hyperparams(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    feature_cols: list[str],
    base_config: TrainConfig,
    device: torch.device,
) -> TrainConfig:
    print("Running lightweight random search...")
    search_space = {
        "hidden_size": [16, 32, 64],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [16, 32, 64],
        "dropout": [0.2, 0.3, 0.4],
    }
    candidates = max(1, int(base_config.tune_candidates))
    best_val = float("inf")
    best_cfg = copy.deepcopy(base_config)

    for i in range(1, candidates + 1):
        trial = copy.deepcopy(base_config)
        trial.hidden_size = random.choice(search_space["hidden_size"])
        trial.learning_rate = random.choice(search_space["learning_rate"])
        trial.batch_size = random.choice(search_space["batch_size"])
        trial.dropout = random.choice(search_space["dropout"])
        trial.epochs = max(1, int(base_config.tune_epochs))

        train_loader, val_loader, *_ = build_dataloaders(
            train_set, val_set, val_set, feature_cols, trial.time_steps, trial.batch_size
        )
        model = PriceLSTM(len(feature_cols), trial.hidden_size, trial.dropout).to(device)
        model, history = train_model(model, train_loader, val_loader, trial, device)
        trial_best_val = min(h["val_loss"] for h in history)
        print(
            f"Trial {i}/{candidates} | val={trial_best_val:.6f} | "
            f"hidden={trial.hidden_size}, lr={trial.learning_rate}, bs={trial.batch_size}, drop={trial.dropout}"
        )
        if trial_best_val < best_val:
            best_val = trial_best_val
            best_cfg = copy.deepcopy(trial)

    best_cfg.epochs = base_config.epochs
    print(f"Best random-search val_loss: {best_val:.6f}")
    return best_cfg


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Refactored PyTorch pipeline for bank closing-price prediction.")
    parser.add_argument("--bank", type=str, help="Bank symbol, e.g. BBCA")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--use-macro", action="store_true")
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-background-size", type=int, default=64)
    parser.add_argument("--shap-eval-size", type=int, default=32)
    parser.add_argument("--shap-nsamples", type=int, default=100)
    args = parser.parse_args()

    bank_name = args.bank.strip().upper() if args.bank else input("Enter bank name (e.g., BBCA): ").strip().upper()
    return TrainConfig(
        bank_name=bank_name,
        data_dir=Path(args.data_dir),
        output_root=Path(args.output_root),
        forecast_days=args.forecast_days,
        epochs=args.epochs,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        optimize=args.optimize,
        use_macro=args.use_macro,
        tune_epochs=args.tune_epochs,
        tune_candidates=args.tune_candidates,
        run_shap=args.run_shap,
        shap_background_size=args.shap_background_size,
        shap_eval_size=args.shap_eval_size,
        shap_nsamples=args.shap_nsamples,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = FEATURE_COLS + MACRO_FEATURE_COLS if config.use_macro else FEATURE_COLS
    print(f"Device: {device}")
    print(f"Bank: {config.bank_name} | Forecast days: {config.forecast_days}")
    print(f"Data dir: {config.data_dir} | Use macro: {config.use_macro}")
    print(f"Feature count: {len(feature_cols)}")

    output_dir = config.output_root / config.bank_name
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_data, feature_scaler, target_scaler = load_data(config, feature_cols)
    train_set = split_dataset_by_date(selected_data, "2014-01-02", "2020-01-02")
    val_set = split_dataset_by_date(selected_data, "2020-01-02", "2022-01-02")
    test_set = split_dataset_by_date(selected_data, "2022-01-02", "2024-12-31")
    print(f"Split sizes | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    if config.optimize:
        config = tune_hyperparams(train_set, val_set, feature_cols, config, device)
        print(
            f"Using tuned params | hidden={config.hidden_size} lr={config.learning_rate} "
            f"bs={config.batch_size} dropout={config.dropout}"
        )

    train_loader, val_loader, x_train, y_train, x_test, y_test = build_dataloaders(
        train_set, val_set, test_set, feature_cols, config.time_steps, config.batch_size
    )
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Insufficient sequence samples. Reduce --time-steps or verify date ranges in data.")

    model = PriceLSTM(len(feature_cols), config.hidden_size, config.dropout).to(device)
    model, history = train_model(model, train_loader, val_loader, config, device)
    metrics, y_true, y_pred = evaluate_test(model, x_test, y_test, target_scaler, device)
    print(
        f"Test metrics | MSE={metrics['mse']:.4f} RMSE={metrics['rmse']:.4f} "
        f"MAE={metrics['mae']:.4f} MAPE={metrics['mape']:.2f}%"
    )

    forecasts = forecast_future(model, x_test[-1], config.forecast_days, target_scaler, device)
    test_dates = test_set.index[config.time_steps:]
    future_dates = pd.date_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=config.forecast_days, freq="D")

    pd.DataFrame(
        {"date": test_dates, "actual": y_true, "predicted": y_pred, "abs_error": np.abs(y_true - y_pred)}
    ).to_csv(output_dir / "test_predictions_pytorch.csv", index=False)
    pd.DataFrame({"date": future_dates, "forecast_price": forecasts}).to_csv(
        output_dir / "forecast_pytorch.csv", index=False
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history_pytorch.csv", index=False)

    shap_importance = None
    if config.run_shap:
        shap_importance = run_shap_kernel_explainer(
            model=model,
            x_train=x_train,
            x_test=x_test,
            feature_cols=feature_cols,
            time_steps=config.time_steps,
            device=device,
            background_size=config.shap_background_size,
            eval_size=config.shap_eval_size,
            nsamples=config.shap_nsamples,
        )
        shap_importance.to_csv(output_dir / "shap_kernel_feature_importance.csv", index=False)
        print("Top SHAP features:")
        for _, row in shap_importance.head(5).iterrows():
            print(f" - {row['feature']}: {row['importance']:.6f}")

    config_dict = asdict(config)
    config_dict["data_dir"] = str(config_dict["data_dir"])
    config_dict["output_root"] = str(config_dict["output_root"])
    summary = {
        "config": config_dict,
        "metrics": metrics,
        "n_train_sequences": int(len(x_train)),
        "n_test_sequences": int(len(x_test)),
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
    }
    if shap_importance is not None:
        summary["shap"] = {
            "method": "kernel_explainer",
            "background_size": int(min(config.shap_background_size, len(x_train))),
            "eval_size": int(min(config.shap_eval_size, len(x_test))),
            "nsamples": int(config.shap_nsamples),
            "top_feature": str(shap_importance.iloc[0]["feature"]),
        }
    with open(output_dir / "summary_pytorch.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Saved outputs to: {output_dir}")
    print(" - test_predictions_pytorch.csv")
    print(" - forecast_pytorch.csv")
    print(" - training_history_pytorch.csv")
    print(" - summary_pytorch.json")
    if shap_importance is not None:
        print(" - shap_kernel_feature_importance.csv")


if __name__ == "__main__":
    main()
