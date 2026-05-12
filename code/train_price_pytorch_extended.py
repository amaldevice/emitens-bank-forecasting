from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_price_pytorch import (
    FEATURE_COLS,
    MACRO_FEATURE_COLS,
    TARGET_COL,
    PriceLSTM,
    TrainConfig,
    build_dataloaders,
    evaluate_test,
    forecast_future,
    load_data,
    run_shap_kernel_explainer,
    set_seed,
    split_dataset_by_date,
    train_model,
)


NORM_MODEL_TYPES = [
    "lstm_pre_norm",
    "lstm_post_norm",
    "lstm_pre_post_norm",
    "bilstm_pre_norm",
    "bilstm_post_norm",
    "bilstm_pre_post_norm",
]
CNN_MODEL_TYPES = ["cnn_lstm", "cnn_bilstm"]
ATTENTION_MODEL_TYPES = ["lstm_attention", "bilstm_attention"]
BASE_MODEL_TYPES = ["lstm", "bilstm"]
ALL_MODEL_TYPES = BASE_MODEL_TYPES + NORM_MODEL_TYPES + CNN_MODEL_TYPES + ATTENTION_MODEL_TYPES


class NormPriceLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        dropout: float,
        bidirectional: bool,
        pre_norm: bool,
        post_norm: bool,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(n_features) if pre_norm else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.post_norm = nn.LayerNorm(lstm_out) if post_norm else nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        out, _ = self.lstm(x)
        last = self.post_norm(out[:, -1, :])
        return self.head(last).squeeze(-1)


class CNNRecurrent(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        dropout: float,
        bidirectional: bool,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class AttentionRecurrent(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Linear(lstm_out, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.head(context).squeeze(-1)


def build_model(model_type: str, n_features: int, hidden_size: int, dropout: float) -> nn.Module:
    if model_type == "lstm":
        return PriceLSTM(n_features, hidden_size, dropout, bidirectional=False)
    if model_type == "bilstm":
        return PriceLSTM(n_features, hidden_size, dropout, bidirectional=True)
    if model_type in NORM_MODEL_TYPES:
        bidirectional = model_type.startswith("bilstm")
        pre_norm = "_pre_norm" in model_type or "_pre_post_norm" in model_type
        post_norm = "_post_norm" in model_type or "_pre_post_norm" in model_type
        return NormPriceLSTM(n_features, hidden_size, dropout, bidirectional, pre_norm, post_norm)
    if model_type in CNN_MODEL_TYPES:
        return CNNRecurrent(
            n_features,
            hidden_size,
            dropout,
            bidirectional=(model_type == "cnn_bilstm"),
        )
    if model_type in ATTENTION_MODEL_TYPES:
        return AttentionRecurrent(
            n_features,
            hidden_size,
            dropout,
            bidirectional=(model_type == "bilstm_attention"),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


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
        model = build_model(trial.model_type, len(feature_cols), trial.hidden_size, trial.dropout).to(device)
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


def parse_args(
    allowed_model_types: list[str],
    default_model_type: str,
    description: str,
) -> TrainConfig:
    parser = argparse.ArgumentParser(description=description)
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
    parser.add_argument("--model-type", choices=allowed_model_types, default=default_model_type)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-only", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--shap-background-size", type=int, default=64)
    parser.add_argument("--shap-eval-size", type=int, default=32)
    parser.add_argument("--shap-nsamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
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
        model_type=args.model_type,
        optimize=args.optimize,
        use_macro=args.use_macro,
        tune_epochs=args.tune_epochs,
        tune_candidates=args.tune_candidates,
        run_shap=args.run_shap,
        shap_only=args.shap_only,
        save_model=args.save_model,
        model_path=Path(args.model_path) if args.model_path else None,
        shap_background_size=args.shap_background_size,
        shap_eval_size=args.shap_eval_size,
        shap_nsamples=args.shap_nsamples,
        seed=args.seed,
    )


def main(
    allowed_model_types: list[str] | None = None,
    default_model_type: str = "lstm_pre_norm",
    description: str = "Train extended PyTorch recurrent stock-price baselines.",
) -> None:
    config = parse_args(allowed_model_types or ALL_MODEL_TYPES, default_model_type, description)
    if config.shap_only and not config.run_shap:
        raise ValueError("`--shap-only` requires `--run-shap`.")
    if config.shap_only and config.optimize:
        raise ValueError("`--shap-only` cannot be combined with `--optimize`.")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = FEATURE_COLS + MACRO_FEATURE_COLS if config.use_macro else FEATURE_COLS
    print(f"Device: {device}")
    print(f"Bank: {config.bank_name} | Forecast days: {config.forecast_days}")
    print(f"Data dir: {config.data_dir} | Use macro: {config.use_macro}")
    print(f"Model type: {config.model_type}")
    print(f"Feature count: {len(feature_cols)}")

    output_dir = config.output_root / config.bank_name
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_data, _, target_scaler = load_data(config, feature_cols)
    train_set = split_dataset_by_date(selected_data, "2014-01-02", "2020-01-02")
    val_set = split_dataset_by_date(selected_data, "2020-01-02", "2022-01-02")
    test_set = split_dataset_by_date(selected_data, "2022-01-02", "2024-12-31")
    print(f"Split sizes | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    if config.optimize and not config.shap_only:
        config = tune_hyperparams(train_set, val_set, feature_cols, config, device)
        print(
            f"Using tuned params | hidden={config.hidden_size} lr={config.learning_rate} "
            f"bs={config.batch_size} dropout={config.dropout}"
        )

    train_loader, val_loader, x_train, _, x_test, y_test = build_dataloaders(
        train_set, val_set, test_set, feature_cols, config.time_steps, config.batch_size
    )
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Insufficient sequence samples. Reduce --time-steps or verify date ranges in data.")

    model = build_model(config.model_type, len(feature_cols), config.hidden_size, config.dropout).to(device)
    history: list[dict[str, float]] = []
    model_file = config.model_path if config.model_path is not None else output_dir / "model.pt"

    if config.shap_only:
        if not model_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
        payload = torch.load(model_file, map_location=device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded model checkpoint: {model_file}")
    else:
        model, history = train_model(model, train_loader, val_loader, config, device)
        if config.save_model:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": config.model_type,
                    "hidden_size": config.hidden_size,
                    "dropout": config.dropout,
                    "time_steps": config.time_steps,
                    "use_macro": config.use_macro,
                    "feature_columns": feature_cols,
                    "bank_name": config.bank_name,
                    "seed": config.seed,
                },
                model_file,
            )
            print(f"Saved model checkpoint: {model_file}")

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
    if history:
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

    config_dict = asdict(config)
    config_dict["data_dir"] = str(config_dict["data_dir"])
    config_dict["output_root"] = str(config_dict["output_root"])
    config_dict["model_path"] = str(config_dict["model_path"]) if config_dict["model_path"] is not None else None
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
    if history:
        print(" - training_history_pytorch.csv")
    print(" - summary_pytorch.json")
    if config.save_model and not config.shap_only:
        print(" - model.pt")
    if shap_importance is not None:
        print(" - shap_kernel_feature_importance.csv")


if __name__ == "__main__":
    main()
