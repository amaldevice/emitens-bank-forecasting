from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_price_pytorch import (
    FEATURE_COLS,
    MACRO_FEATURE_COLS,
    PriceLSTM,
    TrainConfig,
    build_dataloaders,
    load_data,
    set_seed,
    split_dataset_by_date,
)

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP beeswarm plot from a saved checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--nsamples", type=int, default=100)
    parser.add_argument("--max-display", type=int, default=10)
    return parser.parse_args()


def infer_context(model_path: Path) -> dict[str, object]:
    bank_dir = model_path.parent
    summary_path = bank_dir / "summary_pytorch.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file for context inference: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = summary.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    use_macro = bool(cfg.get("use_macro", False))
    context = {
        "bank_name": str(cfg.get("bank_name", bank_dir.name)),
        "data_dir": Path(str(cfg.get("data_dir", "data_with_macro" if use_macro else "data"))),
        "time_steps": int(cfg.get("time_steps", 60)),
        "hidden_size": int(cfg.get("hidden_size", 32)),
        "dropout": float(cfg.get("dropout", 0.3)),
        "model_type": str(cfg.get("model_type", "lstm")),
        "seed": int(cfg.get("seed", 42)),
        "use_macro": use_macro,
    }
    return context


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    ctx = infer_context(args.model_path)
    feature_cols = FEATURE_COLS + MACRO_FEATURE_COLS if ctx["use_macro"] else FEATURE_COLS

    cfg = TrainConfig(
        bank_name=str(ctx["bank_name"]),
        data_dir=Path(ctx["data_dir"]),
        output_root=Path("results"),
        time_steps=int(ctx["time_steps"]),
        hidden_size=int(ctx["hidden_size"]),
        dropout=float(ctx["dropout"]),
        model_type=str(ctx["model_type"]),
        use_macro=bool(ctx["use_macro"]),
        seed=int(ctx["seed"]),
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_data, _, _ = load_data(cfg, feature_cols)
    train_set = split_dataset_by_date(selected_data, "2014-01-02", "2020-01-02")
    val_set = split_dataset_by_date(selected_data, "2020-01-02", "2022-01-02")
    test_set = split_dataset_by_date(selected_data, "2022-01-02", "2024-12-31")
    _, _, x_train, _, x_test, _ = build_dataloaders(
        train_set, val_set, test_set, feature_cols, cfg.time_steps, batch_size=32
    )

    model = PriceLSTM(
        len(feature_cols),
        cfg.hidden_size,
        cfg.dropout,
        bidirectional=(cfg.model_type == "bilstm"),
    ).to(device)

    payload = torch.load(args.model_path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()

    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is not installed. Install it with: uv pip install shap") from exc

    bg_n = min(max(1, args.background_size), len(x_train))
    eval_n = min(max(1, args.eval_size), len(x_test))
    background = x_train[:bg_n]
    explain_samples = x_test[:eval_n]

    time_steps = cfg.time_steps
    n_features = len(feature_cols)
    background_flat = background.reshape(bg_n, -1)
    explain_flat = explain_samples.reshape(eval_n, -1)

    def model_predict(flat_input: np.ndarray) -> np.ndarray:
        seq_input = flat_input.reshape(-1, time_steps, n_features).astype(np.float32)
        with torch.no_grad():
            preds = model(torch.from_numpy(seq_input).to(device)).cpu().numpy()
        return preds

    explainer = shap.KernelExplainer(model_predict, background_flat)
    shap_values = explainer.shap_values(explain_flat, nsamples=args.nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    shap_3d = shap_values.reshape(eval_n, time_steps, n_features)
    # Aggregate temporal contribution so beeswarm shows feature-level impact.
    shap_feature = shap_3d.mean(axis=1)
    feature_matrix = explain_samples.mean(axis=1)

    plt.figure(figsize=(10, 6), dpi=220)
    shap.summary_plot(
        shap_feature,
        feature_matrix,
        feature_names=feature_cols,
        max_display=args.max_display,
        plot_type="dot",
        show=False,
    )
    plt.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_png, bbox_inches="tight")
    plt.close()

    print(f"Saved beeswarm plot: {args.output_png}")


if __name__ == "__main__":
    main()
