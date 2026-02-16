from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study for bank price prediction.")
    parser.add_argument("--bank", required=True, type=str, help="Bank symbol, e.g. BBCA")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-background-size", type=int, default=4)
    parser.add_argument("--shap-eval-size", type=int, default=2)
    parser.add_argument("--shap-nsamples", type=int, default=10)
    parser.add_argument("--base-output", type=Path, default=Path("results_ablation"))
    parser.add_argument(
        "--mode",
        choices=["full", "model_macro", "model_macro_ga"],
        default="full",
        help=(
            "full=4 skenario lama (LSTM/GA/Macro), "
            "model_macro=4 skenario LSTM vs BiLSTM x data vs macro, "
            "model_macro_ga=8 skenario (x2 dengan FastGA)"
        ),
    )
    return parser.parse_args()


def run_case(case_name: str, cmd: list[str], summary_path: Path) -> dict[str, object]:
    print(f"\n=== Running {case_name} ===")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    metrics = summary["metrics"]
    return {
        "case": case_name,
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mape": metrics["mape"],
        "n_features": len(summary["feature_columns"]),
        "model_type": summary["config"].get("model_type", "lstm"),
        "use_macro": summary["config"]["use_macro"],
        "optimize": summary["config"]["optimize"],
    }


def main() -> None:
    args = parse_args()
    bank = args.bank.strip().upper()
    root = args.base_output / bank
    root.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    base = [
        python_exe,
        "train_price_pytorch.py",
        "--bank",
        bank,
        "--epochs",
        str(args.epochs),
        "--forecast-days",
        str(args.forecast_days),
        "--time-steps",
        str(args.time_steps),
        "--tune-epochs",
        str(args.tune_epochs),
        "--tune-candidates",
        str(args.tune_candidates),
    ]
    if args.run_shap:
        base += [
            "--run-shap",
            "--shap-background-size",
            str(args.shap_background_size),
            "--shap-eval-size",
            str(args.shap_eval_size),
            "--shap-nsamples",
            str(args.shap_nsamples),
        ]

    if args.mode == "model_macro":
        cases = [
            ("lstm_data", ["--data-dir", "data", "--model-type", "lstm"]),
            ("lstm_macro", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "lstm"]),
            ("bilstm_data", ["--data-dir", "data", "--model-type", "bilstm"]),
            ("bilstm_macro", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "bilstm"]),
        ]
    elif args.mode == "model_macro_ga":
        cases = [
            ("lstm_data", ["--data-dir", "data", "--model-type", "lstm"]),
            ("lstm_macro", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "lstm"]),
            ("bilstm_data", ["--data-dir", "data", "--model-type", "bilstm"]),
            ("bilstm_macro", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "bilstm"]),
            ("lstm_data_ga", ["--data-dir", "data", "--model-type", "lstm", "--optimize"]),
            ("lstm_macro_ga", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "lstm", "--optimize"]),
            ("bilstm_data_ga", ["--data-dir", "data", "--model-type", "bilstm", "--optimize"]),
            ("bilstm_macro_ga", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "bilstm", "--optimize"]),
        ]
    else:
        cases = [
            ("raw_lstm", ["--data-dir", "data", "--model-type", "lstm"]),
            ("lstm_macro", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "lstm"]),
            ("lstm_ga_macro", ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "lstm", "--optimize"]),
            ("lstm_ga", ["--data-dir", "data", "--model-type", "lstm", "--optimize"]),
        ]

    rows = []
    progress = tqdm(cases, desc=f"Ablation {bank}", unit="case")
    for case_name, extra in progress:
        progress.set_postfix_str(case_name)
        case_output = root / case_name
        cmd = base + ["--output-root", str(case_output)] + extra
        summary_path = case_output / bank / "summary_pytorch.json"
        row = run_case(case_name, cmd, summary_path)
        rows.append(row)
    progress.close()

    result = pd.DataFrame(rows).sort_values("rmse")
    csv_path = root / "ablation_summary.csv"
    result.to_csv(csv_path, index=False)
    print("\n=== Ablation Summary ===")
    print(result.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
