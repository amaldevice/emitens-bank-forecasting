from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4-way ablation study for bank price prediction.")
    parser.add_argument("--bank", required=True, type=str, help="Bank symbol, e.g. BBCA")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--base-output", type=Path, default=Path("results_ablation"))
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

    cases = [
        ("raw_lstm", ["--data-dir", "data"]),
        ("lstm_macro", ["--data-dir", "data_with_macro", "--use-macro"]),
        ("lstm_ga_macro", ["--data-dir", "data_with_macro", "--use-macro", "--optimize"]),
        ("lstm_ga", ["--data-dir", "data", "--optimize"]),
    ]

    rows = []
    for case_name, extra in cases:
        case_output = root / case_name
        cmd = base + ["--output-root", str(case_output)] + extra
        summary_path = case_output / bank / "summary_pytorch.json"
        row = run_case(case_name, cmd, summary_path)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("rmse")
    csv_path = root / "ablation_summary.csv"
    result.to_csv(csv_path, index=False)
    print("\n=== Ablation Summary ===")
    print(result.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
