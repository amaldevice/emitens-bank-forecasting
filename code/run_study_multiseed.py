from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


DEFAULT_BANKS = ["BBCA", "BBNI", "BBRI", "BBTN", "BDMN", "BMRI", "BNGA"]
BASE_SCRIPT = "train_price_pytorch.py"
MODEL_SCRIPTS = {
    "lstm": BASE_SCRIPT,
    "bilstm": BASE_SCRIPT,
    "lstm_pre_norm": "train_price_pytorch_norm.py",
    "lstm_post_norm": "train_price_pytorch_norm.py",
    "lstm_pre_post_norm": "train_price_pytorch_norm.py",
    "bilstm_pre_norm": "train_price_pytorch_norm.py",
    "bilstm_post_norm": "train_price_pytorch_norm.py",
    "bilstm_pre_post_norm": "train_price_pytorch_norm.py",
    "cnn_lstm": "train_price_pytorch_cnn.py",
    "cnn_bilstm": "train_price_pytorch_cnn.py",
    "lstm_attention": "train_price_pytorch_attention.py",
    "bilstm_attention": "train_price_pytorch_attention.py",
}


def build_case_specs() -> dict[str, tuple[str, list[str]]]:
    specs: dict[str, tuple[str, list[str]]] = {}
    for model_type, script in MODEL_SCRIPTS.items():
        specs[f"{model_type}_data"] = (script, ["--data-dir", "data", "--model-type", model_type])
        specs[f"{model_type}_macro"] = (
            script,
            ["--data-dir", "data_with_macro", "--use-macro", "--model-type", model_type],
        )
        specs[f"{model_type}_data_ga"] = (
            script,
            ["--data-dir", "data", "--model-type", model_type, "--optimize"],
        )
        specs[f"{model_type}_macro_ga"] = (
            script,
            ["--data-dir", "data_with_macro", "--use-macro", "--model-type", model_type, "--optimize"],
        )
        specs[f"{model_type}_data_random"] = specs[f"{model_type}_data_ga"]
        specs[f"{model_type}_macro_random"] = specs[f"{model_type}_macro_ga"]
    return specs


CASE_SPECS = build_case_specs()
CASE_ARGS = {name: args for name, (_, args) in CASE_SPECS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed study across banks/configs.")
    parser.add_argument("--banks", type=str, default=",".join(DEFAULT_BANKS))
    parser.add_argument("--cases", type=str, default=",".join(CASE_ARGS.keys()))
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-only", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--shap-background-size", type=int, default=4)
    parser.add_argument("--shap-eval-size", type=int, default=2)
    parser.add_argument("--shap-nsamples", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("results_study"))
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    banks = [b.strip().upper() for b in args.banks.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASE_ARGS]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")

    py = sys.executable
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    total_runs = len(banks) * len(cases) * len(seeds)
    progress = tqdm(total=total_runs, desc="Study runs", unit="run")
    for bank in banks:
        for case in cases:
            for seed in seeds:
                progress.set_postfix_str(f"{bank} | {case} | seed={seed}")
                case_dir = root / bank / case / f"seed_{seed}"
                cmd = [
                    py,
                    CASE_SPECS[case][0],
                    "--bank",
                    bank,
                    "--seed",
                    str(seed),
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
                    "--output-root",
                    str(case_dir),
                ] + CASE_SPECS[case][1]
                if args.shap_only:
                    cmd += ["--shap-only"]
                if args.save_model:
                    cmd += ["--save-model"]
                if args.model_path:
                    cmd += ["--model-path", args.model_path]
                if args.run_shap:
                    cmd += [
                        "--run-shap",
                        "--shap-background-size",
                        str(args.shap_background_size),
                        "--shap-eval-size",
                        str(args.shap_eval_size),
                        "--shap-nsamples",
                        str(args.shap_nsamples),
                    ]

                print(f"\n=== bank={bank} case={case} seed={seed} ===")
                run_cmd(cmd)

                summary_path = case_dir / bank / "summary_pytorch.json"
                test_pred_path = case_dir / bank / "test_predictions_pytorch.csv"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                m = summary["metrics"]
                rows.append(
                    {
                        "bank": bank,
                        "case": case,
                        "seed": seed,
                        "model_type": summary["config"].get("model_type", "lstm"),
                        "use_macro": bool(summary["config"]["use_macro"]),
                        "optimize": bool(summary["config"]["optimize"]),
                        "rmse": float(m["rmse"]),
                        "mae": float(m["mae"]),
                        "mape": float(m["mape"]),
                        "mse": float(m["mse"]),
                        "summary_path": str(summary_path),
                        "test_predictions_path": str(test_pred_path),
                    }
                )
                progress.update(1)
    progress.close()

    df = pd.DataFrame(rows)
    out_csv = root / "study_runs.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved run table: {out_csv}")


if __name__ == "__main__":
    main()
