from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


DEFAULT_BANKS = ["BBCA", "BBNI", "BBRI", "BBTN", "BDMN", "BMRI", "BNGA"]
CASE_ARGS = {
    "informer_data": ["--data-dir", "data", "--model-type", "informer"],
    "informer_macro": ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "informer"],
    "autoformer_data": ["--data-dir", "data", "--model-type", "autoformer"],
    "autoformer_macro": ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "autoformer"],
    "informer_data_ga": ["--data-dir", "data", "--model-type", "informer", "--optimize"],
    "informer_macro_ga": ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "informer", "--optimize"],
    "autoformer_data_ga": ["--data-dir", "data", "--model-type", "autoformer", "--optimize"],
    "autoformer_macro_ga": ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "autoformer", "--optimize"],
    "informer_data_random": ["--data-dir", "data", "--model-type", "informer", "--optimize", "--hpo-method", "random"],
    "informer_macro_random": [
        "--data-dir",
        "data_with_macro",
        "--use-macro",
        "--model-type",
        "informer",
        "--optimize",
        "--hpo-method",
        "random",
    ],
    "autoformer_data_random": ["--data-dir", "data", "--model-type", "autoformer", "--optimize", "--hpo-method", "random"],
    "autoformer_macro_random": [
        "--data-dir",
        "data_with_macro",
        "--use-macro",
        "--model-type",
        "autoformer",
        "--optimize",
        "--hpo-method",
        "random",
    ],
    "informer_lite_data": ["--data-dir", "data", "--model-type", "informer_lite"],
    "informer_lite_macro": ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "informer_lite"],
    "autoformer_lite_data": ["--data-dir", "data", "--model-type", "autoformer_lite"],
    "autoformer_lite_macro": ["--data-dir", "data_with_macro", "--use-macro", "--model-type", "autoformer_lite"],
    "informer_lite_data_ga": ["--data-dir", "data", "--model-type", "informer_lite", "--optimize"],
    "informer_lite_macro_ga": [
        "--data-dir",
        "data_with_macro",
        "--use-macro",
        "--model-type",
        "informer_lite",
        "--optimize",
    ],
    "autoformer_lite_data_ga": ["--data-dir", "data", "--model-type", "autoformer_lite", "--optimize"],
    "autoformer_lite_macro_ga": [
        "--data-dir",
        "data_with_macro",
        "--use-macro",
        "--model-type",
        "autoformer_lite",
        "--optimize",
    ],
    "informer_lite_data_random": [
        "--data-dir",
        "data",
        "--model-type",
        "informer_lite",
        "--optimize",
        "--hpo-method",
        "random",
    ],
    "informer_lite_macro_random": [
        "--data-dir",
        "data_with_macro",
        "--use-macro",
        "--model-type",
        "informer_lite",
        "--optimize",
        "--hpo-method",
        "random",
    ],
    "autoformer_lite_data_random": [
        "--data-dir",
        "data",
        "--model-type",
        "autoformer_lite",
        "--optimize",
        "--hpo-method",
        "random",
    ],
    "autoformer_lite_macro_random": [
        "--data-dir",
        "data_with_macro",
        "--use-macro",
        "--model-type",
        "autoformer_lite",
        "--optimize",
        "--hpo-method",
        "random",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Informer/Autoformer study across banks/configs.")
    parser.add_argument("--banks", type=str, default=",".join(DEFAULT_BANKS))
    parser.add_argument("--cases", type=str, default=",".join(CASE_ARGS.keys()))
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--moving-avg", type=int, default=5)
    parser.add_argument("--no-distill", action="store_true")
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--hpo-method", choices=["ga", "random"], default="ga")
    parser.add_argument("--hpo-generations", type=int, default=5)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--shap-background-size", type=int, default=4)
    parser.add_argument("--shap-eval-size", type=int, default=2)
    parser.add_argument("--shap-nsamples", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("results_transformer_study"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("Command:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    banks = [b.upper() for b in split_csv(args.banks)]
    seeds = [int(s) for s in split_csv(args.seeds)]
    cases = split_csv(args.cases)
    unknown = [case for case in cases if case not in CASE_ARGS]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    total_runs = len(banks) * len(cases) * len(seeds)
    progress = tqdm(total=total_runs, desc="Transformer study", unit="run")
    for bank in banks:
        for case in cases:
            for seed in seeds:
                progress.set_postfix_str(f"{bank} | {case} | seed={seed}")
                case_dir = root / bank / case / f"seed_{seed}"
                cmd = [
                    sys.executable,
                    "train_price_transformers.py",
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
                    "--batch-size",
                    str(args.batch_size),
                    "--hidden-size",
                    str(args.hidden_size),
                    "--dropout",
                    str(args.dropout),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--n-heads",
                    str(args.n_heads),
                    "--n-layers",
                    str(args.n_layers),
                    "--ff-multiplier",
                    str(args.ff_multiplier),
                    "--moving-avg",
                    str(args.moving_avg),
                    "--tune-epochs",
                    str(args.tune_epochs),
                    "--tune-candidates",
                    str(args.tune_candidates),
                    "--hpo-method",
                    args.hpo_method,
                    "--hpo-generations",
                    str(args.hpo_generations),
                    "--output-root",
                    str(case_dir),
                ] + CASE_ARGS[case]
                if args.no_distill:
                    cmd += ["--no-distill"]
                if args.save_model:
                    cmd += ["--save-model"]
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
                run_cmd(cmd, args.dry_run)

                if not args.dry_run:
                    summary_path = case_dir / bank / "summary_transformer.json"
                    test_pred_path = case_dir / bank / "test_predictions_transformer.csv"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    metrics = summary["metrics"]
                    rows.append(
                        {
                            "bank": bank,
                            "case": case,
                            "seed": seed,
                            "model_type": summary["config"].get("model_type"),
                            "use_macro": bool(summary["config"]["use_macro"]),
                            "optimize": bool(summary["config"]["optimize"]),
                            "rmse": float(metrics["rmse"]),
                            "mae": float(metrics["mae"]),
                            "mape": float(metrics["mape"]),
                            "mse": float(metrics["mse"]),
                            "summary_path": str(summary_path),
                            "test_predictions_path": str(test_pred_path),
                        }
                    )
                progress.update(1)
    progress.close()

    if rows:
        import pandas as pd

        out_csv = root / "transformer_study_runs.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved run table: {out_csv}")


if __name__ == "__main__":
    main()
