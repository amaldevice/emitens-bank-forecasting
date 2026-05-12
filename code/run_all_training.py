from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_BANKS = ["BBCA", "BBNI", "BBRI", "BBTN", "BDMN", "BMRI", "BNGA"]
DEFAULT_PYTORCH_CASES = [
    "lstm_data",
    "lstm_macro",
    "bilstm_data",
    "bilstm_macro",
    "lstm_data_ga",
    "lstm_macro_ga",
    "bilstm_data_ga",
    "bilstm_macro_ga",
]
DEFAULT_TRANSFORMER_CASES = [
    "informer_data",
    "informer_macro",
    "autoformer_data",
    "autoformer_macro",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full training suite for recurrent and SOTA stock-price models."
    )
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--banks", type=str, default=",".join(DEFAULT_BANKS))
    parser.add_argument("--run-pytorch", action="store_true", default=True)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--run-sota", action="store_true", default=True)
    parser.add_argument("--skip-sota", action="store_true")
    parser.add_argument("--run-sota-data", action="store_true", default=True)
    parser.add_argument("--skip-sota-data", action="store_true")
    parser.add_argument("--run-sota-macro", action="store_true", default=True)
    parser.add_argument("--skip-sota-macro", action="store_true")
    parser.add_argument("--run-transformer", action="store_true", default=True)
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--prepare-macro-data", action="store_true", default=True)
    parser.add_argument("--skip-macro-data", action="store_true")
    parser.add_argument("--annual-lag-years", type=int, default=0)
    parser.add_argument("--pytorch-cases", type=str, default=",".join(DEFAULT_PYTORCH_CASES))
    parser.add_argument("--transformer-cases", type=str, default=",".join(DEFAULT_TRANSFORMER_CASES))
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--transformer-hidden-size", type=int, default=64)
    parser.add_argument("--transformer-batch-size", type=int, default=32)
    parser.add_argument("--transformer-dropout", type=float, default=0.2)
    parser.add_argument("--transformer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-multiplier", type=int, default=4)
    parser.add_argument("--transformer-moving-avg", type=int, default=5)
    parser.add_argument("--transformer-no-distill", action="store_true")
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--transformer-hpo-method", choices=["ga", "random"], default="ga")
    parser.add_argument("--transformer-hpo-generations", type=int, default=5)
    parser.add_argument("--run-statistics", action="store_true")
    parser.add_argument("--pytorch-baseline-case", type=str, default="lstm_data")
    parser.add_argument("--transformer-baseline-case", type=str, default="informer_data")
    parser.add_argument("--sota-trials", type=int, default=50)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-only", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--shap-background-size", type=int, default=4)
    parser.add_argument("--shap-eval-size", type=int, default=2)
    parser.add_argument("--shap-nsamples", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("results_full_suite"))
    parser.add_argument("--transformer-output-root", type=Path, default=Path("results_transformer_study"))
    parser.add_argument("--sota-output-data", type=Path, default=Path("results_sota_data"))
    parser.add_argument("--sota-output-macro", type=Path, default=Path("results_sota_macro"))
    parser.add_argument("--modal-gpu", type=str, default="L4")
    parser.add_argument("--modal-run-name", type=str, default="modal_training")
    parser.add_argument(
        "--modal-skip-local-sota",
        action="store_true",
        help="With --backend modal, skip local XGBoost/LightGBM SOTA jobs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("Command:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def run_cmd_with_env(cmd: list[str], cwd: Path, dry_run: bool, env: dict[str, str]) -> None:
    print("Command:", " ".join(cmd))
    if dry_run:
        return
    merged_env = dict(os.environ)
    merged_env.update(env)
    subprocess.run(cmd, check=True, cwd=cwd, env=merged_env)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    py = sys.executable

    banks = [bank.upper() for bank in split_csv(args.banks)]
    pytorch_cases = split_csv(args.pytorch_cases)
    transformer_cases = split_csv(args.transformer_cases)
    seeds = split_csv(args.seeds)

    if args.skip_pytorch:
        run_pytorch = False
    else:
        run_pytorch = bool(args.run_pytorch)

    if args.skip_sota:
        run_sota = False
    else:
        run_sota = bool(args.run_sota)

    if args.skip_sota_data:
        run_sota_data = False
    else:
        run_sota_data = bool(args.run_sota_data)

    if args.skip_sota_macro:
        run_sota_macro = False
    else:
        run_sota_macro = bool(args.run_sota_macro)

    if args.skip_transformer:
        run_transformer = False
    else:
        run_transformer = bool(args.run_transformer)

    if args.skip_macro_data:
        prepare_macro_data = False
    else:
        prepare_macro_data = bool(args.prepare_macro_data)

    if prepare_macro_data:
        run_cmd(
            [
                py,
                "build_data_with_macro.py",
                "--source-dir",
                "data",
                "--output-dir",
                "data_with_macro",
                "--macro-dir",
                "data_macro",
                "--annual-lag-years",
                str(args.annual_lag_years),
            ],
            cwd=code_dir,
            dry_run=args.dry_run,
        )

    if args.backend == "modal":
        modal_cmd = [
            py,
            "-m",
            "modal",
            "run",
            "modal_train.py",
            "--banks",
            ",".join(banks),
            "--pytorch-cases",
            ",".join(pytorch_cases),
            "--transformer-cases",
            ",".join(transformer_cases),
            "--seeds",
            ",".join(seeds),
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
            "--transformer-hpo-method",
            args.transformer_hpo_method,
            "--transformer-hpo-generations",
            str(args.transformer_hpo_generations),
            "--transformer-batch-size",
            str(args.transformer_batch_size),
            "--transformer-hidden-size",
            str(args.transformer_hidden_size),
            "--transformer-dropout",
            str(args.transformer_dropout),
            "--transformer-learning-rate",
            str(args.transformer_learning_rate),
            "--transformer-heads",
            str(args.transformer_heads),
            "--transformer-layers",
            str(args.transformer_layers),
            "--transformer-ff-multiplier",
            str(args.transformer_ff_multiplier),
            "--transformer-moving-avg",
            str(args.transformer_moving_avg),
            "--shap-background-size",
            str(args.shap_background_size),
            "--shap-eval-size",
            str(args.shap_eval_size),
            "--shap-nsamples",
            str(args.shap_nsamples),
            "--run-name",
            args.modal_run_name,
        ]
        if not run_pytorch:
            modal_cmd += ["--skip-pytorch"]
        if not run_transformer:
            modal_cmd += ["--skip-transformer"]
        if args.transformer_no_distill:
            modal_cmd += ["--transformer-no-distill"]
        if args.run_shap:
            modal_cmd += ["--run-shap"]
        if args.save_model:
            modal_cmd += ["--save-model"]
        if args.dry_run:
            modal_cmd += ["--dry-run"]

        run_cmd_with_env(
            modal_cmd,
            cwd=code_dir,
            dry_run=args.dry_run,
            env={"MODAL_GPU": args.modal_gpu},
        )
        run_pytorch = False
        run_transformer = False
        if args.modal_skip_local_sota:
            run_sota = False

    if run_pytorch:
        pytorch_cmd = [
            py,
            "run_study_multiseed.py",
            "--banks",
            ",".join(banks),
            "--cases",
            ",".join(pytorch_cases),
            "--seeds",
            ",".join(seeds),
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
            str(args.output_root),
        ]
        if args.run_shap or args.shap_only:
            pytorch_cmd += [
                "--run-shap",
                "--shap-background-size",
                str(args.shap_background_size),
                "--shap-eval-size",
                str(args.shap_eval_size),
                "--shap-nsamples",
                str(args.shap_nsamples),
            ]
        if args.shap_only:
            pytorch_cmd += ["--shap-only"]
        if args.save_model:
            pytorch_cmd += ["--save-model"]
        if args.model_path:
            pytorch_cmd += ["--model-path", args.model_path]

        run_cmd(
            pytorch_cmd,
            cwd=code_dir,
            dry_run=args.dry_run,
        )

    if run_transformer:
        transformer_cmd = [
            py,
            "run_transformer_training.py",
            "--banks",
            ",".join(banks),
            "--cases",
            ",".join(transformer_cases),
            "--seeds",
            ",".join(seeds),
            "--epochs",
            str(args.epochs),
            "--forecast-days",
            str(args.forecast_days),
            "--time-steps",
            str(args.time_steps),
            "--batch-size",
            str(args.transformer_batch_size),
            "--hidden-size",
            str(args.transformer_hidden_size),
            "--dropout",
            str(args.transformer_dropout),
            "--learning-rate",
            str(args.transformer_learning_rate),
            "--n-heads",
            str(args.transformer_heads),
            "--n-layers",
            str(args.transformer_layers),
            "--ff-multiplier",
            str(args.transformer_ff_multiplier),
            "--moving-avg",
            str(args.transformer_moving_avg),
            "--tune-epochs",
            str(args.tune_epochs),
            "--tune-candidates",
            str(args.tune_candidates),
            "--hpo-method",
            args.transformer_hpo_method,
            "--hpo-generations",
            str(args.transformer_hpo_generations),
            "--output-root",
            str(args.transformer_output_root),
        ]
        if args.transformer_no_distill:
            transformer_cmd += ["--no-distill"]
        if args.run_shap:
            transformer_cmd += [
                "--run-shap",
                "--shap-background-size",
                str(args.shap_background_size),
                "--shap-eval-size",
                str(args.shap_eval_size),
                "--shap-nsamples",
                str(args.shap_nsamples),
            ]
        if args.save_model:
            transformer_cmd += ["--save-model"]

        run_cmd(
            transformer_cmd,
            cwd=code_dir,
            dry_run=args.dry_run,
        )

    if args.run_statistics and run_pytorch:
        run_cmd(
            [
                py,
                "analyze_study_results.py",
                "--runs-csv",
                str(args.output_root / "study_runs.csv"),
                "--baseline-case",
                args.pytorch_baseline_case,
                "--output-root",
                str(args.output_root),
            ],
            cwd=code_dir,
            dry_run=args.dry_run,
        )

    if args.run_statistics and run_transformer:
        run_cmd(
            [
                py,
                "analyze_study_results.py",
                "--runs-csv",
                str(args.transformer_output_root / "transformer_study_runs.csv"),
                "--baseline-case",
                args.transformer_baseline_case,
                "--output-root",
                str(args.transformer_output_root),
            ],
            cwd=code_dir,
            dry_run=args.dry_run,
        )

    if run_sota and run_sota_data:
        for bank in banks:
            run_cmd(
                [
                    py,
                    "train_sota_data.py",
                    "--bank",
                    bank,
                    "--output-root",
                    str(args.sota_output_data),
                    "--trials",
                    str(args.sota_trials),
                    "--seeds",
                    ",".join(seeds),
                    "--time-steps",
                    str(args.time_steps),
                ]
                + (
                    [
                        "--run-shap",
                        "--shap-eval-size",
                        str(args.shap_eval_size),
                    ]
                    if args.run_shap
                    else []
                ),
                cwd=code_dir,
                dry_run=args.dry_run,
            )

    if run_sota and run_sota_macro:
        for bank in banks:
            run_cmd(
                [
                    py,
                    "train_sota_macro.py",
                    "--bank",
                    bank,
                    "--output-root",
                    str(args.sota_output_macro),
                    "--trials",
                    str(args.sota_trials),
                    "--seeds",
                    ",".join(seeds),
                    "--time-steps",
                    str(args.time_steps),
                ]
                + (
                    [
                        "--run-shap",
                        "--shap-eval-size",
                        str(args.shap_eval_size),
                    ]
                    if args.run_shap
                    else []
                ),
                cwd=code_dir,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
