from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import modal


APP_DIR = Path(__file__).resolve().parent
REMOTE_APP_DIR = PurePosixPath("/app")
REMOTE_OUTPUT_DIR = PurePosixPath("/outputs")

app = modal.App("siml-stock-training")
volume = modal.Volume.from_name("siml-stock-training-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "numpy>=2.3.0",
        "pandas>=3.0.0",
        "scikit-learn>=1.7.0",
        "torch>=2.8.0",
        "shap>=0.49.1",
        "tqdm",
        "matplotlib",
    )
    .add_local_file(APP_DIR / "train_price_pytorch.py", remote_path=str(REMOTE_APP_DIR / "train_price_pytorch.py"))
    .add_local_file(
        APP_DIR / "train_price_pytorch_extended.py",
        remote_path=str(REMOTE_APP_DIR / "train_price_pytorch_extended.py"),
    )
    .add_local_file(
        APP_DIR / "train_price_pytorch_norm.py",
        remote_path=str(REMOTE_APP_DIR / "train_price_pytorch_norm.py"),
    )
    .add_local_file(
        APP_DIR / "train_price_pytorch_cnn.py",
        remote_path=str(REMOTE_APP_DIR / "train_price_pytorch_cnn.py"),
    )
    .add_local_file(
        APP_DIR / "train_price_pytorch_attention.py",
        remote_path=str(REMOTE_APP_DIR / "train_price_pytorch_attention.py"),
    )
    .add_local_file(APP_DIR / "train_price_transformers.py", remote_path=str(REMOTE_APP_DIR / "train_price_transformers.py"))
    .add_local_dir(APP_DIR / "data", remote_path=str(REMOTE_APP_DIR / "data"))
    .add_local_dir(APP_DIR / "data_with_macro", remote_path=str(REMOTE_APP_DIR / "data_with_macro"))
)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_pytorch_cases() -> list[str]:
    return [
        "lstm_data",
        "lstm_macro",
        "bilstm_data",
        "bilstm_macro",
        "lstm_data_ga",
        "lstm_macro_ga",
        "bilstm_data_ga",
        "bilstm_macro_ga",
    ]


def default_transformer_cases() -> list[str]:
    return [
        "informer_data",
        "informer_macro",
        "autoformer_data",
        "autoformer_macro",
    ]


PYTORCH_MODEL_SCRIPTS = {
    "lstm": "train_price_pytorch.py",
    "bilstm": "train_price_pytorch.py",
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


def build_pytorch_case_specs() -> dict[str, tuple[str, list[str]]]:
    specs: dict[str, tuple[str, list[str]]] = {}
    for model_type, script in PYTORCH_MODEL_SCRIPTS.items():
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


PYTORCH_CASE_SPECS = build_pytorch_case_specs()
PYTORCH_CASE_ARGS = {name: args for name, (_, args) in PYTORCH_CASE_SPECS.items()}

TRANSFORMER_CASE_ARGS = {
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


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    banks = [bank.upper() for bank in split_csv(args.banks)]
    seeds = [int(seed) for seed in split_csv(args.seeds)]
    pytorch_cases = split_csv(args.pytorch_cases)
    transformer_cases = split_csv(args.transformer_cases)

    unknown_pytorch = [case for case in pytorch_cases if case not in PYTORCH_CASE_ARGS]
    unknown_transformer = [case for case in transformer_cases if case not in TRANSFORMER_CASE_ARGS]
    if unknown_pytorch:
        raise ValueError(f"Unknown PyTorch cases: {unknown_pytorch}")
    if unknown_transformer:
        raise ValueError(f"Unknown transformer cases: {unknown_transformer}")

    jobs: list[dict[str, Any]] = []
    if not args.skip_pytorch:
        for bank in banks:
            for case in pytorch_cases:
                for seed in seeds:
                    jobs.append({"family": "pytorch", "bank": bank, "case": case, "seed": seed})

    if not args.skip_transformer:
        for bank in banks:
            for case in transformer_cases:
                for seed in seeds:
                    jobs.append({"family": "transformer", "bank": bank, "case": case, "seed": seed})

    return jobs


def build_remote_command(job: dict[str, Any], options: dict[str, Any]) -> list[str]:
    output_root = REMOTE_OUTPUT_DIR / options["run_name"] / job["bank"] / job["case"] / f"seed_{job['seed']}"
    if job["family"] == "pytorch":
        cmd = [
            sys.executable,
            PYTORCH_CASE_SPECS[job["case"]][0],
            "--bank",
            job["bank"],
            "--seed",
            str(job["seed"]),
            "--epochs",
            str(options["epochs"]),
            "--forecast-days",
            str(options["forecast_days"]),
            "--time-steps",
            str(options["time_steps"]),
            "--tune-epochs",
            str(options["tune_epochs"]),
            "--tune-candidates",
            str(options["tune_candidates"]),
            "--output-root",
            str(output_root),
        ] + PYTORCH_CASE_SPECS[job["case"]][1]
    else:
        cmd = [
            sys.executable,
            "train_price_transformers.py",
            "--bank",
            job["bank"],
            "--seed",
            str(job["seed"]),
            "--epochs",
            str(options["epochs"]),
            "--forecast-days",
            str(options["forecast_days"]),
            "--time-steps",
            str(options["time_steps"]),
            "--batch-size",
            str(options["transformer_batch_size"]),
            "--hidden-size",
            str(options["transformer_hidden_size"]),
            "--dropout",
            str(options["transformer_dropout"]),
            "--learning-rate",
            str(options["transformer_learning_rate"]),
            "--n-heads",
            str(options["transformer_heads"]),
            "--n-layers",
            str(options["transformer_layers"]),
            "--ff-multiplier",
            str(options["transformer_ff_multiplier"]),
            "--moving-avg",
            str(options["transformer_moving_avg"]),
            "--tune-epochs",
            str(options["tune_epochs"]),
            "--tune-candidates",
            str(options["tune_candidates"]),
            "--hpo-method",
            options["transformer_hpo_method"],
            "--hpo-generations",
            str(options["transformer_hpo_generations"]),
            "--output-root",
            str(output_root),
        ] + TRANSFORMER_CASE_ARGS[job["case"]]
        if options["transformer_no_distill"]:
            cmd += ["--no-distill"]

    if options["run_shap"]:
        cmd += [
            "--run-shap",
            "--shap-background-size",
            str(options["shap_background_size"]),
            "--shap-eval-size",
            str(options["shap_eval_size"]),
            "--shap-nsamples",
            str(options["shap_nsamples"]),
        ]
    if options["save_model"]:
        cmd += ["--save-model"]

    return cmd


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "L4"),
    volumes={str(REMOTE_OUTPUT_DIR): volume},
    timeout=60 * 60 * 8,
)
def train_one(job: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    cmd = build_remote_command(job, options)
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=REMOTE_APP_DIR, check=True)
    volume.commit()
    return {
        "family": job["family"],
        "bank": job["bank"],
        "case": job["case"],
        "seed": job["seed"],
        "output_root": str(REMOTE_OUTPUT_DIR / options["run_name"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit PyTorch/Transformer stock training jobs to Modal.")
    parser.add_argument("--banks", type=str, default="BBCA,BBNI,BBRI,BBTN,BDMN,BMRI,BNGA")
    parser.add_argument("--pytorch-cases", type=str, default=",".join(default_pytorch_cases()))
    parser.add_argument("--transformer-cases", type=str, default=",".join(default_transformer_cases()))
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--transformer-hpo-method", choices=["ga", "random"], default="ga")
    parser.add_argument("--transformer-hpo-generations", type=int, default=5)
    parser.add_argument("--transformer-hidden-size", type=int, default=64)
    parser.add_argument("--transformer-batch-size", type=int, default=32)
    parser.add_argument("--transformer-dropout", type=float, default=0.2)
    parser.add_argument("--transformer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-multiplier", type=int, default=4)
    parser.add_argument("--transformer-moving-avg", type=int, default=5)
    parser.add_argument("--transformer-no-distill", action="store_true")
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--shap-background-size", type=int, default=4)
    parser.add_argument("--shap-eval-size", type=int, default=2)
    parser.add_argument("--shap-nsamples", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="modal_training")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "forecast_days": args.forecast_days,
        "time_steps": args.time_steps,
        "tune_epochs": args.tune_epochs,
        "tune_candidates": args.tune_candidates,
        "transformer_hpo_method": args.transformer_hpo_method,
        "transformer_hpo_generations": args.transformer_hpo_generations,
        "transformer_hidden_size": args.transformer_hidden_size,
        "transformer_batch_size": args.transformer_batch_size,
        "transformer_dropout": args.transformer_dropout,
        "transformer_learning_rate": args.transformer_learning_rate,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ff_multiplier": args.transformer_ff_multiplier,
        "transformer_moving_avg": args.transformer_moving_avg,
        "transformer_no_distill": args.transformer_no_distill,
        "run_shap": args.run_shap,
        "save_model": args.save_model,
        "shap_background_size": args.shap_background_size,
        "shap_eval_size": args.shap_eval_size,
        "shap_nsamples": args.shap_nsamples,
        "run_name": args.run_name,
    }


@app.local_entrypoint()
def main(
    banks: str = "BBCA,BBNI,BBRI,BBTN,BDMN,BMRI,BNGA",
    pytorch_cases: str = ",".join(default_pytorch_cases()),
    transformer_cases: str = ",".join(default_transformer_cases()),
    seeds: str = "42,52,62,72,82",
    skip_pytorch: bool = False,
    skip_transformer: bool = False,
    epochs: int = 50,
    forecast_days: int = 30,
    time_steps: int = 60,
    tune_epochs: int = 8,
    tune_candidates: int = 6,
    transformer_hpo_method: str = "ga",
    transformer_hpo_generations: int = 5,
    transformer_hidden_size: int = 64,
    transformer_batch_size: int = 32,
    transformer_dropout: float = 0.2,
    transformer_learning_rate: float = 1e-3,
    transformer_heads: int = 4,
    transformer_layers: int = 2,
    transformer_ff_multiplier: int = 4,
    transformer_moving_avg: int = 5,
    transformer_no_distill: bool = False,
    run_shap: bool = False,
    save_model: bool = False,
    shap_background_size: int = 4,
    shap_eval_size: int = 2,
    shap_nsamples: int = 10,
    run_name: str = "modal_training",
    dry_run: bool = False,
) -> None:
    args = argparse.Namespace(**locals())
    jobs = build_jobs(args)
    options = options_from_args(args)
    print(f"Modal jobs: {len(jobs)}")
    if dry_run:
        for job in jobs:
            print(json.dumps({"job": job, "cmd": build_remote_command(job, options)}, default=str))
        return

    for result in train_one.starmap((job, options) for job in jobs):
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parsed = parse_args()
    jobs_for_preview = build_jobs(parsed)
    opts = options_from_args(parsed)
    print(
        "This file is intended to run with Modal. Use `modal run modal_train.py --help` "
        "or `modal run modal_train.py --dry-run`."
    )
    print(f"Prepared jobs: {len(jobs_for_preview)}")
    if parsed.dry_run:
        for preview_job in jobs_for_preview:
            print(json.dumps({"job": preview_job, "cmd": build_remote_command(preview_job, opts)}, default=str))
