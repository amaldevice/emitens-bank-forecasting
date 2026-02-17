from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHAP-only inference for many saved PyTorch checkpoints."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results_study_full/**/model.pt",
        help="Glob pattern (relative to this script directory) to find checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of checkpoints to process (0 = no limit).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have shap_kernel_feature_importance.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute.",
    )
    parser.add_argument("--shap-background-size", type=int, default=64)
    parser.add_argument("--shap-eval-size", type=int, default=32)
    parser.add_argument("--shap-nsamples", type=int, default=100)
    return parser.parse_args()


def read_summary_config(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    return config if isinstance(config, dict) else {}


def read_checkpoint_meta(model_path: Path) -> dict[str, object]:
    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {}


def resolve_run_context(model_path: Path) -> dict[str, object]:
    bank_dir = model_path.parent
    seed_dir = bank_dir.parent
    bank = bank_dir.name.upper()
    output_root = seed_dir
    summary_cfg = read_summary_config(bank_dir / "summary_pytorch.json")
    ckpt_meta = read_checkpoint_meta(model_path)

    use_macro = bool(summary_cfg.get("use_macro", ckpt_meta.get("use_macro", False)))
    data_dir = str(summary_cfg.get("data_dir", "data_with_macro" if use_macro else "data"))
    model_type = str(summary_cfg.get("model_type", ckpt_meta.get("model_type", "lstm")))
    time_steps = int(summary_cfg.get("time_steps", ckpt_meta.get("time_steps", 60)))
    hidden_size = int(summary_cfg.get("hidden_size", ckpt_meta.get("hidden_size", 32)))
    dropout = float(summary_cfg.get("dropout", ckpt_meta.get("dropout", 0.3)))
    forecast_days = int(summary_cfg.get("forecast_days", 30))
    seed = int(summary_cfg.get("seed", ckpt_meta.get("seed", 42)))

    return {
        "bank": bank,
        "output_root": output_root,
        "data_dir": data_dir,
        "model_type": model_type,
        "use_macro": use_macro,
        "time_steps": time_steps,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "forecast_days": forecast_days,
        "seed": seed,
    }


def build_cmd(args: argparse.Namespace, model_path: Path, ctx: dict[str, object]) -> list[str]:
    cmd = [
        sys.executable,
        "train_price_pytorch.py",
        "--bank",
        str(ctx["bank"]),
        "--data-dir",
        str(ctx["data_dir"]),
        "--output-root",
        str(ctx["output_root"]),
        "--model-type",
        str(ctx["model_type"]),
        "--time-steps",
        str(ctx["time_steps"]),
        "--hidden-size",
        str(ctx["hidden_size"]),
        "--dropout",
        str(ctx["dropout"]),
        "--forecast-days",
        str(ctx["forecast_days"]),
        "--seed",
        str(ctx["seed"]),
        "--model-path",
        str(model_path),
        "--run-shap",
        "--shap-only",
        "--shap-background-size",
        str(args.shap_background_size),
        "--shap-eval-size",
        str(args.shap_eval_size),
        "--shap-nsamples",
        str(args.shap_nsamples),
    ]
    if ctx["use_macro"]:
        cmd.append("--use-macro")
    return cmd


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_paths = sorted(root.glob(args.pattern))
    if args.limit > 0:
        model_paths = model_paths[: args.limit]

    if not model_paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {args.pattern}")

    print(f"Found checkpoints: {len(model_paths)}")
    done = 0
    skipped = 0
    failed = 0
    for i, model_path in enumerate(model_paths, start=1):
        bank_dir = model_path.parent
        shap_file = bank_dir / "shap_kernel_feature_importance.csv"
        if args.skip_existing and shap_file.exists():
            skipped += 1
            print(f"[{i}/{len(model_paths)}] SKIP existing SHAP: {model_path}")
            continue

        ctx = resolve_run_context(model_path)
        cmd = build_cmd(args, model_path, ctx)
        print(f"[{i}/{len(model_paths)}] RUN {model_path}")
        print("Command:", " ".join(cmd))
        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True, cwd=root)
            done += 1
        except subprocess.CalledProcessError:
            failed += 1

    print("\nSummary")
    print(f" - total: {len(model_paths)}")
    print(f" - done: {done}")
    print(f" - skipped: {skipped}")
    print(f" - failed: {failed}")


if __name__ == "__main__":
    main()
