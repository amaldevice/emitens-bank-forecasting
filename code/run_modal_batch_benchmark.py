from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CASES = ["informer_data", "autoformer_data"]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Transformer batch sizes on Modal.")
    parser.add_argument("--banks", type=str, default="BBCA")
    parser.add_argument("--cases", type=str, default=",".join(DEFAULT_CASES))
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--batch-sizes", type=str, default="64,128,256,512")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--modal-gpu", type=str, default="L4")
    parser.add_argument("--run-prefix", type=str, default="batch_benchmark")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print("Command:", " ".join(cmd))
    if dry_run:
        return
    merged_env = dict(env)
    subprocess.run(cmd, check=True, env=merged_env)


def summarize_run(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for history_path in output_root.glob("batch_*/**/training_history_transformer.csv"):
        parts = history_path.parts
        batch_dir = next(part for part in parts if part.startswith("batch_"))
        batch_size = int(batch_dir.replace("batch_", ""))
        bank_dir = history_path.parent
        summary_path = bank_dir / "summary_transformer.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with history_path.open(newline="", encoding="utf-8") as fp:
            history = list(csv.DictReader(fp))
        epoch_seconds = [float(row["epoch_seconds"]) for row in history if row.get("epoch_seconds")]
        train_seconds = [float(row["train_seconds"]) for row in history if row.get("train_seconds")]
        val_seconds = [float(row["val_seconds"]) for row in history if row.get("val_seconds")]
        rows.append(
            {
                "batch_size": batch_size,
                "bank": summary["config"]["bank_name"],
                "case": history_path.parents[2].name,
                "seed": summary["config"]["seed"],
                "model_type": summary["config"]["model_type"],
                "epochs": len(history),
                "mean_epoch_seconds": sum(epoch_seconds) / len(epoch_seconds),
                "mean_train_seconds": sum(train_seconds) / len(train_seconds),
                "mean_val_seconds": sum(val_seconds) / len(val_seconds),
                "rmse": summary["metrics"]["rmse"],
                "mae": summary["metrics"]["mae"],
                "mape": summary["metrics"]["mape"],
                "history_path": str(history_path),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MODAL_GPU"] = args.modal_gpu

    run_names = []
    for batch_size in [int(item) for item in split_csv(args.batch_sizes)]:
        run_name = f"{args.run_prefix}_batch_{batch_size}"
        run_names.append((batch_size, run_name))
        cmd = [
            sys.executable,
            "-m",
            "modal",
            "run",
            "modal_train.py",
            "--banks",
            args.banks,
            "--skip-pytorch",
            "--transformer-cases",
            args.cases,
            "--seeds",
            args.seeds,
            "--epochs",
            str(args.epochs),
            "--time-steps",
            str(args.time_steps),
            "--transformer-batch-size",
            str(batch_size),
            "--transformer-hidden-size",
            str(args.hidden_size),
            "--transformer-heads",
            str(args.heads),
            "--transformer-layers",
            str(args.layers),
            "--run-name",
            run_name,
        ]
        run_cmd(cmd, args.dry_run, env)

        if args.download and not args.dry_run:
            dest = code_dir / "results_batch_benchmark" / f"batch_{batch_size}"
            dest.mkdir(parents=True, exist_ok=True)
            get_cmd = [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "get",
                "siml-stock-training-outputs",
                run_name,
                str(dest),
                "--force",
            ]
            run_cmd(get_cmd, False, env)

    if args.download and not args.dry_run:
        rows = summarize_run(code_dir / "results_batch_benchmark")
        out_csv = code_dir / "results_batch_benchmark" / "batch_benchmark_summary.csv"
        if rows:
            with out_csv.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
