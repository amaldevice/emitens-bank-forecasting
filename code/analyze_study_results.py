from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(
    errors_model: np.ndarray, errors_baseline: np.ndarray, power: int = 2
) -> tuple[float, float]:
    """
    DM test for one-step-ahead forecasts with Newey-West style variance estimate.
    Returns (dm_stat, two_sided_pvalue).
    """
    e1 = np.asarray(errors_model, dtype=float)
    e2 = np.asarray(errors_baseline, dtype=float)
    if len(e1) != len(e2):
        raise ValueError("Error arrays must have same length")
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    if n < 5:
        return np.nan, np.nan

    d_bar = float(np.mean(d))
    m = max(1, int(np.floor(n ** (1 / 3))))
    gamma0 = np.var(d, ddof=1)
    var_d = gamma0
    for k in range(1, m + 1):
        cov = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        weight = 1 - k / (m + 1)
        var_d += 2 * weight * cov

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(var_d / n)
    p = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multi-seed study results.")
    parser.add_argument("--runs-csv", type=Path, default=Path("results_study/study_runs.csv"))
    parser.add_argument("--baseline-case", type=str, default="lstm_data")
    parser.add_argument("--output-root", type=Path, default=Path("results_study"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.runs_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)

    # 1) Mean/std table for paper.
    agg = (
        df.groupby(["bank", "case"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            n_runs=("seed", "count"),
            model_type=("model_type", "first"),
            use_macro=("use_macro", "first"),
            optimize=("optimize", "first"),
        )
        .sort_values(["bank", "rmse_mean"])
    )
    agg_out = args.output_root / "summary_by_bank_case.csv"
    agg.to_csv(agg_out, index=False)

    # 2) Best-vs-baseline significance.
    stat_rows: list[dict[str, object]] = []
    for bank in sorted(df["bank"].unique()):
        bank_df = df[df["bank"] == bank].copy()
        base_df = bank_df[bank_df["case"] == args.baseline_case].copy()
        if base_df.empty:
            continue

        best_case = (
            bank_df.groupby("case", as_index=False)["rmse"].mean().sort_values("rmse").iloc[0]["case"]
        )
        best_df = bank_df[bank_df["case"] == best_case].copy()

        merged = best_df.merge(base_df, on="seed", suffixes=("_best", "_base"))
        if merged.empty:
            continue

        # Paired Wilcoxon on seed-level RMSE.
        try:
            w_stat, w_p = stats.wilcoxon(merged["rmse_best"], merged["rmse_base"], alternative="two-sided")
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        # DM test on concatenated residuals across seeds.
        all_best_err, all_base_err = [], []
        for _, row in merged.iterrows():
            p_best = Path(row["test_predictions_path_best"])
            p_base = Path(row["test_predictions_path_base"])
            df_best = pd.read_csv(p_best)
            df_base = pd.read_csv(p_base)
            # Align by date to ensure fair comparison.
            mdf = df_best.merge(
                df_base,
                on="date",
                suffixes=("_best", "_base"),
                how="inner",
            )
            e_best = (mdf["actual_best"] - mdf["predicted_best"]).to_numpy()
            e_base = (mdf["actual_base"] - mdf["predicted_base"]).to_numpy()
            all_best_err.append(e_best)
            all_base_err.append(e_base)

        dm_stat, dm_p = diebold_mariano_test(np.concatenate(all_best_err), np.concatenate(all_base_err), power=2)
        stat_rows.append(
            {
                "bank": bank,
                "baseline_case": args.baseline_case,
                "best_case": best_case,
                "best_rmse_mean": float(best_df["rmse"].mean()),
                "baseline_rmse_mean": float(base_df["rmse"].mean()),
                "rmse_improvement_pct": float(
                    (base_df["rmse"].mean() - best_df["rmse"].mean()) / base_df["rmse"].mean() * 100
                ),
                "wilcoxon_stat": float(w_stat) if not pd.isna(w_stat) else np.nan,
                "wilcoxon_p": float(w_p) if not pd.isna(w_p) else np.nan,
                "dm_stat_mse_loss": dm_stat,
                "dm_p_mse_loss": dm_p,
                "n_seeds": int(len(merged)),
            }
        )

    stats_df = pd.DataFrame(stat_rows).sort_values("bank")
    stats_out = args.output_root / "best_vs_baseline_stats.csv"
    stats_df.to_csv(stats_out, index=False)

    print(f"Saved: {agg_out}")
    print(f"Saved: {stats_out}")
    if not stats_df.empty:
        print("\nBest-vs-baseline significance snapshot:")
        print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
