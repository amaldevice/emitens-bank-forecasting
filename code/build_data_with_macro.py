from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_worldbank_series(file_path: Path, value_name: str, start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=4)
    df_idn = df[df["Country Name"] == "Indonesia"].copy()
    if df_idn.empty:
        raise ValueError(f"Indonesia row not found in {file_path}")

    years = [str(year) for year in range(start_year, end_year + 1)]
    values: dict[int, float | None] = {}
    for year in years:
        values[int(year)] = df_idn.iloc[0].get(year)

    out = pd.DataFrame({"year": list(values.keys()), value_name: list(values.values())})
    return out


def calculate_cpi_index(df_inflation: pd.DataFrame, base_year: int = 2010, base_value: float = 100.0) -> pd.DataFrame:
    df = df_inflation.sort_values("year").reset_index(drop=True).copy()
    df["cpi_index"] = pd.NA
    idx = df.index[df["year"] == base_year]
    if len(idx) == 0:
        raise ValueError(f"Base year {base_year} not found in inflation data")
    base_idx = int(idx[0])
    df.loc[base_idx, "cpi_index"] = base_value

    for i in range(base_idx + 1, len(df)):
        prev_cpi = df.loc[i - 1, "cpi_index"]
        inf = df.loc[i, "inflation_rate"]
        if pd.notna(prev_cpi) and pd.notna(inf):
            df.loc[i, "cpi_index"] = float(prev_cpi) * (1 + float(inf) / 100)

    for i in range(base_idx - 1, -1, -1):
        next_cpi = df.loc[i + 1, "cpi_index"]
        inf = df.loc[i, "inflation_rate"]
        if pd.notna(next_cpi) and pd.notna(inf):
            df.loc[i, "cpi_index"] = float(next_cpi) / (1 + float(inf) / 100)

    return df


def load_annual_macro(macro_dir: Path, start_year: int = 2004, end_year: int = 2024) -> pd.DataFrame:
    inflation_path = (
        macro_dir
        / "API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_23195"
        / "API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_23195.csv"
    )
    exchange_path = macro_dir / "API_IDN_PA.NUS.FCRF_en_csv_v2_111434.csv"

    inflation = load_worldbank_series(inflation_path, "inflation_rate", start_year, end_year)
    inflation = calculate_cpi_index(inflation)
    exchange = load_worldbank_series(exchange_path, "exchange_rate_usd_idr", start_year, end_year)

    annual = inflation.merge(exchange, on="year", how="outer").sort_values("year").reset_index(drop=True)
    return annual


def load_bi_rate_monthly(macro_dir: Path) -> pd.DataFrame:
    bi_path = macro_dir / "indonesia_bi_rate_fred.csv"
    bi = pd.read_csv(bi_path)
    bi.columns = ["Date", "bi_rate"]
    bi["Date"] = pd.to_datetime(bi["Date"])
    bi = bi.sort_values("Date").dropna(subset=["bi_rate"])
    return bi[["Date", "bi_rate"]]


def load_ihsg_daily(macro_dir: Path) -> pd.DataFrame:
    ihsg_path = macro_dir / "ihsg_daily_2004_2024.csv"
    ihsg = pd.read_csv(ihsg_path)
    ihsg["Date"] = pd.to_datetime(ihsg["Date"])
    ihsg = ihsg.sort_values("Date")
    ihsg["ihsg_close"] = ihsg["Close"].astype(float)
    ihsg["ihsg_return_1d"] = ihsg["ihsg_close"].pct_change().mul(100)
    ihsg["ihsg_return_1d"] = ihsg["ihsg_return_1d"].fillna(0.0)
    return ihsg[["Date", "ihsg_close", "ihsg_return_1d"]]


def enrich_bank_file(
    input_csv: Path,
    output_csv: Path,
    annual_macro: pd.DataFrame,
    bi_monthly: pd.DataFrame,
    ihsg_daily: pd.DataFrame,
    annual_lag_years: int,
) -> None:
    df = pd.read_csv(input_csv)
    if "Date" not in df.columns:
        raise ValueError(f"Date column not found in {input_csv}")

    df["Date"] = pd.to_datetime(df["Date"])
    original_desc = df["Date"].is_monotonic_decreasing
    df = df.sort_values("Date").reset_index(drop=True)
    df["year"] = df["Date"].dt.year - annual_lag_years

    merged = df.merge(annual_macro, on="year", how="left")
    merged = pd.merge_asof(merged.sort_values("Date"), bi_monthly, on="Date", direction="backward")
    merged = pd.merge_asof(merged.sort_values("Date"), ihsg_daily, on="Date", direction="backward")
    merged = merged.drop(columns=["year"])

    if original_desc:
        merged = merged.sort_values("Date", ascending=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bank CSVs enriched with macro features.")
    parser.add_argument("--source-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_with_macro"))
    parser.add_argument(
        "--macro-dir",
        type=Path,
        default=Path("data_macro"),
    )
    parser.add_argument(
        "--annual-lag-years",
        type=int,
        default=0,
        help="Lag annual macro values by N years to reduce look-ahead risk. 0 keeps same-year mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annual_macro = load_annual_macro(args.macro_dir)
    bi_monthly = load_bi_rate_monthly(args.macro_dir)
    ihsg_daily = load_ihsg_daily(args.macro_dir)

    bank_files = sorted(args.source_dir.glob("*.csv"))
    if not bank_files:
        raise FileNotFoundError(f"No CSV files found in {args.source_dir}")

    print(f"Source files: {len(bank_files)}")
    print(f"Annual lag years: {args.annual_lag_years}")
    print("Adding macro columns: inflation_rate, cpi_index, exchange_rate_usd_idr, bi_rate, ihsg_close, ihsg_return_1d")

    for input_csv in bank_files:
        output_csv = args.output_dir / input_csv.name
        enrich_bank_file(
            input_csv=input_csv,
            output_csv=output_csv,
            annual_macro=annual_macro,
            bi_monthly=bi_monthly,
            ihsg_daily=ihsg_daily,
            annual_lag_years=args.annual_lag_years,
        )
        out_df = pd.read_csv(output_csv, nrows=1)
        print(f"Built: {output_csv} | columns={len(out_df.columns)}")


if __name__ == "__main__":
    main()
