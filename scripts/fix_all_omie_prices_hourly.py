#!/usr/bin/env python

import pandas as pd
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "processed"

ALL_CSV = PROCESSED / "all_omie_prices.csv"
ALL_PARQUET = PROCESSED / "all_omie_prices.parquet"
ALL_DUCKDB = PROCESSED / "omie_prices.duckdb"

# 15-min data that we already fixed with split_hourly_and_15min.py
MIN15_CSV = PROCESSED / "15min" / "omie_15min_from_2025-10-01.csv"

CHANGE_DATE = pd.Timestamp("2025-10-01")


def load_all_prices() -> pd.DataFrame:
    """
    Load processed/all_omie_prices.csv.

    Standardise first 8 columns as:
    year, month, day, period, price_main, price_alt, timestamp, zone
    """

    if not ALL_CSV.exists():
        raise FileNotFoundError(f"{ALL_CSV} not found")

    # Peek a few rows without header to detect if first value is header or data
    preview = pd.read_csv(ALL_CSV, nrows=5, header=None)
    first_val = str(preview.iloc[0, 0])

    # Assume header if first value is not numeric
    try:
        float(first_val)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        df = pd.read_csv(ALL_CSV, header=0)
    else:
        df = pd.read_csv(ALL_CSV, header=None)

    # Must have at least 8 columns
    if df.shape[1] < 8:
        raise ValueError(
            f"Expected at least 8 columns in {ALL_CSV}, found {df.shape[1]}"
        )

    # Rename by POSITION
    col_map = {
        df.columns[0]: "year",
        df.columns[1]: "month",
        df.columns[2]: "day",
        df.columns[3]: "period",
        df.columns[4]: "price_main",
        df.columns[5]: "price_alt",
        df.columns[6]: "timestamp",
        df.columns[7]: "zone",
    }
    df = df.rename(columns=col_map)

    return df


def load_15min_data() -> pd.DataFrame:
    """
    Load corrected 15-min data from:
    processed/15min/omie_15min_from_2025-10-01.csv
    (this file should already have correct 15-min timestamps)
    """
    if not MIN15_CSV.exists():
        raise FileNotFoundError(
            f"{MIN15_CSV} not found. Run split_hourly_and_15min.py first."
        )

    df = pd.read_csv(MIN15_CSV)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["period"] = df["period"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    # ---------- 1. Load existing mixed all_omie_prices ----------
    df_all = load_all_prices()

    df_all["year"] = df_all["year"].astype(int)
    df_all["month"] = df_all["month"].astype(int)
    df_all["day"] = df_all["day"].astype(int)
    df_all["period"] = df_all["period"].astype(int)

    df_all["delivery_date"] = pd.to_datetime(df_all[["year", "month", "day"]])

    # Keep only data BEFORE 2025-10-01 (old hourly part)
    df_before = df_all.loc[df_all["delivery_date"] < CHANGE_DATE].copy()

    # Rebuild hourly timestamps for consistency
    df_before["timestamp"] = (
        df_before["delivery_date"]
        + pd.to_timedelta(df_before["period"] - 1, unit="h")
    )

    base_cols = [
        "year",
        "month",
        "day",
        "period",
        "price_main",
        "price_alt",
        "timestamp",
        "zone",
    ]
    df_before = df_before[base_cols]

    # ---------- 2. Load 15-min data and aggregate to hourly ----------
    df_15 = load_15min_data()
    df_15["delivery_date"] = pd.to_datetime(df_15[["year", "month", "day"]])
    df_15 = df_15.loc[df_15["delivery_date"] >= CHANGE_DATE].copy()

    # Hour (0..23) from the 15-min timestamps
    df_15["hour_of_day"] = df_15["timestamp"].dt.hour

    # Group by date, hour, zone: average of 4 quarters
    agg = (
        df_15.groupby(
            ["year", "month", "day", "zone", "hour_of_day"], as_index=False
        )
        .agg(
            price_main=("price_main", "mean"),
            price_alt=("price_alt", "mean"),
        )
    )

    # Map 0..23 → period 1..24
    agg["period"] = agg["hour_of_day"] + 1
    agg["delivery_date"] = pd.to_datetime(agg[["year", "month", "day"]])
    agg["timestamp"] = (
        agg["delivery_date"] + pd.to_timedelta(agg["hour_of_day"], unit="h")
    )

    df_hourly_from_15 = agg[
        ["year", "month", "day", "period", "price_main", "price_alt", "timestamp", "zone"]
    ]

    # ---------- 3. Combine old hourly + new hourly ----------
    df_final = pd.concat(
        [df_before, df_hourly_from_15],
        ignore_index=True,
    )

    df_final = df_final.sort_values(
        ["year", "month", "day", "period"]
    ).reset_index(drop=True)

    # ---------- 4. Overwrite all_omie_prices.* ----------
    # CSV
    df_final.to_csv(ALL_CSV, index=False)
    print(f"Overwritten CSV: {ALL_CSV}")

    # Parquet
    df_final.to_parquet(ALL_PARQUET, index=False)
    print(f"Overwritten Parquet: {ALL_PARQUET}")

    # DuckDB
    con = duckdb.connect(str(ALL_DUCKDB))
    con.register("prices_hourly_df", df_final)
    con.execute(
        "CREATE OR REPLACE TABLE omie_prices AS "
        "SELECT * FROM prices_hourly_df"
    )
    con.close()
    print(f"Overwritten DuckDB: {ALL_DUCKDB} (table omie_prices)")


if __name__ == "__main__":
    main()
