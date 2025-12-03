#!/usr/bin/env python

import pandas as pd
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "processed"

INPUT_CSV = PROCESSED / "all_omie_prices.csv"

# New 15-min folder
OUT_15_DIR = PROCESSED / "15min"
OUT_15_DIR.mkdir(parents=True, exist_ok=True)

OUT_15_CSV = OUT_15_DIR / "omie_15min_from_2025-10-01.csv"
OUT_15_PARQUET = OUT_15_DIR / "omie_15min_from_2025-10-01.parquet"
OUT_15_DUCKDB = OUT_15_DIR / "omie_15min_from_2025-10-01.duckdb"

# Optional: clean hourly-only data until 2025-09-30, also in 3 formats
OUT_HOURLY_CSV = PROCESSED / "all_omie_prices_hourly_until_2025-09-30.csv"
OUT_HOURLY_PARQUET = PROCESSED / "all_omie_prices_hourly_until_2025-09-30.parquet"
OUT_HOURLY_DUCKDB = PROCESSED / "omie_prices_hourly_until_2025-09-30.duckdb"

CHANGE_DATE = pd.Timestamp("2025-10-01")


def load_all_prices() -> pd.DataFrame:
    """
    Load all_omie_prices.csv, detecting automatically whether
    there is a header row or not.

    We then standardise the first 8 columns to:
    year, month, day, period, price_main, price_alt, timestamp, zone
    regardless of the original column names.
    """

    # Peek a few rows with NO header to inspect first value
    preview = pd.read_csv(INPUT_CSV, nrows=5, header=None)
    first_val = str(preview.iloc[0, 0])

    # Try to interpret the first value as a number (year)
    has_header = False
    try:
        float(first_val)  # if this works, it's likely data, not header
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        # Read with header row, then rename columns by position
        df = pd.read_csv(INPUT_CSV, header=0)
    else:
        # No header: read everything as data
        df = pd.read_csv(INPUT_CSV, header=None)

    # Ensure we have at least 8 columns
    if df.shape[1] < 8:
        raise ValueError(
            f"Expected at least 8 columns in {INPUT_CSV}, found {df.shape[1]}"
        )

    # Standardise column names by POSITION, not by their current names
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


def main():
    df = load_all_prices()

    # Ensure correct dtypes
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["period"] = df["period"].astype(int)

    # OMIE delivery date (market day)
    df["delivery_date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Split: hourly (< 2025-10-01) vs 15-min (>= 2025-10-01)
    mask_15 = df["delivery_date"] >= CHANGE_DATE
    df_15 = df.loc[mask_15].copy()
    df_hourly = df.loc[~mask_15].copy()

    # ✅ Rebuild timestamps for 15-min data
    # period: 1..96 → 00:00, 00:15, ..., 23:45
    df_15["timestamp"] = (
        df_15["delivery_date"]
        + pd.to_timedelta((df_15["period"] - 1) * 15, unit="m")
    )

    # (Optional but good) recompute timestamps for hourly data too
    df_hourly["timestamp"] = (
        df_hourly["delivery_date"]
        + pd.to_timedelta(df_hourly["period"] - 1, unit="h")
    )

    cols = [
        "year",
        "month",
        "day",
        "period",
        "price_main",
        "price_alt",
        "timestamp",
        "zone",
    ]

    # ---------- WRITE 15-MIN DATA (3 formats) ----------
    df_15[cols].to_csv(OUT_15_CSV, index=False)
    df_15[cols].to_parquet(OUT_15_PARQUET, index=False)

    con_15 = duckdb.connect(str(OUT_15_DUCKDB))
    con_15.register("prices_15_df", df_15[cols])
    con_15.execute(
        "CREATE OR REPLACE TABLE omie_prices_15min AS "
        "SELECT * FROM prices_15_df"
    )
    con_15.close()

    # ---------- WRITE HOURLY-ONLY DATA UNTIL 2025-09-30 (3 formats) ----------
    df_hourly[cols].to_csv(OUT_HOURLY_CSV, index=False)
    df_hourly[cols].to_parquet(OUT_HOURLY_PARQUET, index=False)

    con_h = duckdb.connect(str(OUT_HOURLY_DUCKDB))
    con_h.register("prices_hourly_df", df_hourly[cols])
    con_h.execute(
        "CREATE OR REPLACE TABLE omie_prices_hourly AS "
        "SELECT * FROM prices_hourly_df"
    )
    con_h.close()

    print(f"15-min data written to:")
    print(f"  {OUT_15_CSV}")
    print(f"  {OUT_15_PARQUET}")
    print(f"  {OUT_15_DUCKDB}")
    print()
    print(f"Hourly-only data (until 2025-09-30) written to:")
    print(f"  {OUT_HOURLY_CSV}")
    print(f"  {OUT_HOURLY_PARQUET}")
    print(f"  {OUT_HOURLY_DUCKDB}")


if __name__ == "__main__":
    main()

