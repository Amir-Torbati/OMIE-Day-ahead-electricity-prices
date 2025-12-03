#!/usr/bin/env python

import pandas as pd
from pathlib import Path
import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "processed"

ALL_CSV = PROCESSED / "all_omie_prices.csv"
ALL_PARQUET = PROCESSED / "all_omie_prices.parquet"
ALL_DUCKDB = PROCESSED / "omie_prices.duckdb"

MIN15_CSV = PROCESSED / "15min" / "omie_15min_from_2025-10-01.csv"

CHANGE_DATE = pd.Timestamp("2025-10-01")


def load_all_prices():
    """
    Load processed/all_omie_prices.csv and return:
      - df_work : dataframe with canonical names (year, month, day, period, price_main, price_alt, timestamp, zone)
      - has_header : bool, whether original file had a header row
      - orig_cols : list of original column names (to restore structure later)

    We assume the first 8 columns are the ones we care about and match the data you showed.
    """

    if not ALL_CSV.exists():
        raise FileNotFoundError(f"{ALL_CSV} not found")

    # Peek few rows without header to detect header/data
    preview = pd.read_csv(ALL_CSV, nrows=5, header=None)
    first_val = str(preview.iloc[0, 0])

    try:
        float(first_val)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        df_raw = pd.read_csv(ALL_CSV, header=0)
    else:
        df_raw = pd.read_csv(ALL_CSV, header=None)

    orig_cols = list(df_raw.columns)

    if df_raw.shape[1] < 8:
        raise ValueError(
            f"Expected at least 8 columns in {ALL_CSV}, found {df_raw.shape[1]}"
        )

    # Work on a copy with canonical column names for the first 8 columns
    df = df_raw.copy()

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

    return df, has_header, orig_cols


def load_15min_data() -> pd.DataFrame:
    """
    Load corrected 15-min data from:
    processed/15min/omie_15min_from_2025-10-01.csv
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
    # ---------- 1. Load current mixed all_omie_prices ----------
    df_all, has_header, orig_cols = load_all_prices()

    # Make sure we can work with these columns
    df_all["year"] = df_all["year"].astype(int)
    df_all["month"] = df_all["month"].astype(int)
    df_all["day"] = df_all["day"].astype(int)
    df_all["period"] = df_all["period"].astype(int)

    # Compute delivery date from year/month/day
    df_all["delivery_date"] = pd.to_datetime(df_all[["year", "month", "day"]])

    # Keep only data BEFORE the change date (old hourly, good data)
    df_before = df_all.loc[df_all["delivery_date"] < CHANGE_DATE].copy()

    # Rebuild hourly timestamps to be sure structure is consistent
    df_before["timestamp"] = (
        df_before["delivery_date"]
        + pd.to_timedelta(df_before["period"] - 1, unit="h")
    )

    # Only keep the 8 main columns (as in your original example)
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

    # Hour of day 0..23 from 15-min timestamps
    df_15["hour_of_day"] = df_15["timestamp"].dt.hour

    # Average of 4 quarters for each hour
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

    # ---------- 4. Restore original column names / structure ----------
    # We assume original file had exactly 8 columns like your sample.
    # If so, rename the 8 canonical columns back to the original names,
    # preserving column order and header/no-header style.

    if len(orig_cols) == 8:
        # Map from canonical → original
        rename_back = {
            "year": orig_cols[0],
            "month": orig_cols[1],
            "day": orig_cols[2],
            "period": orig_cols[3],
            "price_main": orig_cols[4],
            "price_alt": orig_cols[5],
            "timestamp": orig_cols[6],
            "zone": orig_cols[7],
        }
        df_final = df_final.rename(columns=rename_back)
        df_final = df_final[orig_cols]  # enforce original column order
    else:
        # If there were more than 8 columns, we keep canonical names;
        # but in your repo it looks like there are exactly 8.
        pass

    # ---------- 5. Overwrite all_omie_prices.* keeping structure ----------

    # CSV: keep header style exactly (header row or not)
    df_final.to_csv(ALL_CSV, index=False, header=has_header)
    print(f"Overwritten CSV (same structure): {ALL_CSV}")

    # Parquet & DuckDB: they don't care about header, just column names;
    # they will use whatever df_final has now (original names if 8-cols).
    df_final.to_parquet(ALL_PARQUET, index=False)
    print(f"Overwritten Parquet: {ALL_PARQUET}")

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
