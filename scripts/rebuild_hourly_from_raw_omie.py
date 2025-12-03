#!/usr/bin/env python

import os
import re
from pathlib import Path
from datetime import datetime, date

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = REPO_ROOT / "processed"

CSV_PATH = PROCESSED_DIR / "all_omie_prices.csv"
PARQUET_PATH = PROCESSED_DIR / "all_omie_prices.parquet"
DUCKDB_PATH = PROCESSED_DIR / "omie_prices.duckdb"

# Date when OMIE switched to 15-min (delivery date)
CHANGE_DATE = date(2025, 10, 1)


# ---------- helpers ----------

def parse_daily_file(filepath: Path) -> pd.DataFrame:
    """
    Parse a raw OMIE file like marginalpdbc_YYYYMMDD.1/.2

    This matches your existing clean_file() logic:

      - sep=";"
      - skip first line (MARGINALPDBC;)
      - drop any rows containing '*'
      - drop column 6
      - columns: Year, Month, Day, Period, Price1, Price2

    BUT we do NOT turn Period into hours here. We keep the 1..96 periods
    and let a separate function aggregate to hourly.
    """

    df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=1,
        header=None,
        engine="python",
    )

    # Drop rows with any "*" (same as your old code)
    mask_star = df.apply(lambda x: x.astype(str).str.contains(r"\*").any(), axis=1)
    df = df[~mask_star]

    # Make sure we at least have 7 cols, drop the last one (duplicate/empty)
    if df.shape[1] < 7:
        raise ValueError(f"Unexpected column count in {filepath}: {df.shape[1]}")

    df = df.drop(columns=[6])

    df.columns = ["Year", "Month", "Day", "Period", "Price1", "Price2"]

    # Types
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Day"] = df["Day"].astype(int)
    df["Period"] = df["Period"].astype(int)
    df["Price1"] = df["Price1"].astype(float)
    df["Price2"] = df["Price2"].astype(float)

    return df


def hourly_from_daily_file(filepath: Path) -> pd.DataFrame:
    """
    Take one daily OMIE file and return a HOURLY DataFrame with columns:

      Year, Month, Day, Hour, Price1, Price2, Datetime, Country

    Logic:

      - If Period goes up to 96 → treat as 96 x 15-min, average blocks of 4.
      - If Period <= 24 → treat as already-hourly (just rename Period->Hour).
    """

    df = parse_daily_file(filepath)

    # Extract date from filename
    m = re.search(r"marginalpdbc_(\d{8})\.(\d+)$", filepath.name)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {filepath.name}")

    date_str, version_str = m.groups()
    file_date = datetime.strptime(date_str, "%Y%m%d").date()

    # Decide country from suffix (same logic you used)
    country = "Spain" if filepath.suffix == ".1" else "Portugal"

    max_period = df["Period"].max()

    if max_period > 24:
        # 96 x 15-min → group by hour_of_day = (period-1)//4
        df["hour_of_day"] = (df["Period"] - 1) // 4

        agg = (
            df.groupby("hour_of_day", as_index=False)
            .agg(
                Price1=("Price1", "mean"),
                Price2=("Price2", "mean"),
            )
            .sort_values("hour_of_day")
        )

        # Round to 2 decimals like OMIE style
        agg["Price1"] = agg["Price1"].round(2)
        agg["Price2"] = agg["Price2"].round(2)

        agg["Hour"] = agg["hour_of_day"] + 1

    else:
        # Already hourly (24 rows)
        agg = df.copy()
        agg = agg.sort_values("Period").rename(columns={"Period": "Hour"})
        agg = agg[["Hour", "Price1", "Price2"]]

    # Attach date and Datetime
    agg["Year"] = file_date.year
    agg["Month"] = file_date.month
    agg["Day"] = file_date.day

    # Datetime = delivery date + (Hour-1) hours
    agg["Datetime"] = pd.to_datetime(
        {
            "year": agg["Year"],
            "month": agg["Month"],
            "day": agg["Day"],
        }
    ) + pd.to_timedelta(agg["Hour"] - 1, unit="h")

    agg["Country"] = country

    # Final column order matching your DB
    agg = agg[
        ["Year", "Month", "Day", "Hour", "Price1", "Price2", "Datetime", "Country"]
    ]

    return agg


def build_hourly_from_raw_files() -> pd.DataFrame:
    """
    Build hourly data for ALL dates >= CHANGE_DATE
    from raw OMIE files in data/marginalpdbc_YYYYMMDD.{1,2}
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    paths = sorted(DATA_DIR.glob("marginalpdbc_*.1")) + sorted(
        DATA_DIR.glob("marginalpdbc_*.2")
    )

    all_dfs = []

    for path in paths:
        m = re.search(r"marginalpdbc_(\d{8})\.(\d+)$", path.name)
        if not m:
            continue

        date_str = m.group(1)
        file_date = datetime.strptime(date_str, "%Y%m%d").date()

        # Only rebuild region with 15-min / possibly-broken data
        if file_date < CHANGE_DATE:
            continue

        print(f"🧮 Rebuilding hourly for {path.name}")
        df_day = hourly_from_daily_file(path)
        all_dfs.append(df_day)

    if not all_dfs:
        raise RuntimeError(
            f"No OMIE raw files found for dates >= {CHANGE_DATE} in {DATA_DIR}"
        )

    hourly = pd.concat(all_dfs, ignore_index=True)
    hourly = hourly.sort_values(["Datetime", "Country"]).reset_index(drop=True)

    return hourly


# ---------- main ----------

def main():
    # 1) Load existing hourly DB (current mixed file)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df_all = pd.read_csv(CSV_PATH)
    if "Datetime" not in df_all.columns:
        raise ValueError("Expected column 'Datetime' in all_omie_prices.csv")

    df_all["Datetime"] = pd.to_datetime(df_all["Datetime"])
    df_all["delivery_date"] = df_all["Datetime"].dt.date

    # 2) Keep only data BEFORE the change date
    df_before = df_all[df_all["delivery_date"] < CHANGE_DATE].copy()
    df_before = df_before.drop(columns=["delivery_date"])

    print(
        f"✅ Keeping {len(df_before)} rows before {CHANGE_DATE} "
        f"from existing hourly DB"
    )

    # 3) Rebuild hourly data from raw files for CHANGE_DATE and after
    df_after = build_hourly_from_raw_files()
    print(f"✅ Rebuilt {len(df_after)} rows from raw OMIE daily files")

    # 4) Combine and sort
    df_final = pd.concat([df_before, df_after], ignore_index=True)
    df_final = df_final.sort_values(["Datetime", "Country"]).reset_index(drop=True)

    # 5) Save CSV
    df_final.to_csv(CSV_PATH, index=False)
    print(f"💾 Overwrote {CSV_PATH}")

    # 6) Save Parquet
    df_final.to_parquet(PARQUET_PATH, index=False)
    print(f"💾 Overwrote {PARQUET_PATH}")

    # 7) Save DuckDB (table name 'prices' as in your scripts)
    con = duckdb.connect(DUCKDB_PATH)
    con.execute("DROP TABLE IF EXISTS prices")
    con.register("df", df_final)
    con.execute("CREATE TABLE prices AS SELECT * FROM df")
    con.close()
    print(f"💾 Overwrote {DUCKDB_PATH} (table 'prices')")

    print("🎉 Hourly OMIE database successfully rebuilt for October 2025 onwards!")


if __name__ == "__main__":
    main()
