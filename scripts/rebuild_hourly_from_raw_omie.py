#!/usr/bin/env python

import re
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "processed"
DATA_DIR = REPO_ROOT / "data"

ALL_CSV = PROCESSED / "all_omie_prices.csv"
ALL_PARQUET = PROCESSED / "all_omie_prices.parquet"
ALL_DUCKDB = PROCESSED / "omie_prices.duckdb"

CHANGE_DATE = pd.Timestamp("2025-10-01")


# ---------- helpers ----------

def load_all_prices():
    """
    Load processed/all_omie_prices.csv and return:
      - df_all with canonical columns (year, month, day, period, price_main, price_alt, timestamp, zone)
      - has_header: whether CSV originally had header row
      - orig_cols: original column names (so we can restore structure)
    """

    if not ALL_CSV.exists():
        raise FileNotFoundError(f"{ALL_CSV} not found")

    # Peek at first line to detect if there's a header
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

    df = df_raw.copy()

    # Canonical names for first 8 columns
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


def parse_omie_raw_file(path: Path) -> pd.DataFrame:
    """
    Read one OMIE daily raw file like:
      data/marginalpdbc_20251001.1  or  data/marginalpdbc_20251001.2

    Lines look like:
      2025;10;01;1;105.1;105.1;
      year;month;day;period;price_main;price_alt;[empty]

    Returns DataFrame:
      year, month, day, period, price_main, price_alt
      (96 rows, period 1..96)
    """

    df_raw = pd.read_csv(
        path,
        sep=";",
        header=None,
        engine="python",
    )

    # Keep rows where first column looks like a year (4 digits)
    mask = df_raw[0].astype(str).str.match(r"^\d{4}$")
    df_raw = df_raw[mask].copy()

    # Only first 6 columns
    df_raw = df_raw.iloc[:, :6]
    df_raw.columns = [
        "year",
        "month",
        "day",
        "period",
        "price_main",
        "price_alt",
    ]

    # Types
    df_raw["year"] = df_raw["year"].astype(int)
    df_raw["month"] = df_raw["month"].astype(int)
    df_raw["day"] = df_raw["day"].astype(int)
    df_raw["period"] = df_raw["period"].astype(int)
    df_raw["price_main"] = df_raw["price_main"].astype(float)
    df_raw["price_alt"] = df_raw["price_alt"].astype(float)

    return df_raw


def hourly_from_raw_file(path: Path) -> pd.DataFrame:
    """
    For one OMIE raw file (96 periods of 15-min),
    aggregate to 24 hourly prices (average of each block of 4).

    Returns DataFrame with canonical columns:
      year, month, day, period, price_main, price_alt, timestamp, zone
    """

    m = re.search(r"marginalpdbc_(\d{8})\.(\d+)$", path.name)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {path}")

    date_str = m.group(1)
    date = pd.to_datetime(date_str, format="%Y%m%d")

    df = parse_omie_raw_file(path)

    # Sanity check: single date in file
    unique_dates = df[["year", "month", "day"]].drop_duplicates()
    if len(unique_dates) != 1:
        raise ValueError(f"File {path} contains multiple dates: {unique_dates}")

    # Hour index 0..23 from period 1..96
    df["hour_of_day"] = (df["period"] - 1) // 4

    # Average each block of 4 quarters per hour
    agg = (
        df.groupby("hour_of_day", as_index=False)
        .agg(
            price_main=("price_main", "mean"),
            price_alt=("price_alt", "mean"),
        )
        .sort_values("hour_of_day")
    )

    # Round to 2 decimals like OMIE
    agg["price_main"] = agg["price_main"].round(2)
    agg["price_alt"] = agg["price_alt"].round(2)

    # Add date info
    agg["year"] = date.year
    agg["month"] = date.month
    agg["day"] = date.day

    # period 1..24
    agg["period"] = agg["hour_of_day"] + 1

    # Timestamp = date + hour
    agg["timestamp"] = date + pd.to_timedelta(agg["hour_of_day"], unit="h")

    # For now single zone
    agg["zone"] = "Spain"

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
    return agg[cols]


def get_best_raw_files():
    """
    Scan data/ for marginalpdbc_YYYYMMDD.X files with X in {1,2,...},
    and for each date keep the file with the highest X.

    Returns a list of Paths sorted by date.
    """

    best_by_date = {}

    for path in DATA_DIR.glob("marginalpdbc_*.*"):
        m = re.search(r"marginalpdbc_(\d{8})\.(\d+)$", path.name)
        if not m:
            continue

        date_str = m.group(1)
        version = int(m.group(2))

        # Keep highest version for that date
        if date_str not in best_by_date or version > best_by_date[date_str][0]:
            best_by_date[date_str] = (version, path)

    # Sort by date string
    sorted_dates = sorted(best_by_date.keys())
    return [best_by_date[d][1] for d in sorted_dates]


def build_hourly_from_raw_files() -> pd.DataFrame:
    """
    Loop over all marginalpdbc_YYYYMMDD.X files in data/,
    pick the best version per date (max X), keep only dates >= CHANGE_DATE,
    and build one big hourly DataFrame.
    """

    all_dfs = []

    for path in get_best_raw_files():
        # Check date
        m = re.search(r"marginalpdbc_(\d{8})\.(\d+)$", path.name)
        date_str = m.group(1)
        date = pd.to_datetime(date_str, format="%Y%m%d")

        if date < CHANGE_DATE:
            continue

        df_day = hourly_from_raw_file(path)
        all_dfs.append(df_day)

    if not all_dfs:
        raise RuntimeError(
            f"No OMIE raw files >= {CHANGE_DATE.date()} found in data/"
        )

    df_hourly = pd.concat(all_dfs, ignore_index=True)

    df_hourly = df_hourly.sort_values(
        ["year", "month", "day", "period"]
    ).reset_index(drop=True)

    return df_hourly


# ---------- main logic ----------

def main():
    # 1) Load existing mixed hourly file
    df_all, has_header, orig_cols = load_all_prices()

    df_all["year"] = df_all["year"].astype(int)
    df_all["month"] = df_all["month"].astype(int)
    df_all["day"] = df_all["day"].astype(int)
    df_all["period"] = df_all["period"].astype(int)

    df_all["delivery_date"] = pd.to_datetime(df_all[["year", "month", "day"]])

    # 2) Keep only data BEFORE October 1, 2025
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

    # 3) Build new hourly data from best raw OMIE daily files (>= 2025-10-01)
    df_after = build_hourly_from_raw_files()

    # 4) Combine old hourly + new hourly
    df_final = pd.concat(
        [df_before, df_after],
        ignore_index=True,
    )

    df_final = df_final.sort_values(
        ["year", "month", "day", "period"]
    ).reset_index(drop=True)

    # 5) Restore original column names / structure

    if len(orig_cols) == 8:
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
        df_final = df_final[orig_cols]

    # 6) Overwrite CSV / Parquet / DuckDB

    # CSV – keep header / no-header exactly as before
    df_final.to_csv(ALL_CSV, index=False, header=has_header)
    print(f"Overwritten CSV with clean hourly data: {ALL_CSV}")

    df_final.to_parquet(ALL_PARQUET, index=False)
    print(f"Overwritten Parquet: {ALL_PARQUET}")

    con = duckdb.connect(str(ALL_DUCKDB))
    con.register("prices_hourly_df", df_final)
    con.execute(
        "CREATE OR REPLACE TABLE omie_prices AS "
        "SELECT * FROM prices_hourly_df"
    )
    con.close()
    print(f"Overwritten DuckDB (table omie_prices): {ALL_DUCKDB}")


if __name__ == "__main__":
    main()
