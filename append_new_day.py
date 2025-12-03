import os
import re
from datetime import datetime, date

import duckdb
import pandas as pd
from glob import glob

# Folders
DATA_DIR = "data"
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hourly master DB (same as before)
HOURLY_CSV = os.path.join(OUTPUT_DIR, "all_omie_prices.csv")
HOURLY_PARQUET = os.path.join(OUTPUT_DIR, "all_omie_prices.parquet")
HOURLY_DUCKDB = os.path.join(OUTPUT_DIR, "omie_prices.duckdb")

# New 15-min DB from 2025-10-01 onwards
FIFTEEN_CSV = os.path.join(OUTPUT_DIR, "omie_15min_from_2025-10-01.csv")
FIFTEEN_PARQUET = os.path.join(OUTPUT_DIR, "omie_15min_from_2025-10-01.parquet")
FIFTEEN_DUCKDB = os.path.join(OUTPUT_DIR, "omie_15min_from_2025-10-01.duckdb")

CHANGE_DATE = date(2025, 10, 1)


def parse_daily_file(filepath: str) -> pd.DataFrame:
    """
    Read one daily OMIE file marginalpdbc_YYYYMMDD.1/.2
    using the SAME logic you had in append_new_day.py/build_master_database.py
    (skip first line, sep=';', drop '*', drop last col).
    """
    df = pd.read_csv(filepath, sep=";", skiprows=1, header=None)

    # Drop rows with any '*'
    df = df[~df.apply(lambda x: x.astype(str).str.contains(r"\*").any(), axis=1)]

    if df.shape[1] < 7:
        raise ValueError(f"Unexpected column count in {filepath}: {df.shape[1]}")

    # Drop duplicate / trailing column
    df = df.drop(columns=[6])

    df.columns = ["Year", "Month", "Day", "Period", "Price1", "Price2"]

    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Day"] = df["Day"].astype(int)
    df["Period"] = df["Period"].astype(int)
    df["Price1"] = df["Price1"].astype(float)
    df["Price2"] = df["Price2"].astype(float)

    return df


def extract_date_and_country(filepath: str):
    """
    Get (delivery_date, country) from filename.
    .1 -> Spain, .2 -> Portugal
    """
    name = os.path.basename(filepath)
    m = re.search(r"marginalpdbc_(\d{8})\.(\d+)$", name)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {name}")

    date_str, version = m.groups()
    file_date = datetime.strptime(date_str, "%Y%m%d").date()
    country = "Spain" if name.endswith(".1") else "Portugal"
    return file_date, country


def build_hourly_df(df_quarters: pd.DataFrame, file_date, country: str) -> pd.DataFrame:
    """
    From one daily file (24 or 96 periods) build HOURLY data
    matching the structure of all_omie_prices.*
    """
    max_period = df_quarters["Period"].max()

    if max_period > 24:
        # 96 x 15-min -> average groups of 4
        df = df_quarters.copy()
        df["Hour"] = (df["Period"] - 1) // 4 + 1
        agg = (
            df.groupby("Hour", as_index=False)
            .agg(Price1=("Price1", "mean"), Price2=("Price2", "mean"))
            .sort_values("Hour")
        )
        agg["Price1"] = agg["Price1"].round(2)
        agg["Price2"] = agg["Price2"].round(2)
    else:
        # Already hourly
        agg = df_quarters.copy()
        agg = agg.sort_values("Period").rename(columns={"Period": "Hour"})
        agg = agg[["Hour", "Price1", "Price2"]]

    agg["Year"] = file_date.year
    agg["Month"] = file_date.month
    agg["Day"] = file_date.day

    agg["Datetime"] = pd.to_datetime(
        {"year": agg["Year"], "month": agg["Month"], "day": agg["Day"]}
    ) + pd.to_timedelta(agg["Hour"] - 1, unit="h")

    agg["Country"] = country

    return agg[["Year", "Month", "Day", "Hour", "Price1", "Price2", "Datetime", "Country"]]


def build_15min_df(df_quarters: pd.DataFrame, file_date, country: str) -> pd.DataFrame:
    """
    Build 15-min data for a 96-period file.
    Period 1 -> 00:00, Period 2 -> 00:15, ... Period 96 -> 23:45
    """
    df = df_quarters.copy()
    df["Datetime"] = pd.to_datetime(
        {"year": df["Year"], "month": df["Month"], "day": df["Day"]}
    ) + pd.to_timedelta((df["Period"] - 1) * 15, unit="m")
    df["Country"] = country
    return df[["Year", "Month", "Day", "Period", "Price1", "Price2", "Datetime", "Country"]]


def main():
    # -------- load existing HOURLY DB --------
    if os.path.exists(HOURLY_PARQUET):
        hourly_existing = pd.read_parquet(HOURLY_PARQUET)
        hourly_existing["delivery_date"] = hourly_existing["Datetime"].dt.date
        hourly_pairs = set(
            zip(hourly_existing["delivery_date"], hourly_existing["Country"])
        )
    else:
        hourly_existing = pd.DataFrame(
            columns=["Year", "Month", "Day", "Hour", "Price1", "Price2", "Datetime", "Country"]
        )
        hourly_pairs = set()

    # -------- load existing 15-min DB --------
    if os.path.exists(FIFTEEN_PARQUET):
        q_existing = pd.read_parquet(FIFTEEN_PARQUET)
        q_existing["delivery_date"] = q_existing["Datetime"].dt.date
        q_pairs = set(zip(q_existing["delivery_date"], q_existing["Country"]))
    else:
        q_existing = pd.DataFrame(
            columns=["Year", "Month", "Day", "Period", "Price1", "Price2", "Datetime", "Country"]
        )
        q_pairs = set()

    new_hourly = []
    new_q = []

    # Find all raw files (.1 Spain, .2 Portugal)
    files = sorted(glob(os.path.join(DATA_DIR, "marginalpdbc_*.1"))) + sorted(
        glob(os.path.join(DATA_DIR, "marginalpdbc_*.2"))
    )

    for filepath in files:
        try:
            df_quarters = parse_daily_file(filepath)
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")
            continue

        try:
            file_date, country = extract_date_and_country(filepath)
        except Exception as e:
            print(f"⚠️ Error parsing name {filepath}: {e}")
            continue

        key = (file_date, country)

        # If this (date, country) already in hourly DB, skip
        if key in hourly_pairs:
            print(f"⏭️ Skipping {os.path.basename(filepath)} (already in hourly DB)")
            continue

        print(f"✅ Adding new day: {os.path.basename(filepath)}")

        # ---- Hourly aggregation ----
        h_df = build_hourly_df(df_quarters, file_date, country)
        new_hourly.append(h_df)

        # ---- 15-min series (only for 96-period files from CHANGE_DATE onwards) ----
        if file_date >= CHANGE_DATE and df_quarters["Period"].max() > 24:
            q_df = build_15min_df(df_quarters, file_date, country)
            new_q.append(q_df)

    # -------- merge + save HOURLY --------
    if new_hourly:
        hourly_new = pd.concat(new_hourly, ignore_index=True)
        if not hourly_existing.empty:
            hourly_combined = pd.concat([hourly_existing, hourly_new], ignore_index=True)
        else:
            hourly_combined = hourly_new

        hourly_combined = (
            hourly_combined
            .drop_duplicates(subset=["Datetime", "Country"])
            .sort_values(["Datetime", "Country"])
            .reset_index(drop=True)
        )

        hourly_combined.to_csv(HOURLY_CSV, index=False)
        hourly_combined.to_parquet(HOURLY_PARQUET, index=False)

        con = duckdb.connect(HOURLY_DUCKDB)
        con.execute("DROP TABLE IF EXISTS prices")
        con.register("df_hourly", hourly_combined)
        con.execute("CREATE TABLE prices AS SELECT * FROM df_hourly")
        con.close()

        print(f"💾 Updated hourly DB: {HOURLY_CSV}, {HOURLY_PARQUET}, {HOURLY_DUCKDB}")
    else:
        print("🟰 No new hourly days to add.")

    # -------- merge + save 15-min --------
    if new_q:
        q_new = pd.concat(new_q, ignore_index=True)
        if not q_existing.empty:
            q_combined = pd.concat([q_existing, q_new], ignore_index=True)
        else:
            q_combined = q_new

        q_combined = (
            q_combined
            .drop_duplicates(subset=["Datetime", "Country"])
            .sort_values(["Datetime", "Country"])
            .reset_index(drop=True)
        )

        q_combined.to_csv(FIFTEEN_CSV, index=False)
        q_combined.to_parquet(FIFTEEN_PARQUET, index=False)

        con = duckdb.connect(FIFTEEN_DUCKDB)
        con.execute("DROP TABLE IF EXISTS prices_15min")
        con.register("df_q", q_combined)
        con.execute("CREATE TABLE prices_15min AS SELECT * FROM df_q")
        con.close()

        print(f"💾 Updated 15-min DB: {FIFTEEN_CSV}, {FIFTEEN_PARQUET}, {FIFTEEN_DUCKDB}")
    else:
        print("🟰 No new 15-min days to add.")


if __name__ == "__main__":
    main()
