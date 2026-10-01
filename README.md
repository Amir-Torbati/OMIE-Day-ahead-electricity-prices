# OMIE day-ahead electricity prices

This repository owns OMIE downloading, gap repair, validation and publication of clean price tables. [ree-data-platform](https://github.com/Amir-Torbati/ree-data-platform) consumes these outputs alongside REE data; it does not download or repair OMIE prices.

## Validated outputs (schema version 2)

| File in `processed/` | Contents |
| --- | --- |
| `omie_native.parquet` | Original hourly intervals before 1 October 2025 and 15-minute intervals afterward |
| `omie_hourly.parquet` | Full hourly curve; four quarter-hours averaged after the transition |
| `omie_15min.parquet` | Original quarter-hour prices from 1 October 2025 |
| `omie_prices.duckdb` | All three tables plus `spain_native`, `spain_hourly` and `spain_15min` views |
| `manifest.json` | Coverage, counts, source-file hashes and output checksums; published last |

Use **`spain_hourly`** or **`spain_15min`** for Spanish-market analysis. Parquet rows use `country='ES'` for Spain and `country='PT'` for Portugal. Portugal is retained separately for optional comparison. No price is substituted between countries.

Native columns: `country`, `delivery_date`, `period`, `resolution_minutes`, `datetime_utc`, `datetime_local`, `price_eur_mwh`, `source_version`, `source_file`, `source_sha256`. The UTC timestamp is the unique time key. Local timestamps include the offset, so both occurrences of the repeated autumn hour remain distinct.

The first published v2 archive covers 1 April 2023 through 1 October 2026: 1,280 days, 114,144 native observations, 61,440 hourly observations and 70,272 quarter-hour observations across both countries. Each country has half these counts. Later runs extend the end date; read the manifest for current coverage.

## What changed

The 22 previously missing daily files have been recovered from OMIE and committed here. Complete tables are rebuilt from validated raw files, including historical quarter-hours absent from earlier processed tables. Country columns and clock-change handling are corrected using the [official OMIE format specification, page 13](https://www.omie.es/sites/default/files/2025-09/formato_ficheros_inf_pub_137_1.pdf): the first price is Portugal, the second Spain, and `.1`/`.2` are revision numbers.

**Schema migration:** previous CSV files, duplicate quarter-hour directory and separate historical databases have been retired. They remain recoverable in Git history. Existing queries using `Price1`, `Price2`, `Datetime`, `Country`, `prices` or `omie_prices` must migrate to the explicit v2 tables/columns above. The retained `omie_prices.duckdb` filename now contains v2 tables, not the legacy schema. Old script entry points delegate to the new pipeline and cannot recreate stale tables.

## Run and validate

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python omie_pipeline.py collect
python omie_pipeline.py build
```

`collect` requests missing dates and refreshes the latest three delivery dates through tomorrow in **Europe/Madrid**. It probes revisions 1 and 2 even when a file already exists; use `--max-version N` for an explicit wider scan (up to 10). `--end YYYY-MM-DD` sets the required end date. `build` is offline and validates all dates between the first and last source files. Unchanged source content and intact outputs produce no new Git commit.

Files must have the expected header, terminator, delivery date, finite prices and every unique period. Normal days have 24 hourly or 96 quarter-hour periods; clock-change days have 23/25 or 92/100. Zero and negative prices are valid. A missing required day or corrupt file blocks publication. HTTP requests use bounded retries and timeouts. Existing committed prices remain available if an update fails.

## Single scheduled collector

Only `download_prices.yml` collects prices, at **13:17, 17:17 and 21:17 UTC** daily. These are three attempts at collecting/rechecking the next delivery day. `OMIE_COLLECTION_ENABLED=true` enables scheduling; manual runs also work. Former overlapping processing, rebuilding and splitting workflows are retired. `validate.yml` only tests code and the archive.

One concurrency group prevents overlapping publishers. A failed validation or Git push fails visibly. The new platform imports this repository's validated publication once overnight; no cross-repository write token is needed.

Raw files and current clean outputs are committed here and **do not expire like Actions artifacts**. Daily binary snapshots grow Git history; migrate bulk data to durable object storage before substantially expanding scope. GitHub schedules are best effort, not a guarantee of immediate publication delivery. Scheduled version probing does not certify the latest possible historical revision of every date.
