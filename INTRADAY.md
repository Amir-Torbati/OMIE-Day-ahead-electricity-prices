# Intraday auction collection

The original OMIE repository remains the source collector. Day-ahead and
intraday auction prices are distinct datasets. Each intraday session, delivery
period, country and file revision is retained separately; no session averaging.

Official specification: [OMIE file model 1.38, section 5.2.1.1](https://www.omie.es/sites/default/files/2026-09/formato_ficheros_inf_pub_138_1.pdf).
The fields contain **Portugal then Spain**. A filename suffix is a revision.
Native intraday delivery is hourly through 18 March 2025, then quarter-hourly.
The collector accepts the published horizon and flags internal gaps; it does
not certify that a truncated edge or empty auction file is a cancelled session.
The 2024 transition between six regional auctions and three IDAs is retained.

## Schedules and storage

The existing collection workflow now runs at **13:23, 17:23, 21:23 and 23:23
Europe/Madrid**, four scheduled opportunities per day. The added late check
allows capture after the 22:00 IDA auction. These are buffered polling times,
not promises of source publication or GitHub start times. Recent file dates
through tomorrow are checked, with revisions 1 and 2. A 404 is logged as
unavailable, not a zero price. Revision discovery beyond version 2 remains a
limitation. Continuous intraday trades and orders are outside this collector.

`intraday/raw/` holds gzip responses identified by filename and SHA256.
`intraday/index.json` points to the most recently retrieved bytes for each
revision. Older bytes remain. `intraday/intraday_prices.parquet` contains the
highest available revision per auction file date/session, with empty latest
revisions removing superseded observations. `manifest.json` records hashes,
row counts and published horizons. These files are committed in this repository.

The manual **Backfill intraday auction history** workflow stores monthly ZIPs
and SHA256 files in `intraday-YYYY-MM` releases. It checkpoints each month and
uses two year jobs at most. No additional recurring workflow is created.
Release data is durable while retained in the repository; summary artifacts
expire after seven days. Archive backfill is a one-off use of runner time.

REE consumes validated producer outputs; it must not recollect OMIE source data.
