"""Authoritative OMIE collector and validated price-table publisher."""
import argparse
import collections
from datetime import date, datetime, time, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import zipfile
from zoneinfo import ZoneInfo
import pyarrow as pa
import pyarrow.parquet as pq

MADRID = ZoneInfo('Europe/Madrid')
UTC = timezone.utc
CHANGE_DATE = date(2025, 10, 1)
FILE = re.compile(r'marginalpdbc_(\d{8})\.(\d+)$')
SCHEMA = pa.schema([
    ('country', pa.string()), ('delivery_date', pa.date32()), ('period', pa.int32()),
    ('resolution_minutes', pa.int32()), ('datetime_utc', pa.timestamp('us', tz='UTC')),
    ('datetime_local', pa.string()), ('price_eur_mwh', pa.float64()),
    ('source_version', pa.int32()), ('source_file', pa.string()), ('source_sha256', pa.string()),
    ('source_commit', pa.string()), ('retrieved_at', pa.timestamp('us', tz='UTC')), ('batch_id', pa.string()),
])


def expected(day):
    minutes = 15 if day >= CHANGE_DATE else 60
    a = datetime.combine(day, time(), MADRID).astimezone(UTC)
    b = datetime.combine(day + timedelta(days=1), time(), MADRID).astimezone(UTC)
    return minutes, a, int((b-a).total_seconds() / (minutes*60))


def parse(raw, filename, captured, batch_id, commit):
    match = FILE.fullmatch(Path(filename).name)
    if not match:
        raise ValueError('Unrecognised OMIE filename')
    day, version = datetime.strptime(match[1], '%Y%m%d').date(), int(match[2])
    lines = [line.strip() for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    if not lines or lines[0] != 'MARGINALPDBC;' or lines[-1] != '*':
        raise ValueError('Missing OMIE header/terminator; possible HTML error or partial file')
    minutes, midnight, n = expected(day)
    periods, rows = set(), []
    sha = hashlib.sha256(raw).hexdigest()
    for line in lines[1:-1]:
        fields = line.split(';')
        if len(fields) != 7 or fields[-1] != '':
            raise ValueError('Unexpected OMIE field count')
        y, m, d, period = map(int, fields[:4])
        if date(y,m,d) != day or period not in range(1,n+1) or period in periods:
            raise ValueError('Wrong delivery date, duplicate or invalid period')
        periods.add(period)
        ts = midnight + timedelta(minutes=(period-1)*minutes)
        # Official format: Portugal first, Spain second; suffix is a revision, not country.
        for country, text in zip(('PT','ES'), fields[4:6]):
            value = float(text)
            if not math.isfinite(value):
                raise ValueError('Missing or non-finite price')
            rows.append(dict(country=country, delivery_date=day, period=period,
                resolution_minutes=minutes, datetime_utc=ts, datetime_local=ts.astimezone(MADRID).isoformat(),
                price_eur_mwh=value, source_version=version, source_file=Path(filename).name,
                source_sha256=sha, source_commit=commit, retrieved_at=captured, batch_id=batch_id))
    if periods != set(range(1,n+1)):
        raise ValueError(f'Incomplete day: expected {n} unique periods, found {len(periods)}')
    return pa.Table.from_pylist(sorted(rows,key=lambda r:(r['datetime_utc'],r['country'])),schema=SCHEMA)


NATIVE_COLUMNS = [f.name for f in SCHEMA if f.name not in ('retrieved_at', 'batch_id', 'source_commit')]
CONTRACT_VERSION = 2


def atomic_write(path, data):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def json_data(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()


def raw_files(root):
    grouped = collections.defaultdict(list)
    for path in (Path(root) / 'data').glob('marginalpdbc_*'):
        match = FILE.fullmatch(path.name)
        if match:
            grouped[datetime.strptime(match[1], '%Y%m%d').date()].append((int(match[2]), path))
    if not grouped:
        raise ValueError('No raw source files')
    return grouped


class FileNotPublished(OSError):
    """Official endpoint explicitly returned HTTP 404 for a requested revision."""


def fetch(filename):
    """Bounded retries and timeout; reject invalid bodies before storing any file."""
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    import time as clock
    url = 'https://www.omie.es/es/file-download?' + urlencode({'parents': 'marginalpdbc', 'filename': filename})
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'OMIE-price-archive/2'}), timeout=25) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError('Unexpectedly large source response')
            parse(raw, filename, datetime.now(UTC), 'validation', '')
            return raw
        except HTTPError as exc:
            if exc.code == 404:
                exc.close()
                raise FileNotPublished(filename) from None
            if attempt == 2:
                raise
            clock.sleep(attempt + 1)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            clock.sleep(attempt + 1)


def collect(root, end, refresh_days=3, max_version=2, downloader=fetch, allow_pending_end=False, start=None):
    root = Path(root)
    grouped = raw_files(root)
    first = min(min(grouped),start) if start is not None else min(grouped)
    if end < first:
        raise ValueError('End date precedes archive')
    targets = [first + timedelta(days=i) for i in range((end-first).days+1)
               if first + timedelta(days=i) not in grouped or
               first + timedelta(days=i) >= end - timedelta(days=refresh_days-1)]
    changed, errors, repaired = 0, [], []
    unpublished = set()
    for day in targets:
        available = False
        not_published = 0
        for version in range(1, max_version+1):
            name = f'marginalpdbc_{day:%Y%m%d}.{version}'
            try:
                raw = downloader(name)
                parse(raw, name, datetime.now(UTC), 'validation', '')
                path = root / 'data' / name
                if not path.exists() or path.read_bytes() != raw:
                    atomic_write(path, raw)
                    changed += 1
                available = True
            except (OSError, ValueError) as exc:
                errors.append({'file': name, 'error': type(exc).__name__})
                if isinstance(exc, FileNotPublished):not_published += 1
        if not_published == max_version:unpublished.add(day)
        if day not in grouped and available:
            repaired.append(str(day))
    # build() checks completeness, so an unavailable required new date fails the run.
    try:
        if start is not None and min(raw_files(root))>start:
            raise ValueError('Requested historical start is unavailable')
        pending = allow_pending_end and end in unpublished and end not in raw_files(root)
        report = build(root, end-timedelta(days=1) if pending else end)
    except ValueError:
        print(json.dumps({'required_end':str(end),'changed_files':changed,
                          'version_attempt_errors':errors},indent=2), flush=True)
        raise
    return {'status':'publication_pending' if pending else 'complete',
            'pending_delivery_date':str(end) if pending else None,
            'changed_files': changed, 'new_or_repaired_dates': repaired,
            'version_attempt_errors': errors, 'published': report}


def build(root, end=None):
    """Publish complete latest-version tables; the manifest is the commit marker."""
    import duckdb
    import tempfile
    root = Path(root)
    grouped = raw_files(root)
    first, last = min(grouped), max(grouped)
    required_end = max(last, end or last)
    missing = [str(first+timedelta(days=i)) for i in range((required_end-first).days+1)
               if first+timedelta(days=i) not in grouped]
    if missing:
        raise ValueError('Missing required delivery dates: ' + ', '.join(missing))
    tables, hashes = [], []
    for day, versions in sorted(grouped.items()):
        _, path = max(versions)
        raw = path.read_bytes()
        tables.append(parse(raw, path.name, datetime(2000,1,1,tzinfo=UTC), '', '').select(NATIVE_COLUMNS))
        hashes.append({'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest()})
    fingerprint = hashlib.sha256(json_data({'contract': CONTRACT_VERSION, 'sources': hashes})).hexdigest()
    output = root / 'processed'
    output.mkdir(exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('source_fingerprint') == fingerprint and all(
            (output/name).is_file() and hashlib.sha256((output/name).read_bytes()).hexdigest() == info['sha256']
            for name, info in previous.get('files', {}).items()
        ) and len(previous.get('files', {})) == 4:
            return previous
    with tempfile.TemporaryDirectory(prefix='omie-build-') as temp:
        stage = Path(temp)
        with duckdb.connect(str(stage/'omie_prices.duckdb')) as db:
            db.execute("SET TimeZone='UTC'")
            db.register('incoming', pa.concat_tables(tables))
            db.execute('CREATE TABLE omie_native AS SELECT * FROM incoming ORDER BY datetime_utc,country')
            duplicate = db.execute('SELECT count(*) FROM (SELECT country,datetime_utc,count(*) n FROM omie_native GROUP BY ALL HAVING n<>1)').fetchone()[0]
            if duplicate:
                raise ValueError('Duplicate native keys')
            invalid = db.execute("SELECT count(*) FROM (SELECT country,date_trunc('hour',datetime_utc) h,sum(resolution_minutes) duration FROM omie_native GROUP BY country,h HAVING duration<>60)").fetchone()[0]
            if invalid:
                raise ValueError('Incomplete hourly groups')
            db.execute("""CREATE TABLE omie_hourly AS SELECT country,
                date_trunc('hour',datetime_utc) AS datetime_utc,
                min(delivery_date) AS delivery_date, arg_min(datetime_local,datetime_utc) AS datetime_local,
                round(avg(price_eur_mwh),4) AS price_eur_mwh,
                count(*) AS native_periods, min(resolution_minutes) AS native_resolution_minutes
                FROM omie_native GROUP BY country,date_trunc('hour',datetime_utc)
                ORDER BY datetime_utc,country""")
            db.execute('CREATE TABLE omie_15min AS SELECT * FROM omie_native WHERE resolution_minutes=15 ORDER BY datetime_utc,country')
            for table in ('omie_native', 'omie_hourly', 'omie_15min'):
                db.execute(f"CREATE VIEW spain_{table.removeprefix('omie_')} AS SELECT * FROM {table} WHERE country='ES'")
                pq.write_table(db.execute(f'SELECT * FROM {table} ORDER BY datetime_utc,country').to_arrow_table(), stage/(table+'.parquet'), compression='zstd')
            counts = {t: db.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('omie_native','omie_hourly','omie_15min')}
        files = {}
        for name in ('omie_native.parquet','omie_hourly.parquet','omie_15min.parquet','omie_prices.duckdb'):
            data = (stage/name).read_bytes()
            files[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            atomic_write(output/name, data)
    manifest = {'contract_version': CONTRACT_VERSION, 'status': 'complete',
        'source_fingerprint': fingerprint, 'first_day': str(first), 'last_day': str(last),
        'delivery_days': len(grouped), 'countries': ['ES','PT'], 'unit': 'EUR/MWh',
        'timezone': 'Europe/Madrid', 'counts': counts, 'files': files,
        'qa': {'missing_days': [], 'duplicate_keys': 0, 'null_prices': 0},
        'revision_policy': 'Highest version present; scheduled refresh probes .1 and .2 for latest 3 delivery dates',
        'sources': hashes}
    atomic_write(manifest_path, json_data(manifest))
    return manifest


def delivery_target(now):
    """Before the afternoon collection window, require today, not unreleased tomorrow.

    13:23 Madrid is our first check, not a guaranteed OMIE publication deadline.
    Explicit --end remains available for backfills and strict operator requests.
    """
    local = now.astimezone(MADRID)
    return local.date() + timedelta(days=int((local.hour, local.minute) >= (13, 23)))


def publication_pending_allowed(now, target, explicit_end=False):
    """Only tomorrow may be pending before our final 23:23 Madrid check."""
    local = now.astimezone(MADRID)
    return (not explicit_end and target == local.date() + timedelta(days=1)
            and (local.hour, local.minute) < (23, 23))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('collect','build'))
    p.add_argument('--root', default='.')
    p.add_argument('--end', type=date.fromisoformat)
    p.add_argument('--start', type=date.fromisoformat,help='Extend the required historical start; never truncates existing data')
    p.add_argument('--refresh-days', type=int, default=3)
    p.add_argument('--max-version', type=int, choices=range(1,11), default=2)
    a = p.parse_args()
    if a.refresh_days < 1:
        p.error('--refresh-days must be positive')
    if a.command == 'collect':
        now = datetime.now(UTC)
        target = a.end or delivery_target(now)
        allow_pending = publication_pending_allowed(now, target, explicit_end=a.end is not None)
        result = collect(a.root, target, a.refresh_days, a.max_version, allow_pending_end=allow_pending,start=a.start)
        result['required_delivery_end'] = str(target)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
                f.write(f"OMIE: **{result['status']}**. Requested through {target}; validated through {result['published']['last_day']}. Changed source files: {result['changed_files']}.\n")
    else:
        result = build(a.root, a.end)
    # The full source hash inventory is in processed/manifest.json.
    compact = result.copy()
    if 'published' in compact:
        compact['published'] = {k:v for k,v in compact['published'].items() if k != 'sources'}
    else:
        compact.pop('sources', None)
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
