"""OMIE auction prices; distinct sessions, native periods and retained revisions.

Format source: OMIE public information file model 1.38, section 5.2.1.1.
Coverage never assumes that every session covers an entire delivery day.
"""
import argparse
from datetime import date,datetime,timedelta,time,timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time as clock
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
from omie_pipeline import atomic_write,json_data,FileNotPublished

MADRID=ZoneInfo('Europe/Madrid');UTC=timezone.utc
FILE=re.compile(r'marginalpibc_(\d{8})(\d{2})\.(\d+)')
SCHEMA=pa.schema([
 ('auction_file_date',pa.date32()),('session',pa.int16()),('country',pa.string()),
 ('delivery_date',pa.date32()),('period',pa.int16()),('resolution_minutes',pa.int16()),
 ('datetime_utc',pa.timestamp('us',tz='UTC')),('price_eur_mwh',pa.float64()),
 ('source_file',pa.string()),('source_version',pa.int16()),('source_sha256',pa.string())])


def parse(raw,filename):
    m=FILE.fullmatch(filename)
    if not m:raise ValueError('Unknown intraday filename')
    file_day=datetime.strptime(m[1],'%Y%m%d').date();session=int(m[2]);version=int(m[3])
    if session not in range(1,7):raise ValueError('Unknown auction session')
    lines=[l.strip() for l in raw.decode('utf-8-sig').splitlines() if l.strip()]
    if not lines or lines[0]!='MARGINALPIBC;' or lines[-1]!='*':raise ValueError('Missing intraday header/terminator')
    records=[];seen=set();sha=hashlib.sha256(raw).hexdigest()
    for line in lines[1:-1]:
        f=line.split(';')
        if len(f)!=7 or f[-1]!='':raise ValueError('Intraday field count changed')
        y,m,d,p=map(int,f[:4]);day=date(y,m,d)
        if abs((day-file_day).days)>1:raise ValueError('Delivery date outside auction horizon')
        # Quarter-hour delivery starts 19 March; 18 March auctions prepare it.
        minutes=15 if day>=date(2025,3,19) else 60
        a=datetime.combine(day,time(),MADRID).astimezone(UTC)
        b=datetime.combine(day+timedelta(days=1),time(),MADRID).astimezone(UTC)
        n=int((b-a).total_seconds()/(minutes*60))
        if not 1<=p<=n or (day,p) in seen:raise ValueError('Invalid or duplicate intraday period')
        seen.add((day,p))
        for country,text in zip(('PT','ES'),f[4:6]):
            value=float(text)
            if not math.isfinite(value):raise ValueError('Non-finite intraday price')
            records.append(dict(auction_file_date=file_day,session=session,country=country,delivery_date=day,
                period=p,resolution_minutes=minutes,datetime_utc=a+timedelta(minutes=(p-1)*minutes),
                price_eur_mwh=value,source_file=filename,source_version=version,source_sha256=sha))
    by_day={}
    for day,p in seen:by_day.setdefault(day,[]).append(p)
    gaps=sum(max(ps)-min(ps)+1-len(ps) for ps in by_day.values())
    qa={'rows':len(records),'internal_missing_periods':gaps,'status':'empty_source_file' if not records else 'internal_gaps' if gaps else 'validated_published_horizon',
        'horizons':{str(d):{'first_period':min(ps),'last_period':max(ps),'periods':len(ps)} for d,ps in sorted(by_day.items())},
        'scope':'Published horizon only; session-specific expected horizon and cancelled auctions not independently certified'}
    return pa.Table.from_pylist(sorted(records,key=lambda r:(r['datetime_utc'],r['country'])),schema=SCHEMA),qa


def fetch(name):
    clock.sleep(.2)
    url='https://www.omie.es/es/file-download?'+urlencode({'parents':'marginalpibc','filename':name})
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'Energy-data-collection/1'}),timeout=30) as r:raw=r.read(2_000_001)
            if len(raw)>2_000_000:raise ValueError('Oversize intraday source')
            parse(raw,name);return raw
        except HTTPError as e:
            if e.code==404:raise FileNotPublished(name) from None
            if e.code not in (429,500,502,503,504) or attempt==2:raise
        except (OSError,TimeoutError):
            if attempt==2:raise
        clock.sleep(2**attempt)


def collect(root,start,end,downloader=fetch,max_version=2):
    root=Path(root);raw_root=root/'raw';raw_root.mkdir(parents=True,exist_ok=True)
    index_path=root/'index.json';index=json.loads(index_path.read_text()) if index_path.exists() else {}
    attempts=[];changed=0
    for offset in range((end-start).days+1):
        day=start+timedelta(days=offset)
        # Transition day includes the final regional sessions and first IDA.
        sessions=range(1,7) if day<=date(2024,6,13) else range(1,4)
        for session in sessions:
            for version in range(1,max_version+1):
                name=f'marginalpibc_{day:%Y%m%d}{session:02d}.{version}'
                try:raw=downloader(name)
                except FileNotPublished:
                    attempts.append({'file':name,'status':'not_published_404'});continue
                except Exception as e:
                    attempts.append({'file':name,'status':'failed','error_type':type(e).__name__});continue
                table,qa=parse(raw,name);sha=hashlib.sha256(raw).hexdigest()
                key=f'raw/{name}.{sha}.gz'
                if not (root/key).exists():atomic_write(root/key,gzip.compress(raw,mtime=0));changed+=1
                if index.get(name,{}).get('sha256')!=sha:
                    index[name]={'key':key,'sha256':sha,'retrieved_at':datetime.now(UTC).isoformat(),'qa':qa}
                attempts.append({'file':name,**qa})
            atomic_write(index_path,json_data(index))
    report={'start':str(start),'end':str(end),'requests':len(attempts),'changed_source_files':changed,'attempts':attempts,
        'status':'request_failures' if any(a['status']=='failed' for a in attempts) else 'completed_with_explicit_source_availability'}
    manifest=build(root,index)
    report['health']=health(manifest,report)
    atomic_write(root/'last-run.json',json_data(report))
    print(json.dumps({k:v for k,v in report.items() if k!='attempts'}))
    return report


def health(manifest,report,now=None):
    now=now or datetime.now(UTC)
    cutoff=now.astimezone(MADRID).date()-timedelta(days=1)
    actual=[datetime.strptime(FILE.fullmatch(s['file'])[1],'%Y%m%d').date()
            for s in manifest['sessions'] if s['rows'] and not s['internal_missing_periods']]
    latest=max(actual) if actual else None
    invalid=[s['file'] for s in manifest['sessions'] if s['internal_missing_periods']]
    empty=[s['file'] for s in manifest['sessions'] if not s['rows']]
    missing=[]
    first=date.fromisoformat(report['start']);last=min(date.fromisoformat(report['end']),cutoff)
    known={(FILE.fullmatch(s['file'])[1],int(FILE.fullmatch(s['file'])[2])) for s in manifest['sessions']}
    for offset in range(max(0,(last-first).days+1)):
        day=first+timedelta(days=offset)
        for session in range(1,7 if day<=date(2024,6,13) else 4):
            if (f'{day:%Y%m%d}',session) not in known:missing.append(f'{day}:{session}')
    return {'checked_at':now.isoformat(),'execution':'failed' if report['status']=='request_failures' else 'success',
        'freshness':'current' if latest and latest>=cutoff else 'stale',
        'required_auction_file_date':str(cutoff),'latest_auction_file_date':str(latest) if latest else None,
        'completeness':'partial_or_unconfirmed' if missing or empty or invalid else 'published_horizons_validated',
        'missing_closed_date_sessions':missing,'empty_source_files':empty,'internal_gap_files':invalid,
        'policy':'Freshness requires an auction file dated yesterday or later. This is our operational allowance, not a verified publisher deadline; expected auction horizons/cancellations remain unconfirmed.'}


def build(root,index=None):
    root=Path(root);index=index or json.loads((root/'index.json').read_text());latest={}
    for name,entry in index.items():
        m=FILE.fullmatch(name)
        if not m:raise ValueError('Invalid indexed filename')
        key=(m[1],m[2]);version=int(m[3])
        if key not in latest or version>latest[key][0]:latest[key]=(version,name,entry)
    tables=[];coverage=[]
    for _,name,e in sorted(latest.values(),key=lambda v:v[1]):
        path=(root/e['key']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Unsafe intraday source path')
        raw=gzip.decompress(path.read_bytes())
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('Intraday raw checksum mismatch')
        table,qa=parse(raw,name);coverage.append({'file':name,**qa})
        if not qa['internal_missing_periods']:tables.append(table)
    if tables:
        table=pa.concat_tables(tables).sort_by([('auction_file_date','ascending'),('session','ascending'),('datetime_utc','ascending'),('country','ascending')])
    else:table=pa.Table.from_pylist([],schema=SCHEMA)
    buf=io.BytesIO();pq.write_table(table,buf,compression='zstd');atomic_write(root/'intraday_prices.parquet',buf.getvalue())
    manifest={'schema_version':1,'market':'OMIE intraday auctions','rows':table.num_rows,
        'parquet_sha256':hashlib.sha256(buf.getvalue()).hexdigest(),'sessions':coverage,
        'source_format':'https://www.omie.es/sites/default/files/2026-09/formato_ficheros_inf_pub_138_1.pdf',
        'revision_policy':'Highest available file revision; same-name byte revisions retained by SHA; empty latest revisions remove older prices'}
    atomic_write(root/'manifest.json',json_data(manifest));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='intraday');p.add_argument('--start',type=date.fromisoformat);p.add_argument('--end',type=date.fromisoformat)
    args=p.parse_args();today=datetime.now(MADRID).date();end=args.end or today+timedelta(days=1)
    report=collect(args.root,args.start or end-timedelta(days=3),end)
    h=report['health']
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
            f.write(f"\n## Intraday health\nExecution: {h['execution']}; freshness: {h['freshness']}; coverage: {h['completeness']}.\n")
            f.write(f"Latest auction file: {h['latest_auction_file_date']}; required: {h['required_auction_file_date']}. Missing closed-date sessions: {len(h['missing_closed_date_sessions'])}; source-empty files: {len(h['empty_source_files'])}.\n")
    raise SystemExit(1 if h['execution']=='failed' or h['freshness']=='stale' or h['internal_gap_files'] else 0)
