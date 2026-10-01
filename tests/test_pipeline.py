from datetime import date, datetime, timezone
import hashlib
import json
import duckdb
import pytest
from omie_pipeline import expected, parse, build, collect

NOW = datetime(2026,10,1,tzinfo=timezone.utc)

def raw(day, pt=-10, es=20):
    return ('MARGINALPDBC;\n'+''.join(f'{day.year};{day.month};{day.day};{i};{pt};{es};\n' for i in range(1,expected(day)[2]+1))+'*\n').encode()

def seed(root, day, version=1, **prices):
    (root/'data').mkdir(exist_ok=True)
    (root/'data'/f'marginalpdbc_{day:%Y%m%d}.{version}').write_bytes(raw(day, **prices))

@pytest.mark.parametrize('day,n', [(date(2023,10,29),25),(date(2024,3,31),23),(date(2025,10,26),100),(date(2026,3,29),92)])
def test_dst_and_countries(day,n):
    rows=parse(raw(day),f'marginalpdbc_{day:%Y%m%d}.2',NOW,'','').to_pylist()
    assert len(rows)==n*2
    assert len({(r['country'],r['datetime_utc']) for r in rows})==n*2
    assert all(r['datetime_local'][:10]==str(day) for r in rows)
    assert {r['price_eur_mwh'] for r in rows if r['country']=='ES'}=={20}

@pytest.mark.parametrize('invalid', [b'<html>error</html>', b'MARGINALPDBC;\n2026;9;29;1;nan;20;\n*\n', b'MARGINALPDBC;\n2026;9;29;1;10;20;\n*\n'])
def test_bad_file_rejected(invalid):
    with pytest.raises(ValueError):parse(invalid,'marginalpdbc_20260929.1',NOW,'','')

def test_build_revisions_dst_and_idempotency(tmp_path):
    day=date(2025,10,26);seed(tmp_path,day);seed(tmp_path,day,2,es=40)
    m=build(tmp_path)
    assert m['counts']=={'omie_native':200,'omie_hourly':50,'omie_15min':200}
    assert build(tmp_path)==m
    with duckdb.connect(str(tmp_path/'processed/omie_prices.duckdb'),read_only=True) as db:
        assert db.execute('SELECT count(*),min(price_eur_mwh) FROM spain_hourly').fetchone()==(25,40)
        assert db.execute('SELECT count(DISTINCT datetime_utc) FROM omie_hourly').fetchone()[0]==25

def test_missing_day_blocks_publication(tmp_path):
    seed(tmp_path,date(2026,9,27));seed(tmp_path,date(2026,9,29))
    with pytest.raises(ValueError,match='2026-09-28'):build(tmp_path)
    assert not (tmp_path/'processed/manifest.json').exists()

def test_collect_repairs_missing_and_rechecks_existing(tmp_path):
    seed(tmp_path,date(2026,9,27));seed(tmp_path,date(2026,9,29))
    called=[]
    def fetch(name):
        called.append(name)
        day=datetime.strptime(name.split('_')[1].split('.')[0],'%Y%m%d').date()
        if name.endswith('.2'):raise ValueError('not published')
        return raw(day,es=30)
    r=collect(tmp_path,date(2026,9,29),downloader=fetch)
    assert r['new_or_repaired_dates']==['2026-09-28']
    assert len(called)==6
    assert r['published']['delivery_days']==3

def test_unavailable_tomorrow_does_not_replace_good_manifest(tmp_path):
    day=date(2026,9,29);seed(tmp_path,day);m=build(tmp_path)
    def fail(name):raise OSError('offline')
    with pytest.raises(ValueError):collect(tmp_path,date(2026,9,30),downloader=fail)
    assert json.loads((tmp_path/'processed/manifest.json').read_text())==m

def test_corrupt_export_rebuilt(tmp_path):
    seed(tmp_path,date(2026,9,29));build(tmp_path)
    p=tmp_path/'processed/omie_native.parquet';p.write_bytes(b'corrupt')
    m=build(tmp_path)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==m['files'][p.name]['sha256']
