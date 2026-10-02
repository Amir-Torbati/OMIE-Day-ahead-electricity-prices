from datetime import date
import json
import pytest
from intraday_pipeline import parse,collect,build
from omie_pipeline import FileNotPublished

def raw(day,periods):
    return ('MARGINALPIBC;\n'+''.join(f'{day.year};{day.month};{day.day};{p};1.23;-2.34;\n' for p in periods)+'*\n').encode()

def test_country_order_partial_session_and_dst():
    d=date(2026,10,25);t,qa=parse(raw(d,range(53,101)),'marginalpibc_2026102503.1')
    assert t.num_rows==96 and qa['internal_missing_periods']==0
    assert {r['price_eur_mwh'] for r in t.to_pylist() if r['country']=='ES'}=={-2.34}
    assert t['datetime_utc'][0].as_py().hour==11

def test_transition_and_corrupt_body():
    for d,n in [(date(2025,3,18),24),(date(2025,3,19),96)]:
        t,qa=parse(raw(d,range(1,n+1)),f'marginalpibc_{d:%Y%m%d}02.1')
        assert t.num_rows==n*2 and qa['status']=='validated_published_horizon'
    with pytest.raises(ValueError):parse(b'<html>error</html>','marginalpibc_2026100101.1')
    with pytest.raises(ValueError):parse(raw(date(2026,10,1),[1,1]),'marginalpibc_2026100101.1')

def test_empty_latest_revision_removes_prices_and_same_name_history_remains(tmp_path):
    d=date(2026,10,1)
    def first(n):
        if n.endswith('01.1'):return raw(d,range(1,97))
        raise FileNotPublished(n)
    collect(tmp_path,d,d,first)
    assert build(tmp_path)['rows']==192
    def corrected(n):
        if n.endswith('01.1'):return b'MARGINALPIBC;\n*\n'
        raise FileNotPublished(n)
    collect(tmp_path,d,d,corrected)
    assert build(tmp_path)['rows']==0
    assert len(list((tmp_path/'raw').glob('*.gz')))==2
