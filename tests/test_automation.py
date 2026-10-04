from datetime import datetime,timezone
from types import SimpleNamespace
import pytest
import github_release
from intraday_pipeline import health

def test_freshness_is_not_inferred_from_successful_requests():
    report={'start':'2026-10-01','end':'2026-10-05','status':'completed_with_explicit_source_availability'}
    m={'sessions':[{'file':'marginalpibc_2026100101.1','rows':192,'internal_missing_periods':0}]}
    h=health(m,report,datetime(2026,10,4,12,tzinfo=timezone.utc))
    assert h['execution']=='success' and h['freshness']=='stale'
    assert '2026-10-03:3' in h['missing_closed_date_sessions']

def test_latest_empty_file_does_not_hide_staleness():
    m={'sessions':[{'file':'marginalpibc_2026100401.1','rows':0,'internal_missing_periods':0}]}
    h=health(m,{'start':'2026-10-04','end':'2026-10-04','status':'completed_with_explicit_source_availability'},datetime(2026,10,4,12,tzinfo=timezone.utc))
    assert h['freshness']=='stale' and h['completeness']=='partial_or_unconfirmed'

def test_github_transient_retry_and_redacted_failure(monkeypatch):
    monkeypatch.setattr(github_release.time,'sleep',lambda _:None)
    responses=iter([SimpleNamespace(returncode=1,stderr='HTTP 503',stdout=''),SimpleNamespace(returncode=0,stderr='',stdout='ok')])
    monkeypatch.setattr(github_release.subprocess,'run',lambda *a,**k:next(responses))
    assert github_release.gh(['release','view','test']).stdout=='ok'
    monkeypatch.setenv('GH_TOKEN','secret-fixture')
    monkeypatch.setattr(github_release.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stderr='HTTP 403 forbidden secret-fixture',stdout=''))
    with pytest.raises(RuntimeError,match='REDACTED') as error:github_release.gh(['api','test'])
    assert 'secret-fixture' not in str(error.value)
