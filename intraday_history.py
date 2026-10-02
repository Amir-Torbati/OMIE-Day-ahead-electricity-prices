"""Manual, resumable monthly releases; no recurring workflow added."""
from datetime import date,datetime,timedelta
import argparse,hashlib,json,os,subprocess,zipfile
from pathlib import Path
from intraday_pipeline import collect,MADRID

REPO='Amir-Torbati/OMIE-Day-ahead-electricity-prices'
def gh(args,check=True):return subprocess.run(['gh',*args],capture_output=True,text=True,check=check)

def run(year):
    if os.environ.get('GITHUB_REPOSITORY')!=REPO:raise ValueError('Wrong history repository')
    today=datetime.now(MADRID).date();summary=[]
    for month in range(1,13):
        start=date(year,month,1)
        if start>=today:break
        end=min(date(year+1,1,1) if month==12 else date(year,month+1,1),today)-timedelta(days=1)
        tag=f'intraday-{start:%Y-%m}';old=gh(['api',f'repos/{REPO}/releases/tags/{tag}'],False)
        if old.returncode and 'HTTP 404' not in old.stderr:raise RuntimeError('Cannot read history checkpoint')
        if not old.returncode:
            desc=json.loads(old.stdout)['body']
            try:cp=json.loads(desc)
            except ValueError:cp={}
            if cp.get('end')==str(end) and cp.get('status')=='completed_with_explicit_source_availability':
                summary.append({'month':tag,'status':'already_downloaded'});continue
        root=Path('history')/f'{start:%Y-%m}';report=collect(root,start,end)
        archive=Path('history')/f'{tag}-{os.environ["GITHUB_RUN_ID"]}-{os.environ.get("GITHUB_RUN_ATTEMPT","1")}.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(root).as_posix())
        sha=hashlib.sha256(archive.read_bytes()).hexdigest();sha_path=archive.with_suffix('.zip.sha256');sha_path.write_text(sha+'  '+archive.name+'\n')
        if old.returncode:gh(['release','create',tag,'--repo',REPO,'--target',os.environ['GITHUB_SHA'],'--latest=false','--title',tag,'--notes','Collection in progress'])
        gh(['release','upload',tag,str(archive),str(sha_path),'--repo',REPO])
        cp={'end':str(end),'status':report['status'],'asset':archive.name,'sha256':sha,
            'rows':json.loads((root/'manifest.json').read_text())['rows']}
        notes=root/'release-notes.json';notes.write_text(json.dumps(cp));gh(['release','edit',tag,'--repo',REPO,'--notes-file',str(notes)])
        summary.append({'month':tag,**cp})
    Path('history/summary.json').write_text(json.dumps(summary,indent=2))
    if any(x.get('status')=='request_failures' for x in summary):raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,choices=range(2023,2027),required=True);run(p.parse_args().year)
