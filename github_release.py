"""Bounded GitHub CLI retries with useful, credential-redacted diagnostics."""
import os
import subprocess
import time

def gh(args, check=True):
    result=None
    for attempt in range(4):
        try:
            result=subprocess.run(['gh',*args],capture_output=True,text=True,timeout=180)
        except subprocess.TimeoutExpired:
            if attempt==3:raise RuntimeError('GitHub operation timed out after four attempts') from None
            time.sleep(10*(attempt+1));continue
        if result.returncode==0:return result
        message=(result.stderr or result.stdout or 'No diagnostic returned')
        for name in ('GH_TOKEN','GITHUB_TOKEN'):
            if os.environ.get(name):message=message.replace(os.environ[name],'[REDACTED]')
        transient=any(s in message.lower() for s in ('rate limit','secondary rate','http 429','http 500','http 502','http 503','http 504','timeout','connection reset','unexpected eof'))
        if transient and attempt<3:
            time.sleep(15*(attempt+1));continue
        if check:raise RuntimeError(f'GitHub {" ".join(args[:3])} failed: {message[-1600:]}')
        return result
    return result

def ensure_release(tag,repo,sha):
    existing=gh(['api',f'repos/{repo}/releases/tags/{tag}'],False)
    if not existing.returncode:return
    if 'HTTP 404' not in existing.stderr:raise RuntimeError('Cannot inspect release: '+existing.stderr[-1000:])
    created=gh(['release','create',tag,'--repo',repo,'--target',sha,'--latest=false','--title',tag,'--notes','Collection in progress'],False)
    if created.returncode:
        # A timed-out request or competing operation may already have created it.
        found=gh(['api',f'repos/{repo}/releases/tags/{tag}'],False)
        if found.returncode:raise RuntimeError('Cannot create history release: '+created.stderr[-1600:])
