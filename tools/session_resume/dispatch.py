"""Queue one scheduled continuation, with an external stop switch and slot deduplication."""
import argparse, datetime as dt, fcntl, json, os, pathlib, subprocess
from zoneinfo import ZoneInfo

def dispatch(config_path, dry_run=False):
    path=pathlib.Path(config_path); folder=path.parent
    with (folder/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads(path.read_text()); now=dt.datetime.now(ZoneInfo('Asia/Shanghai'))
        if not cfg['enabled'] or now<dt.datetime.fromisoformat(cfg['not_before']):
            print('disabled or before first delivery'); return
        slots=[now.replace(hour=h,minute=0,second=0,microsecond=0) for h in (4,10,16,22)]
        slots += [(now-dt.timedelta(days=1)).replace(hour=22,minute=0,second=0,microsecond=0)]
        slot=max(t for t in slots if t<=now); key=slot.strftime('%Y%m%dT%H%M')
        record=folder/(key+'.json')
        if record.exists(): print('slot already attempted:',key); return
        cmd=[cfg['codex'],'queue','--thread',cfg['thread'],'--message',f"北京时间{slot.isoformat()}定时续接。"+cfg['message']]
        if dry_run: print(json.dumps({'slot':key,'thread':cfg['thread'],'message':cmd[-1]},ensure_ascii=False));return
        # Record before submitting: an uncertain delivery is never blindly retried.
        data={'slot':slot.isoformat(),'attempted_at':now.isoformat(),'status':'submitting'}
        record.write_text(json.dumps(data,ensure_ascii=False,indent=2))
        try:
            result=subprocess.run(cmd,cwd=cfg['cwd'],capture_output=True,text=True,timeout=90)
            data.update(status='queued' if result.returncode==0 else 'failed',exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except subprocess.TimeoutExpired:
            data.update(status='delivery_unknown')
        record.write_text(json.dumps(data,ensure_ascii=False,indent=2));print(data['status'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--dry-run',action='store_true');a=p.parse_args();dispatch(a.config,a.dry_run)
