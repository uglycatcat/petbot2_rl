"""One-shot completion/failure messages to the already-authorized current session."""
import json
from pathlib import Path
import shutil
import subprocess
import time


def save(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n'); temporary.replace(path)


def notify(out, key, thread, message):
    path=Path(out)/(key+'_notice.json')
    try:
        with path.open('x') as handle:
            json.dump({'submission_unknown':'attempt_started','at':time.time()},handle)
    except FileExistsError:
        return  # unknown delivery is reconciled, never blindly duplicated
    try:
        executable=shutil.which('codex')
        if not executable or not thread: raise RuntimeError('Same-session event delivery unavailable')
        result=subprocess.run([executable,'queue','--thread',thread,'--message',message],
                              capture_output=True,text=True,timeout=45)
        save(path,dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,at=time.time()))
    except Exception as error:
        save(path,dict(submission_unknown=repr(error),at=time.time()))
