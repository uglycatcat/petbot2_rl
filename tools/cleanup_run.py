"""Release only processes carrying one supervisor's exact unique run token."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import time


def owned_processes(token):
    if not re.fullmatch('[0-9a-f]{32}',token):raise ValueError('Expected a unique supervisor token')
    marker=('STEP_RUN_TOKEN='+token).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)<=1:continue
        try:
            if marker in (proc/'environ').read_bytes().split(b'\0'):yield int(proc.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass


def cleanup(token):
    found=set()
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid in owned_processes(token):
            if pid==os.getpid():raise RuntimeError('Cleanup must run outside the supervised process group')
            found.add(pid)
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        time.sleep(.2)
    survivors=list(owned_processes(token))
    if survivors:raise RuntimeError('Owned run processes did not exit')
    return dict(owned_processes_cleaned=len(found))


def main():
    if not Path('/.dockerenv').exists():raise RuntimeError('Cleanup is restricted to the dedicated container')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--token',required=True)
    print(json.dumps(cleanup(p.parse_args().token)))


if __name__=='__main__':main()
