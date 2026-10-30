"""Host-only finite job supervisor, isolated to physical GPU2/3 and one job token."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from job_control import signal_job
from physics_log import PhysicsLogReader, PythonLogReader


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def usage(gpu):
    text = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.total,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    identity, total, used = text.strip().split(',')
    return identity.strip(), int(total), int(used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(2,3), required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--reserve-mib', type=int, default=6000)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not args.workdir.startswith('/workspace/petbot/demo_project/runs/'):
        raise ValueError('Explicit command and fixed run workspace are required')
    container = f'petbot-upright-gpu{args.gpu}'
    identity, total, used = usage(args.gpu)
    inspected = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
    if inspected['HostConfig']['DeviceRequests'][0]['DeviceIDs'] != [identity]:
        raise RuntimeError('Container does not exclusively expose the assigned physical GPU')
    lock = open(f'/tmp/petbot-upright-gpu-{os.getuid()}-{args.gpu}.lock','a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.log.exists() or args.log.with_suffix('.exit.json').exists():
        raise FileExistsError('Do not overwrite an existing attempt')
    if used + args.reserve_mib >= total*.78:
        save(args.log.with_suffix('.exit.json'), dict(exit_code=80, guard_stopped=True,
             reason='insufficient_headroom_before_launch', gpu=args.gpu, launched=False, sampled_peak_mib=used))
        return 80
    token = uuid.uuid4().hex
    invocation = ['docker','exec','-w',args.workdir,'-e',f'PETBOT_JOB_TOKEN={token}',container,
                  '/isaac-sim/python.sh',*command]
    physics, python = PhysicsLogReader(args.log), PythonLogReader(args.log)
    begin = time.time(); peak = used; cause = None
    with args.log.open('x') as output:
        child = subprocess.Popen(invocation, stdout=output, stderr=subprocess.STDOUT)
        save(args.log.with_suffix('.running.json'), dict(pid=child.pid, supervisor_pid=os.getpid(),
             gpu=args.gpu, container=container, token=token, command=invocation, started=time.time()))
        try:
            while child.poll() is None:
                p_error, runtime_error = physics.check(), python.check()
                _, _, used = usage(args.gpu); peak = max(peak, used)
                if p_error or runtime_error or used >= total*.78:
                    cause = dict(physics_error=p_error, runtime_error=runtime_error, guard_stopped=used >= total*.78)
                    signal_job(container, token, 15)
                    try: child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        signal_job(container, token, 9); child.wait(timeout=15)
                    break
                time.sleep(1)
        except BaseException:
            signal_job(container, token, 15)
            raise
        process_code = child.wait()
    p_error = (cause or {}).get('physics_error') or physics.check()
    runtime_error = (cause or {}).get('runtime_error') or python.check()
    guard_stopped = bool((cause or {}).get('guard_stopped'))
    code = 81 if p_error else 82 if runtime_error else 80 if guard_stopped else process_code
    save(args.log.with_suffix('.exit.json'), dict(exit_code=code, process_exit_code=process_code,
         physics_error=p_error, runtime_error=runtime_error, guard_stopped=guard_stopped,
         gpu=args.gpu, gpu_uuid=identity, container=container, job_token=token,
         sampled_peak_mib=peak, total_mib=total, elapsed_s=time.time()-begin, command=invocation))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
