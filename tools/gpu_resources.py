"""Read shared-card usage without attributing other users' memory to STEP."""
import subprocess


def memory_by_container(gpu, containers):
    apps = subprocess.check_output([
        'nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    memory = {}
    for row in apps.splitlines():
        pid, _, value = row.partition(',')
        if value.strip().isdigit():
            memory[pid.strip()] = float(value)
    result = {}
    for name in containers:
        pids = subprocess.check_output(['docker', 'top', name, '-eo', 'pid'], text=True)
        result[name] = sum(memory.get(pid.strip(), 0.) for pid in pids.splitlines()[1:])
    return result


def pressure_victim(container, usage):
    """The second slot yields first; the first waits until its memory is released."""
    return max({container, *(name for name, mib in usage.items() if mib > 0)})
