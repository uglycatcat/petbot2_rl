"""Run identity, source isolation, atomic status and same-run interruption recovery."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import uuid
import platform
from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec
from .config import ROOT
from .model import source_fingerprint, task_implementation_fingerprint

SCHEMA = 'upright_independent_run_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def software_versions():
    versions = {'python': platform.python_version()}
    for name in ('torch', 'rsl-rl-lib', 'tensordict', 'numpy', 'isaaclab', 'onnx', 'onnxruntime'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = 'not-installed'
    spec = find_spec('isaaclab')
    if spec is not None and spec.origin:
        for parent in Path(spec.origin).parents:
            version_file = parent/'VERSION'
            if version_file.is_file() and (parent/'source/isaaclab').is_dir():
                versions['isaaclab_source_version'] = version_file.read_text().strip()
                break
    return versions


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def identities(config):
    return dict(code_sha256=task_implementation_fingerprint(), model_sha256=source_fingerprint(),
                config_sha256=digest(config.to_dict()))


def check_identity(saved, current):
    if saved.get('schema') != SCHEMA or saved.get('initialization') != 'random':
        raise ValueError('Only independently initialized restart runs are supported')
    for key in ('run_id', 'code_sha256', 'model_sha256', 'config_sha256', 'software_versions'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'Same-run recovery identity mismatch: {key}')


def validate_resume(folder, checkpoint, current):
    folder, checkpoint = Path(folder).resolve(), Path(checkpoint).resolve()
    if checkpoint.parent != folder/'checkpoints' or checkpoint.suffix != '.pt':
        raise ValueError('Resume checkpoint must belong to this run/checkpoints directory')
    state = json.loads((folder/'status.json').read_text())
    if state['state'] not in ('interrupted', 'failed_oom', 'running'):
        raise ValueError('Only a forced interruption may resume; completed/new/failed-code runs cannot')
    if state['state'] == 'running':
        # A stale running record may represent SIGKILL/OOM. Never duplicate a live process.
        pid = state.get('pid')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Previous process may still be alive; refusing duplicate resume')
    manifest = json.loads((folder/'run.json').read_text())
    check_identity(manifest, current)
    if manifest.get('software_versions') != current.get('software_versions'):
        raise ValueError('Same-run recovery software versions changed')
    if state.get('run_id') != manifest['run_id']:
        raise ValueError('Run status identity mismatch')
    latest = folder/'checkpoints/latest.json'
    if not latest.exists() or json.loads(latest.read_text())['file'] != checkpoint.name:
        raise ValueError('Resume must use the latest durable checkpoint; cannot silently roll back updates')
    return manifest


@contextlib.contextmanager
def locked_run(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This run is already active') from exc
        yield folder


def new_manifest(config, kind):
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit the complete implementation before starting a fixed run')
    return dict(schema=SCHEMA, run_id=str(uuid.uuid4()), initialization='random', kind=kind,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                **identities(config), config=config.to_dict(), historical_models=[], frozen_parents=False,
                software_versions=software_versions(),
                physics_resume='reset unfinished episodes; optimizer, course and RNG restored; not frame-exact',
                motion_observation='simulator motion estimate interface; real estimator not validated')


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
