"""Bundle published project code, selected policy, frozen assets and physical evidence.
Host-side file operations only; no simulation or training is launched.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--release', required=True)
args = parser.parse_args()
release = PROJECT / 'release' / args.release
selection = json.loads((release / 'selection.json').read_text())
height = json.loads((release / 'evaluation.json').read_text())['step_height']
evidence_paths = selection['evidence']
output = PROJECT / 'deliverables' / (args.release + '.zip')
if output.exists():
    raise FileExistsError('Preserve the existing delivery bundle; use a new release name')
if subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT, text=True).strip():
    raise RuntimeError('Commit a clean project before packaging its source')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
batch = PROJECT / selection['batch']
digest = lambda data: hashlib.sha256(data).hexdigest()
raw_hash = digest((release / 'policy_raw.pt').read_bytes())
for directory in [evidence_paths['export_eval'], evidence_paths['video']]:
    result = json.loads((batch / directory / 'playback.json').read_text())
    assert result['policy_sha256'] == raw_hash
    assert result['result']['success_rate'] == 1.0
    assert abs(result['result']['step_height'] - height) < 1e-9
assert digest((batch / 'assets/robot.usda').read_bytes()) == selection['asset_sha256']

files = {}
archive = subprocess.check_output(['git', 'archive', commit], cwd=PROJECT)
with tarfile.open(fileobj=io.BytesIO(archive)) as source:
    for item in source.getmembers():
        if item.isfile():
            files[item.name] = source.extractfile(item).read()
for name in ['robot.usda', 'model.json']:
    files['assets/' + name] = (batch / 'assets' / name).read_bytes()
evidence = {
    'exported64': (evidence_paths['export_eval'], ['evaluation.json', 'evaluation_traces.npz',
                    'policy_inputs.npz', 'playback.json', 'trajectory_review.json', 'tensorboard_review.json']),
    'checkpoint64': (evidence_paths['checkpoint_eval'], ['evaluation.json', 'evaluation_traces.npz']),
    'video': (evidence_paths['video'], ['execution.mp4', 'evaluation.json', 'evaluation_traces.npz',
                    'policy_inputs.npz', 'playback.json', 'success.png']),
}
for target, (directory, names) in evidence.items():
    for name in names:
        files[f'evidence/{target}/{name}'] = (batch / directory / name).read_bytes()
# Optional existing research visuals keep comparison links usable in an unpacked bundle.
for name in selection.get('supplementary_files', []):
    path = (PROJECT / name).resolve()
    path.relative_to(PROJECT)
    if name in files:
        raise ValueError('Supplement would overwrite an archived file: ' + name)
    files[name] = path.read_bytes()
files['SOURCE_COMMIT'] = (commit + '\n').encode()
files['SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name, data in sorted(files.items())).encode()
output.parent.mkdir(exist_ok=True)
temporary = output.with_suffix('.pending')
with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for name, data in sorted(files.items()):
        bundle.writestr('step_explore/' + name, data)
with zipfile.ZipFile(temporary) as bundle:
    assert bundle.testzip() is None
    for name, data in files.items():
        assert digest(bundle.read('step_explore/' + name)) == digest(data), name
temporary.replace(output)
bundle_hash = digest(output.read_bytes())
output.with_suffix('.zip.sha256').write_text(f'{bundle_hash}  {output.name}\n')
print(json.dumps({'bundle': str(output.relative_to(PROJECT)), 'bytes': output.stat().st_size,
                  'source_commit': commit, 'sha256': bundle_hash, 'files': len(files)}, indent=2))
