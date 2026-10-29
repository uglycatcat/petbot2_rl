"""Record installed dependency identities without host/account configuration."""
import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path
report={'python':sys.version.split()[0],'packages':{}}
for name in ('torch','isaaclab','isaaclab-physx','rsl-rl-lib','tensordict','numpy','onnx','onnxruntime'):
    report['packages'][name]=md.version(name)
report['isaac_sim']=Path('/isaac-sim/VERSION').read_text().strip()
# The vendor image has a source snapshot rather than a Git checkout.
dist=md.distribution('isaaclab');source=Path(json.loads(dist.read_text('direct_url.json'))['url'].removeprefix('file://'))
h=hashlib.sha256()
for p in sorted(source.rglob('*.py')):
    h.update(str(p.relative_to(source)).encode());h.update(p.read_bytes())
report['isaaclab_python_source_sha256']=h.hexdigest()
print(json.dumps(report,indent=2))
