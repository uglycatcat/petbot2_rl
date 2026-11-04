"""CPU-only release integrity checks; no simulator, weights, or training."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from upright.config import load_config
from upright.model import load_model,source_fingerprint
from upright.experiment import digest
r=json.loads((ROOT/'release.json').read_text())
for item in r['source_files']:
 assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
c=load_config();assert c.to_dict()==r['resolved_training_configuration']
assert digest(c.to_dict())==r.get('training_config_sha256',r['source_config_sha256'])
assert source_fingerprint()==r['source_model_sha256']
s=json.loads((ROOT/'project.json').read_text())
assert s['historical_weights_allowed'] is False and s['allowed_host_gpus']==[2,3]
assert not any((ROOT/'outputs').glob('**/*.pt'))
print(json.dumps(dict(release=r['release'],source_files_checked=len(r['source_files']),effective_config_exact=True,seed=c.seed,iterations=c.iterations,num_envs=c.num_envs,source_model_exact=True,training_enabled=s['training_enabled'],automatic_iteration_enabled=s['automatic_iteration_enabled'],simulation_or_training_started=False),indent=2))
