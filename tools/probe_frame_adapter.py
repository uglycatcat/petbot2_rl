"""Fixed-policy amplitude probe for the learned balance-only frame correction."""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--frame-adapter-factor',type=float,choices=[.5,1.,2.],default=1.)
options,remaining=p.parse_known_args()
assert '--evaluate' in remaining and '--checkpoint' in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_probe(runner,folder):
    actor=runner.alg.get_policy()
    assert abs(float(actor.adapter_scale[0])-.03)<1e-7
    actor.adapter_scale[0]*=options.frame_adapter_factor
    (folder/'frame_adapter_probe.json').write_text(json.dumps(dict(diagnostic_only=True,
        factor=options.frame_adapter_factor,active='existing balance adapter gate',
        unchanged='base actor, common/yaw corrections, feedback, slew and absolute bounds',
        source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,('Evaluator changed',old)
    source=source.replace(old,new)
replace('policy=runner.get_inference_policy(device=args.device);',
        'install_probe(runner,folder);policy=runner.get_inference_policy(device=args.device);')
replace('correction.new_tensor([.03,.08,.8])',
        'correction.new_tensor([.03*options.frame_adapter_factor,.08,.8])')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
