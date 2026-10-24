"""Cycle-v10 frozen-actor diagnostic of policy cadence with a 60ms history grid.

The source training entry remains strict. This explicit actor-transfer experiment
changes the control clock, reports a separate contract and never trains a model.
"""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--policy-period',type=float,choices=[.06,.02],default=.06)
options,remaining=p.parse_known_args()
assert '--probe' in remaining and '--initialize-cycle-from' in remaining
assert '--checkpoint' not in remaining and '--evaluate' not in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def cadence_observations(obs):
    if options.policy_period==.06:return obs
    obs=dict(obs)
    value=obs['policy']
    obs['policy']=value.reshape(value.shape[0],15,58)[:,2::3].reshape(value.shape[0],290)
    return obs


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,('Fixed evaluator changed',old)
    source=source.replace(old,new)

replace('TensorDict(self.env.observation_manager.compute(),batch_size=[self.num_envs])',
        'TensorDict(cadence_observations(self.env.observation_manager.compute()),batch_size=[self.num_envs])')
replace('TensorDict(obs,batch_size=[self.num_envs])','TensorDict(cadence_observations(obs),batch_size=[self.num_envs])')
replace('    torch.manual_seed(args.seed);env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)',
        '    cfg.decimation=round(options.policy_period/cfg.sim.dt);cfg.sim.render_interval=cfg.decimation\n'
        '    cfg.observations.policy.history_length=15 if options.policy_period==.02 else 5\n'
        '    torch.manual_seed(args.seed);env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)')
replace("task='upright_cycle_v10'","task='diagnostic_cycle_v10_policy_cadence'")
replace("!='upright_cycle_v9'","!='upright_cycle_v10'")
replace('d13cd449c5e8296ef336976679c3989f56e2ae8e','90d55f0f9f0d9428c49fce1d99805c89a0785469')
replace("'frame_slew','control_dt','source_sha256'","'frame_slew','source_sha256'")
replace("        loaded=torch.load(initial,map_location='cpu',weights_only=False)",
        "        assert prior['contract']['control_dt']==.06 and prior['contract']['yaw_feedback_design']=='fast20'\n"
        "        loaded=torch.load(initial,map_location='cpu',weights_only=False)")
replace("    if args.checkpoint:runner.load(args.checkpoint)",
        "    (folder/'cadence_probe.json').write_text(json.dumps(dict(diagnostic_only=True,policy_period=options.policy_period,history_grid_seconds=.06,history_frames=5,physics_dt=env.physics_dt,contact_history_seconds=.06,source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))\n"
        "    if args.checkpoint:runner.load(args.checkpoint)")
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
