"""Container-only PhysX engineering check; scripted actions are not training data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, default=Path('artifacts/robot'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--on-platform',action='store_true',help='Isolated top-contact check; not a valid task rollout')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        import torch
        from step_demo.config import load_config, PROJECT_ROOT
        from step_demo.isaac_env import StepEnv, tensor
        training = json.loads((PROJECT_ROOT/'configs/train_v1.json').read_text())
        training['num_envs'] = 8
        env = StepEnv(load_config(model_dir=args.model_dir), training, args.asset_dir, args.device)
        env.trace_ids = list(range(env.num_envs))
        obs, _ = env.reset()
        if args.on_platform:
            pose=tensor(env.robot.data.default_root_pose).clone()
            pose[:,:3]+=env.scene.env_origins
            pose[:,0]+=.6;pose[:,2]+=.02
            env.robot.write_root_pose_to_sim_index(root_pose=pose)
        initial = dict(joints=env.robot.joint_names, bodies=env.robot.body_names,
                       mass=tensor(env.robot.data.body_mass)[0].sum().item(), observation_shape=list(obs['policy'].shape))
        print(json.dumps(initial), flush=True)
        with (args.output/'trace.jsonl').open('w') as file:
            for step in range(args.steps):
                actions = torch.tensor([[0.,-1.],[0.,-1.],[0.,-.65],[0.,-.65],
                                        [-.5,-.9],[.5,-.9],[-.8,-.7],[.8,-.7]],device=env.device)
                if step < 50 or args.on_platform: actions[:] = torch.tensor([0.,-1.],device=env.device)
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                assert env.captured_substeps == (step+1)*training['decimation']
                assert (env.packet_count == 1).all()
                if args.on_platform:
                    assert not env.front_hit.any(), 'Ordinary top contact was incorrectly classified as edge collision'
                    assert not terminated.any(), 'Static top-contact probe unexpectedly terminated'
                trace = env.last_trace if hasattr(env,'last_trace') else env.trace(slice(None))
                file.write(json.dumps(dict(step=step, **trace))+'\n')
        if args.on_platform:
            assert env.supported_top.all(), 'All wheels must establish real top support'
            assert not env.completed.any(), 'Starting on top cannot prove front clearance or task completion'
        (args.output/'result.json').write_text(json.dumps(dict(status='passed',initial=initial,
            isolated_top_contact=args.on_platform,all_wheels_on_top=env.supported_top.all(-1).tolist(),
            captured_substeps=env.captured_substeps,totals=env.totals,episodes=env.terminal_records),indent=2))
        env.close()
    except BaseException:
        import traceback
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        app.close()

if __name__ == '__main__': main()
