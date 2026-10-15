"""PhysX integration checks for pose/reset/history, success events, and action wiring."""
import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', required=True)
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app


def main():
    import math
    import gymnasium as gym
    import torch
    import agile.rl_env.tasks.stand_up.petbot
    from isaaclab_tasks.utils import load_cfg_from_registry
    from agile.petbot.pre_learn import pre_learn
    from agile.petbot.core import as_torch, recovery_potential
    from agile.petbot.mdp import state, actor_observation
    from agile.petbot.model import POSITION_JOINTS, load_model
    from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner

    task='PetBot-Recovery-Repair-v0'
    cfg=load_cfg_from_registry(task,'env_cfg_entry_point');cfg.scene.num_envs=64;cfg.sim.device='cuda:0'
    agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point');agent.device='cuda:0'
    agent.petbot_collection.split='eval';agent.petbot_collection.seed=20261015;agent.petbot_collection.num_states=512
    agent.petbot_eval_dataset=args.dataset;agent.petbot_allow_legacy_xyzw=True
    env=gym.make(task,cfg=cfg);base=env.unwrapped;pre_learn(base,task,agent)
    robot=base.scene['robot']; n=base.num_envs; dev=base.device
    ids=robot.find_joints(list(POSITION_JOINTS),preserve_order=True)[0]
    home=as_torch(robot.data.default_joint_pos).clone()
    spec=load_model();bounds=torch.tensor([spec['joints'][k]['range'] for k in POSITION_JOINTS],device=dev)
    action=torch.zeros(n,7,device=dev);action[:,:5]=(home[:,ids]-bounds.mean(-1))/((bounds[:,1]-bounds[:,0])/2)
    default_root=as_torch(robot.data.default_root_state).clone()
    c=math.sqrt(.5)
    q=torch.tensor([[0,0,0,1],[0,c,0,c],[0,-c,0,c],[c,0,0,c],[1,0,0,0]],device=dev)
    expected_g=torch.tensor([[0,0,-1],[1,0,0],[-1,0,0],[0,-1,0],[0,0,1]],device=dev,dtype=torch.float32)
    root=default_root.clone();root[:,:3]=torch.tensor([0.,0.,.7],device=dev)
    root[:,3:7]=q[torch.arange(n,device=dev)%5];root[:,7:]=torch.tensor([.1,.2,.3,.4,.5,.6],device=dev)
    base.petbot_dataset={'root':root.cpu(),'joint_pos':home.cpu(),'joint_vel':torch.zeros_like(home).cpu()}
    expected=expected_g[torch.arange(n,device=dev)%5]
    checks=[]
    for repeat in range(3):
        # Deliberately touch upstream lazy data before reset: the task must not inherit it.
        _=as_torch(robot.data.projected_gravity_b).clone()
        base.petbot_eval_cursor=0;obs,_=env.reset()
        s=state(base)
        torch.testing.assert_close(s['gravity'],expected,atol=1e-6,rtol=0)
        torch.testing.assert_close(as_torch(robot.data.root_link_pose_w)[:,3:7],root[:,3:7])
        torch.testing.assert_close(base.petbot_previous_potential,recovery_potential(expected,torch.zeros(n,5,device=dev)))
        frames=obs['policy'].reshape(n,10,27)
        torch.testing.assert_close(frames,actor_observation(base)[:,None,:].expand_as(frames))
        # Known 90-degree Y rotations give independent expected velocity transformations.
        torch.testing.assert_close(s['lin_vel'][1],torch.tensor([-.3,.2,.1],device=dev),atol=1e-6,rtol=0)
        torch.testing.assert_close(s['ang_vel'][2],torch.tensor([.6,.5,-.4],device=dev),atol=1e-6,rtol=0)
        assert torch.count_nonzero(s['wheel_forces'])==0, 'Contact forces leaked across reset'
    checks.append('known_xyzw_pose_velocity_potential_and_full_history_reset')

    base.petbot_episode_records=[]
    base.episode_length_buf[:]=base.max_episode_length-1
    obs,rew,term,trunc,_=env.step(action)
    assert trunc.all() and not term.any();assert len(base.petbot_episode_records)==n
    assert sorted(r['initial_index'] for r in base.petbot_episode_records)==list(range(n))
    assert not base.petbot_hold.ever.any()
    frames=obs['policy'].reshape(n,10,27)
    torch.testing.assert_close(frames,actor_observation(base)[:,None,:].expand_as(frames))
    base.petbot_episode_records.clear();base.episode_length_buf[0]=base.max_episode_length-1
    obs,rew,term,trunc,_=env.step(action)
    assert trunc.sum()==1 and trunc[0] and not term.any()
    assert [r['env_id'] for r in base.petbot_episode_records]==[0]
    torch.testing.assert_close(obs['policy'][0].reshape(10,27),actor_observation(base)[0].expand(10,27))
    checks.append('automatic_full_and_partial_timeout_reset_and_episode_records')
    # Physical states agree with native frame conversions after a fresh simulation tick.
    s=state(base)
    base.scene.update(dt=base.physics_dt)
    torch.testing.assert_close(s['lin_vel'],as_torch(robot.data.root_lin_vel_b),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(s['ang_vel'],as_torch(robot.data.root_ang_vel_b),atol=1e-5,rtol=1e-5)
    # Real failure dominates simultaneous horizon expiration in the returned PPO contract.
    pose=as_torch(robot.data.root_link_pose_w).clone();pose[:,0]+=1.5
    robot.write_root_pose_to_sim(pose)
    base.episode_length_buf[:]=base.max_episode_length-1
    _,_,term,trunc,_=env.step(action)
    assert term.all() and not trunc.any()
    checks.append('fresh_native_velocity_agreement_and_failure_over_timeout_precedence')

    # Standing is a positive integration control for the actual reward and success managers.
    root=default_root.clone();root[:,:2]=0;root[:,2]=base.petbot_standing_height+.002;root[:,3:7]=torch.tensor([0.,0.,0.,1.],device=dev);root[:,7:]=0
    base.petbot_dataset={'root':root.cpu(),'joint_pos':home.cpu(),'joint_vel':torch.zeros_like(home).cpu()}
    base.petbot_eval_cursor=0;env.reset();base.petbot_episode_records.clear()
    event_count=torch.zeros(n,device=dev);positive_reward=False
    for _ in range(150):
        obs,rew,term,trunc,_=env.step(action)
        assert not term.any() and not trunc.any()
        event_count+=base.petbot_new_success
        positive_reward|=bool((rew>0).any())
    assert (event_count==1).all(), f'Continuous success not reached exactly once: {event_count.tolist()}'
    assert base.petbot_hold.ever.all() and positive_reward
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=make_rsl_rl_runner(wrapped,agent,log_dir=None,device=dev)
    policy=runner.get_inference_policy(device=dev)
    with torch.no_grad(): actual=policy(wrapped.get_observations())
    torch.testing.assert_close(actual,action,atol=1e-6,rtol=0)
    torch.testing.assert_close(robot.data.default_joint_pos.torch if hasattr(robot.data.default_joint_pos,'torch') else robot.data.default_joint_pos,home)
    checks.append('one_success_event_stable_reward_and_bounded_policy_home_initialization')
    output={'checks':checks,'passed':True,'state_contract':base.petbot_provenance['state_contract'],'demo_commit':base.petbot_provenance['demo_commit']}
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
    env.close()


if __name__=='__main__':
    try: main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    app.close()
