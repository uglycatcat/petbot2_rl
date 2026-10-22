"""Instrument a clean fixed checkout without changing its policy, task or scoring.

Invoke this tool by absolute path with cwd set to the checkpoint's fixed checkout.
Pass normal train_moving.py evaluation arguments including --contact-audit.
"""
import pathlib,sys
source=pathlib.Path('train_moving.py').resolve();sys.path.insert(0,str(source.parent))
code=source.read_text()
needle='        with torch.inference_mode():'
assert code.count(needle)==1
code=code.replace(needle,'        physics_records=[]\n'+needle)
needle="                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()"
assert code.count(needle)==1
record='''                from upright.task import tensor
                h=tensor(env.scene['contact_forces'].data.net_forces_w_history)
                valid=before&m['active']&~done.bool()
                current=(h[:,0,env.upright_ids[3],2]>.05).all(-1)
                assert torch.equal(current[valid],s['rear'][valid]), 'Stale contact history'
                physics_records.append(dict(
                    force=h[:,:,env.upright_ids[3],2].flip(1).cpu().clone(),
                    valid=valid.cpu(),since=since.cpu(),pos=s['pos'].cpu(),
                    quat=s['quat'].cpu(),qfull=s['qfull'].cpu(),
                    rear_z=s['rear_z'].cpu(),front_z=s['front_z'].cpu(),
                    support=m['support'].cpu(),held=m['held'].cpu(),
                    raw=raw.cpu(),applied=applied.cpu(),velocity=s['vel'].cpu()))
'''
code=code.replace(needle,record+needle)
needle="        print({k:v for k,v in result.items() if not isinstance(v,(list,dict))})"
assert code.count(needle)==1
record='''        import numpy as np
        np.savez_compressed(folder/'physics_trace.npz',**{k:torch.stack([r[k] for r in physics_records]).numpy() for k in physics_records[0]})
        (folder/'replay_meta.json').write_text(json.dumps(dict(joint_names=env.scene['robot'].joint_names,physics_dt=env.physics_dt,control_dt=env.step_dt,history_order='oldest_to_newest',source_commit=metadata['commit'],note='Recorded Isaac state replay only; not MuJoCo policy validation')))
'''
code=code.replace(needle,record+needle)
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
