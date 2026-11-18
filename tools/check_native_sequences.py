"""Compare the compiled stepping loop with independently saved Python replays."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from recover.model import PROJECT_ROOT,source_fingerprint
from recover.poses import load_poses
from recover.sequence_native import NativeSequences


def main():
    baseline=PROJECT_ROOT/'artifacts/baseline/target_replay'
    actions=np.load(baseline/'four_bounded_sequences.npz')['actions'].transpose(1,0,2)
    initial,_,_=load_poses()
    start=time.monotonic()
    with NativeSequences() as engine:
        summary,trace=engine.run(initial,actions,record=True)
        errors=[];expected=[]
        for i in range(4):
            name='native_pose_01_bounded' if i==0 else f'native_pose_{i+1:02d}'
            reference=np.load(baseline/f'{name}.npz');report=json.loads((baseline/f'{name}.json').read_text())
            err={key:float(np.max(np.abs(trace[i,:,sl]-reference[key][:,0]))) for key,sl in
                 [('qpos',slice(0,engine.nq)),('qvel',slice(engine.nq,engine.nq+engine.nv))]}
            err['hold']=float(np.max(np.abs(trace[i,:,-2]-reference['hold'][:,0])))
            err['contacts']=int(np.count_nonzero(trace[i,:,-4]!=np.sum(reference['contacts'][:,0]*(1<<np.arange(4)),axis=-1)))
            errors.append(err);expected.append(report['first_success_s'][0])
        assert max(e['qpos'] for e in errors)<1e-8,errors
        assert max(e['qvel'] for e in errors)<1e-7,errors
        assert max(e['hold'] for e in errors)<1e-9 and not any(e['contacts'] for e in errors),errors
        assert np.max(np.abs(summary[:,1]-expected))<1e-9,(summary[:,1],expected)
        # Independent episode workspaces must also be invariant to batch ordering.
        repeat,_=engine.run(initial[[3,1,3,0]],actions[[3,1,3,0]])
        assert np.array_equal(repeat,summary[[3,1,3,0]])
        result=dict(status='passed',source_hash=source_fingerprint(),kernel_sha256=engine.kernel_sha256,
                    errors=errors,first_success_s=summary[:,1].tolist(),batch_order_exact=True,
                    seconds=time.monotonic()-start,
                    note='Known demo is diagnostic only, never a search seed or training sample.')
    output=PROJECT_ROOT/'artifacts/engineering/native_sequences.json'
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
