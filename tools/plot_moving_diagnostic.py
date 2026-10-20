"""Plot relative-time medians from detailed native-range evaluations."""
import argparse,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--directory',type=pathlib.Path,required=True);a=p.parse_args()
fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True)
for seed in (42,43):
    rows=json.loads((a.directory/f'diagnose_b{seed}_native/diagnostic.json').read_text())
    buckets=[[] for _ in range(202)]
    for row in rows:
        for i,active in enumerate(row['active']):
            if active:
                k=min(201,round(row['since'][i]/.06))
                buckets[k].append((row['raw_distance'][i],abs(row['heading_error'][i]),float(row['support'][i])))
    valid=[(i,np.asarray(b)) for i,b in enumerate(buckets) if b]
    t=np.array([i*.06 for i,b in valid])
    axes[0].plot(t,[np.median(b[:,0]) for i,b in valid],label=f'B{seed}')
    axes[1].plot(t,[np.median(b[:,1]) for i,b in valid])
    axes[2].plot(t,[b[:,2].mean() for i,b in valid])
for ax,threshold,label in zip(axes,[.15,.25,None],['Median path error (m)','Median heading error (rad)','Rear support fraction']):
    ax.set_ylabel(label);ax.grid(alpha=.25)
    if threshold is not None:ax.axhline(threshold,color='red',linestyle='--',label='Success threshold')
axes[0].legend();axes[2].set_xlabel('Seconds after takeover')
fig.suptitle('Batch 2 final policies: balance retained, pose error accumulates')
fig.tight_layout();fig.savefig(a.directory/'diagnostic_trajectory.png',dpi=160)
