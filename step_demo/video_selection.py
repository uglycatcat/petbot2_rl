"""Select among independently evaluated policies in this batch only."""
import json
import math
from pathlib import Path


def select_best(folder, jobs):
    folder=Path(folder);candidates=[];excluded=[]
    for job in sorted(jobs,key=lambda j:j['id']):
        name=job['id'];directory=folder/name
        try:
            state=json.loads((directory/'job_status.json').read_text())
            if state['status']!='complete':raise ValueError('job did not complete evaluation')
            report=json.loads((directory/'analysis.json').read_text())['evaluation']
            if not report['independent_physics']['all_terminal']:
                raise ValueError('physical review incomplete')
            result=json.loads((directory/'evaluation/result.json').read_text())
            episodes=result['episodes'];n=len(episodes)
            if not n or n!=report['outcomes']['total']:raise ValueError('episode count mismatch')
            successes=sum(bool(e['success']) for e in episodes)
            if successes!=report['outcomes']['successes']:raise ValueError('success count mismatch')
            phase=max((int(k) for k,v in report['phase_frames'].items() if v and 0<=int(k)<=6),default=0)
            mean=lambda field:sum(float(e[field]) for e in episodes)/n
            score=[successes/n,phase,mean('drive_distance'),mean('hold_s'),mean('return_sum'),-mean('quality_cost')]
            if not all(math.isfinite(v) for v in score):raise ValueError('nonfinite selection score')
            checkpoint=result['checkpoint']
            if Path(checkpoint).name!=checkpoint or not (directory/'train'/checkpoint).is_file():
                raise ValueError('evaluated checkpoint missing')
            candidates.append(dict(job=name,gpu=job['gpu'],checkpoint=checkpoint,score=score,
                successes=successes,episodes=n,completed_updates=result['completed_updates']))
        except (OSError,ValueError,KeyError,TypeError) as error:
            excluded.append(dict(job=name,reason=str(error)))
    best=max(candidates,key=lambda c:c['score']) if candidates else None
    return dict(selected=best,candidates=candidates,excluded=excluded,
        criterion=['success_rate','highest_verified_legal_phase','mean_net_drive','mean_hold',
                   'mean_episode_return','negative_mean_quality_cost'],
        scope='evaluated checkpoints in this batch only; not a search of historical runs or every saved checkpoint')
