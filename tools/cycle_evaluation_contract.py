"""Validate engineering completeness separately from cycle/qualified task success."""
import math

def validate(result,count=64):
    if result.get('count')!=count or result.get('unfinished')!=0:raise ValueError('Wrong count or unfinished evaluation')
    for key in ['takeover','balance','landing','handoff','complete','qualified','failure']:
        v=result.get(key);xs=result.get('outcomes',{}).get(key)
        if type(v) is not int or not isinstance(xs,list) or len(xs)!=count or any(type(x) is not bool for x in xs) or sum(xs)!=v:raise ValueError('Bad outcomes '+key)
    for a,b in zip(result['outcomes']['qualified'],result['outcomes']['complete']):
        if a and not b:raise ValueError('Qualified without complete cycle')
    if result.get('contract',{}).get('command_sampling')=='independent_six_groups_v1':
        groups=['forward','reverse','left','right','curve','stop']
        coverage=result.get('command_coverage',{})
        for stage in ['active','balance']:
            if set(coverage.get(stage,{}))!=set(groups):raise ValueError('Missing command coverage '+stage)
            for name,n in coverage[stage].items():
                if type(n) is not int or not 0<=n<=count:raise ValueError('Invalid coverage '+name)
        if count>=6 and result['takeover']==count and not all(coverage['active'].values()):raise ValueError('Missing active command group')
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite metric')
        if isinstance(x,dict):
            for v in x.values():finite(v)
        if isinstance(x,list):
            for v in x:finite(v)
    finite(result);return result
