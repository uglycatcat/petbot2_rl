"""Validate engineering completeness separately from cycle/qualified task success."""
import math

def validate(result,count=64):
    if result.get('count')!=count or result.get('unfinished')!=0:raise ValueError('Wrong count or unfinished evaluation')
    for key in ['takeover','balance','landing','handoff','complete','qualified','failure']:
        v=result.get(key);xs=result.get('outcomes',{}).get(key)
        if type(v) is not int or not isinstance(xs,list) or len(xs)!=count or any(type(x) is not bool for x in xs) or sum(xs)!=v:raise ValueError('Bad outcomes '+key)
    for a,b in zip(result['outcomes']['qualified'],result['outcomes']['complete']):
        if a and not b:raise ValueError('Qualified without complete cycle')
    if result.get('contract',{}).get('command_sampling') in ('independent_six_groups_v1','mixed_six_sequence_v2','all_directed_six_sequence_v3'):
        groups=['forward','reverse','left','right','curve','stop']
        coverage=result.get('command_coverage',{})
        for stage in ['active','balance']:
            if set(coverage.get(stage,{}))!=set(groups):raise ValueError('Missing command coverage '+stage)
            for name,n in coverage[stage].items():
                if type(n) is not int or not 0<=n<=count:raise ValueError('Invalid coverage '+name)
        if count>=6 and result['takeover']==count and not all(coverage['active'].values()):raise ValueError('Missing active command group')
    if result.get('contract',{}).get('command_sampling')in ('mixed_six_sequence_v2','all_directed_six_sequence_v3'):
        switches=result.get('command_switches');matrix=result.get('command_transition_matrix')
        if not isinstance(switches,list) or len(switches)!=count or any(type(x) is not int or x<0 for x in switches):raise ValueError('Bad command switch counts')
        if not isinstance(matrix,list) or len(matrix)!=6 or any(len(row)!=6 for row in matrix):raise ValueError('Bad transition matrix')
        if any(type(x) is not int or x<0 for row in matrix for x in row) or sum(map(sum,matrix))!=sum(switches):raise ValueError('Transition count mismatch')
        if any(matrix[i][i] for i in range(6)):raise ValueError('Self transition counted as category switch')
        durations=result.get('balance_duration');motion=result.get('takeover_motion')
        if not isinstance(durations,list) or len(durations)!=count or any(x<0 for x in durations):raise ValueError('Bad balance duration')
        if not isinstance(motion,list) or len(motion)!=count or any(len(row)!=2 for row in motion):raise ValueError('Missing actual takeover motion')
        for complete,changes,duration in zip(result['outcomes']['complete'],switches,durations):
            if complete and duration>=8. and changes<1:raise ValueError('Completed long hold without mixed commands')
    if result.get('contract',{}).get('task')in ('upright_cycle_v16','upright_cycle_v17'):
        f=result.get('functional',{});episodes=f.get('episodes',[])
        if f.get('count')!=count or len(episodes)!=count:raise ValueError('Missing functional report')
        if f.get('definition')!=result['contract'].get('functional'):raise ValueError('Functional definition mismatch')
        for key in ('functional_pass','coverage_complete'):
            if f.get(key)!=sum(e[key] for e in episodes):raise ValueError('Functional count mismatch '+key)
        if f.get('stable_completed')!=result['complete']:raise ValueError('Functional completion mismatch')
        if any(e['functional_pass'] and not (e['complete'] and e['coverage_complete']) for e in episodes):raise ValueError('Functional pass without completion/coverage')
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite metric')
        if isinstance(x,dict):
            for v in x.values():finite(v)
        if isinstance(x,list):
            for v in x:finite(v)
    finite(result);return result
