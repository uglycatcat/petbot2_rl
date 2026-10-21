"""Validate a fresh evaluation before marking a queue job successful."""
import math

def validate(result, count=64):
    if not isinstance(result,dict) or result.get('count')!=count:
        raise ValueError('Wrong evaluation population')
    for key in ('taken_over','success','maintained_final_2s','final_success','premature','warmup_failed','unfinished','lifted'):
        v=result.get(key)
        if type(v) is not int or not 0<=v<=count:raise ValueError('Invalid count: '+key)
    if result['unfinished']:raise ValueError('Unfinished evaluation episodes')
    for key in ('success_per_env','maintained_per_env','premature_per_env','lifted_per_env'):
        values=result.get(key)
        if not isinstance(values,list) or len(values)!=count or any(type(x) is not bool for x in values):
            raise ValueError('Invalid environment outcomes: '+key)
    for scalar,vector in [('success','success_per_env'),('maintained_final_2s','maintained_per_env'),('premature','premature_per_env'),('lifted','lifted_per_env')]:
        if result[scalar]!=sum(result[vector]):raise ValueError('Outcome sum mismatch: '+scalar)
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite evaluation metric')
        if isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,list):
            for v in x:finite(v)
    finite(result)
    return result
