"""Observational diagnostic: execute fixed evaluator with extra state fields only.
Run from the fixed run checkout. Output is diagnostic, never a replacement gate.
"""
import pathlib,sys
if '--evaluate' not in sys.argv:raise RuntimeError('Evaluation only')
sys.path.insert(0,str(pathlib.Path.cwd()))
source=pathlib.Path('train_cycle.py').read_text()
old='trace.append(dict(t='
new="trace.append(dict(q=s['q'].tolist(),g=s['g'].tolist(),vel=s['vel'].tolist(),body_history=s['body_history'].tolist(),four=s['four'].tolist(),front_contact=s['front_contact'].tolist(),destination=ensure(env).destination.clone().tolist(),t="
assert source.count(old)==1
source=source.replace(old,new)
print('DIAGNOSTIC: extra logging only, original fixed task, no control/observation/reward changes',flush=True)
exec(compile(source,'train_cycle.py','exec'),globals())
