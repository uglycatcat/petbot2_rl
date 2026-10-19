"""Read-only detailed evaluation of a preserved moving-task checkout.

Run from its fixed checkout with normal train_moving evaluation arguments.
Only adds recordings in memory; leaves source, reward and success rules intact.
"""
import pathlib, sys
source=pathlib.Path('train_moving.py').resolve();sys.path.insert(0,str(source.parent))
code=source.read_text()
needle='        with torch.inference_mode():'
assert code.count(needle)==1
code=code.replace(needle,'        diagnostic=[]\n'+needle)
needle="                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()"
assert code.count(needle)==1
record='''                diagnostic.append(dict(active=active.tolist(),since=since.tolist(),
                    support=m['support'].tolist(),held=m['held'].tolist(),
                    raw_error=m['raw_error'].tolist(),raw_distance=m['raw_distance'].tolist(),
                    heading_error=m['heading_error'].tolist(),fixed_error=m['fixed_error'].tolist(),
                    roll=s['g'][:,1].tolist(),tilt_rate=m['tilt_rate'].tolist(),
                    physical_failure=m['physical_failure'].tolist(),
                    pos=s['pos'].tolist(),raw=raw.tolist(),forward=s['forward'].tolist()))
'''
code=code.replace(needle,record+needle)
needle="        print({k:v for k,v in result.items() if not isinstance(v,(list,dict))})"
code=code.replace(needle,"        (folder/'diagnostic.json').write_text(json.dumps(diagnostic))\n"+needle)
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
