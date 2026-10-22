"""Diagnostic scripted lowering after policy balance; never training or policy success evidence."""
import pathlib,sys
assert '--evaluate' in sys.argv and '--checkpoint' in sys.argv
s=pathlib.Path('train_cycle.py');code=s.read_text()
needle='                action=policy(obs)'
assert code.count(needle)==1
code=code.replace(needle,needle+'''
                progress=ensure(env);landing=progress.phase==3
                action=action.clone()
                action[landing,0]=2*(progress.destination[landing]-lo)/(hi-lo)-1
                action[landing,1]=(progress.command[landing,0]/.03/25).clamp(-1,1)
                action[landing,2]=(progress.command[landing,1]/3).clamp(-1,1)
''')
needle="        (folder/'evaluation.json').write_text(json.dumps(result,indent=2));"
assert code.count(needle)==1
code=code.replace(needle,"        result['diagnostic_intervention']='scripted landing targets after policy balance; not policy evaluation'\n"+needle)
exec(compile(code,str(s),'exec'),{'__name__':'__main__','__file__':str(s)})
