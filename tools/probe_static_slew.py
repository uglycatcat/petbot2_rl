"""Read-only counterfactual: static policy under moving-task target slew limits.

Run from a preserved static checkout. Executes its unchanged train.py in memory;
the evaluation config records the added frame_slew and wheel_slew arguments.
This is not a moving-task result or a modification of the reference checkpoint.
"""
import pathlib
source=pathlib.Path('train.py').resolve()
code=source.read_text()
code=code.replace('AppLauncher.add_app_launcher_args(parser)',
    "parser.add_argument('--frame-slew',type=float,default=6.)\n"
    "parser.add_argument('--wheel-slew',type=float,default=120.)\n"
    'AppLauncher.add_app_launcher_args(parser)')
needle='    wrapped=Adapter(env)'
assert code.count(needle)==1
patch='''    import types
    action=env.action_manager.get_term('symmetric')
    original_process=action.process_actions
    def limited_process(self, actions):
        previous_frame=self.target[:,3].clone()
        previous_wheel=self.speed.clone()
        original_process(actions)
        self.target[:,3:]=previous_frame[:,None]+(self.target[:,3:]-previous_frame[:,None]).clamp(-args.frame_slew*env.step_dt,args.frame_slew*env.step_dt)
        self.speed[:]=previous_wheel+(self.speed-previous_wheel).clamp(-args.wheel_slew*env.step_dt,args.wheel_slew*env.step_dt)
    action.process_actions=types.MethodType(limited_process,action)
'''
code=code.replace(needle,patch+needle)
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
