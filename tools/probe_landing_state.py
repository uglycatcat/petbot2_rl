"""Read-only landing-state diagnostics under the fixed cycle evaluation loop."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
assert '--evaluate' in sys.argv or '--probe' in sys.argv


def install_capture():
    from upright.cycle import CycleProgress
    update = CycleProgress.update

    def capture(self, state, fixed):
        result = update(self, state, fixed)
        # Capture before the environment resets terminal rows, like moving_snapshot.
        result['audit_destination'] = self.destination.clone()
        result['audit_initial_frame'] = self.frame.clone()
        result['audit_ground_run'] = self.ground_run.clone()
        return result

    CycleProgress.update = capture


source = pathlib.Path('train_cycle.py').read_text()
anchor = '    cfg=CycleEnvCfg();'
assert source.count(anchor) == 1
source = source.replace(anchor, '    install_capture()\n'+anchor)
anchor = '                trace.append(dict(rise_adapter_gate='
assert source.count(anchor) == 1
source = source.replace(anchor, '''                trace.append(dict(
                    joint_positions=s['q'].tolist(),
                    destination_frame=m['audit_destination'].tolist(),
                    initial_frame=m['audit_initial_frame'].tolist(),
                    landing_ground_run=m['audit_ground_run'].tolist(),
                    angular_velocity=s['vel'][:,3:5].tolist(),
                    rise_adapter_gate=''')
exec(compile(source, 'train_cycle.py', 'exec'), globals())
