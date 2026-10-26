"""Check evaluation-only commands and reject hidden training/source drift."""
import copy
import importlib.util
import pathlib
import sys
from unittest.mock import patch

source = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source/'tools'))
spec = importlib.util.spec_from_file_location('candidate_queue', source/'tools/evaluate_candidates.py')
queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
root = pathlib.Path('/repository')
plan = dict(fixed='demo_project/runs/_validation/fixed', container='owned',
            common_args=['--observation-frame', 'balance_local'])
job = dict(name='parent', mode='untuned_parent', seed=123,
           checkpoint='demo_project/runs/parent/logs/parent/model_199.pt')
out = root/'demo_project/runs/_validation'
command = queue.command(root, out, plan, job, 2)
assert '--probe' in command and '--initialize-cycle-from' in command
assert '--iterations' not in command and '--evaluate' not in command
assert command[command.index('--reserve-mib')+1] == '4000'
trained = {**job, 'mode':'checkpoint', 'entrypoint':'tools/probe_landing_state.py'}
command = queue.command(root, out, plan, trained, 3)
assert '--evaluate' in command and '--checkpoint' in command and '--probe' not in command
for invalid in [dict(mode='train'), dict(args=['--iterations=200']),
                dict(checkpoint='../outside.pt'), dict(entrypoint='train.py')]:
    try:
        queue.command(root, out, plan, {**job, **invalid}, 2)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('Unsafe evaluation accepted: '+str(invalid))
invalid_plan = {**plan, 'common_args':['--allow-dirty']}
try:
    queue.command(root, out, invalid_plan, job, 2)
except AssertionError:
    pass
else:
    raise AssertionError('Dirty source flag accepted')
for outputs in [('wrong\n',), ('fixed\n', ' M train_cycle.py\n'), ('fixed\n', '', 'task changed')]:
    candidate = {**plan, 'commit':'fixed', 'accepted_task_commit':'accepted', 'jobs':[job]}
    with patch.object(queue.subprocess, 'check_output', side_effect=outputs):
        try:
            queue.verify_plan(root, candidate)
        except AssertionError:
            pass
        else:
            raise AssertionError('Source drift accepted')
print('Evaluation-only commands, source drift rejection and4GiB resource reserve passed')
