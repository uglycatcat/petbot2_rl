"""A finite training loss/exit0 must never override a simulator error."""
import argparse,json,pathlib,sys,tempfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from physics_log import PhysicsLogReader, PythonLogReader
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
with tempfile.TemporaryDirectory(dir=pathlib.Path(args.output).parent) as directory:
    path=pathlib.Path(directory)/'run.log';reader=PhysicsLogReader(path)
    assert reader.check() is None
    path.write_text('[Warning] graphics fallback\nLearning iteration 0\n');assert reader.check() is None
    with path.open('a') as f:f.write('[Error] [omni.phys')
    assert reader.check() is None
    with path.open('a') as f:f.write('x.plugin] PhysX error: limit of 64K materials reached.\n')
    assert '64K materials' in reader.check()
    assert '64K materials' in PhysicsLogReader(path).check()
    path=path.with_name('python.log');reader=PythonLogReader(path)
    path.write_text('[Warning] graphics fallback\nLearning iteration 0\n');assert reader.check() is None
    with path.open('a') as f:f.write('Traceback (most recent ')
    assert reader.check() is None
    with path.open('a') as f:f.write('call last):\nFileNotFoundError: missing asset\n')
    assert 'Traceback' in reader.check()
    assert 'Traceback' in PythonLogReader(path).check()
    assert PhysicsLogReader(path).check() is None
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,incremental_errors=True,split_write=True,post_exit_audit=True,python_traceback_even_if_exit_zero=True,python_failure_not_mislabeled_physics=True),indent=2));print('PASS physics and Python validity log guards')
