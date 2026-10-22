"""Host orchestration regression: a serialized recheck must avoid stale-memory kills."""
import json,pathlib,runpy,sys,tempfile
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import job_control
class Child:
 def __init__(self):self.calls=0;self.returncode=0
 def poll(self):self.calls+=1;return None if self.calls==1 else self.returncode
 def wait(self,timeout=None):return self.returncode
for final_usage,expected in [(1000,False),(20000,True)]:
 with tempfile.TemporaryDirectory(prefix='petbot-guard-check-') as directory:
  log=pathlib.Path(directory)/'job.log';child=Child();responses=[f'0,24564,{v}\n' for v in (1000,20000,final_usage)]
  with patch('sys.argv',['run_guarded.py','--gpu','0','--log',str(log)]),patch('subprocess.check_output',side_effect=responses),patch('subprocess.Popen',return_value=child),patch('time.sleep'),patch.object(job_control,'signal_job') as kill:
   try:runpy.run_path(str(pathlib.Path(__file__).with_name('run_guarded.py')),run_name='__main__')
   except SystemExit as e:assert e.code==(80 if expected else 0)
  data=json.loads(log.with_suffix('.exit.json').read_text());assert data['guard_stopped']==expected
  assert kill.call_count==int(expected)
print('PASS serialized guard recheck: no stale kill after memory freed; still stops own job on sustained breach')
