"""Process ownership regression for native Kit worker cleanup."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid


@unittest.skipUnless(Path('/.dockerenv').exists(),'Process isolation test runs inside the container')
class CleanupTests(unittest.TestCase):
    def test_cleanup_stops_owned_child_and_preserves_unrelated_child(self):
        from tools.cleanup_run import cleanup
        token=uuid.uuid4().hex
        children=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],
            env={**os.environ,'STEP_RUN_TOKEN':t}) for t in (token,uuid.uuid4().hex)]
        try:
            result=cleanup(token)
            children[0].wait(timeout=3)
            self.assertIsNone(children[1].poll())
            self.assertEqual(result['owned_processes_cleaned'],1)
        finally:
            for p in children:
                if p.poll() is None:p.terminate()
                p.wait(timeout=3)


if __name__=='__main__':unittest.main()
