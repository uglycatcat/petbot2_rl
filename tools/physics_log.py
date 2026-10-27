"""PhysX errors invalidate a simulation even when Python exits successfully."""
import re

PATTERN=re.compile(r'PhysX error:|\[Error\].*\[omni\.physx[^\]]*\]',re.IGNORECASE)


class PhysicsLogReader:
    def __init__(self,path,pattern=PATTERN):
        self.path=path;self.pattern=pattern;self.offset=0;self.partial=''

    def check(self):
        if not self.path.exists():return None
        with self.path.open(errors='replace') as f:
            f.seek(self.offset)
            while True:
                chunk=f.read(65536)
                if not chunk:break
                text=self.partial+chunk;lines=text.splitlines(keepends=True)
                self.partial=''
                for line in lines:
                    if self.pattern.search(line):return line.strip()[:1000]
                if lines and not lines[-1].endswith(('\n','\r')):self.partial=lines[-1][-4096:]
            self.offset=f.tell()
        return None


class PythonLogReader(PhysicsLogReader):
    """Isaac shutdown can suppress Python's failing exit code; retain traceback failure."""
    def __init__(self,path):
        super().__init__(path,re.compile(r'^\s*Traceback \(most recent call last\):',re.MULTILINE))
