"""PhysX errors invalidate a simulation even when Python exits successfully."""
import re

PATTERN=re.compile(r'PhysX error:|\[Error\].*\[omni\.physx[^\]]*\]',re.IGNORECASE)


class PhysicsLogReader:
    def __init__(self,path):
        self.path=path;self.offset=0;self.partial=''

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
                    if PATTERN.search(line):return line.strip()[:1000]
                if lines and not lines[-1].endswith(('\n','\r')):self.partial=lines[-1][-4096:]
            self.offset=f.tell()
        return None
