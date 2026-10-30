"""Create a new immutable run input copy from a clean committed checkout."""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    p.add_argument('--asset-dir',type=Path,default=Path('artifacts/robot'))
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain'],cwd=project,text=True).strip():
        raise RuntimeError('Commit the complete change before creating a fixed run copy')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    archive=subprocess.check_output(['git','archive',revision],cwd=project)
    a.destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(a.destination/'source')
    (a.destination/'asset').mkdir()
    for name in ('petbot.usd','asset.json'): shutil.copy2(a.asset_dir/name,a.destination/'asset'/name)
    (a.destination/'revision.json').write_text(json.dumps(dict(commit=revision),indent=2)+'\n')
    print(a.destination)

if __name__=='__main__':main()
