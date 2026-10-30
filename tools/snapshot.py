"""Create a pinned shallow checkout with the declared relative model layout."""
import argparse
import os
from pathlib import Path
import subprocess

SOURCE = Path(__file__).resolve().parents[1]
REPOSITORY = SOURCE.parents[1]
RUNS = REPOSITORY/'demo_project/runs'


def create(name):
    if not name or Path(name).name != name:
        raise ValueError('Use a single unambiguous new experiment name')
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Commit the fixed source before snapshot creation')
    destination = RUNS/name/'workspace/demo_project/upright_rl_demo'
    if (RUNS/name).exists():
        raise FileExistsError('Experiment exists; inspect/resume rather than duplicate it')
    destination.parent.mkdir(parents=True)
    subprocess.run(['git','clone','--quiet','--depth','1','--branch','demoproj/upright_rl_demo',
                    SOURCE.as_uri(),str(destination)],check=True)
    workspace = destination.parents[1]
    (workspace/'robot_description').symlink_to(os.path.relpath(REPOSITORY/'robot_description',workspace))
    commit = subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    expected = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if commit != expected: raise RuntimeError('Snapshot commit mismatch')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('name')
    print(create(parser.parse_args().name))
