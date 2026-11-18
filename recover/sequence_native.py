"""Bounded CPU batch stepping in the container, using the unchanged native model."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from .contract import POSITION_LOW,POSITION_HIGH
from .model import PROJECT_ROOT,workspace_root,POSITION_JOINTS,POSITION_KP,POSITION_KD,REAR_WHEELS,FRONT_WHEELS


class NativeSequences:
    def __init__(self, physics_dt=.0005, threads=4):
        if not 1<=threads<=4:raise ValueError('This auxiliary search is limited to four CPU threads')
        self.threads=threads
        source=Path(__file__).with_suffix('.cpp')
        package=Path(mujoco.__file__).parent
        library=next(package.glob('libmujoco.so.*'))
        identity=hashlib.sha256(source.read_bytes()+Path(__file__).read_bytes()+mujoco.__version__.encode()).hexdigest()[:20]
        build=PROJECT_ROOT/'artifacts/native_sequence'/identity;build.mkdir(parents=True,exist_ok=True)
        binary=build/'sequence.so'
        if not binary.is_file():
            temporary=binary.with_name(f'sequence.{os.getpid()}.so')
            subprocess.run(['g++','-std=c++17','-O3','-ffp-contract=off','-shared','-fPIC','-fopenmp',
                            '-I'+str(package/'include'),str(source),str(library),'-Wl,-rpath,'+str(package),
                            '-o',str(temporary)],check=True)
            temporary.replace(binary)
        self.library=ct.CDLL(str(binary))
        self.library.sequence_create.argtypes=[ct.c_char_p,ct.c_double,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int]
        self.library.sequence_create.restype=ct.c_void_p
        self.library.sequence_destroy.argtypes=[ct.c_void_p]
        self.library.sequence_rollout.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
        xml=workspace_root()/'robot_description/mjcf/scene.xml'
        model=mujoco.MjModel.from_xml_path(str(xml));self.nq=model.nq;self.nv=model.nv
        ids=np.array([*[model.joint(n).qposadr[0] for n in POSITION_JOINTS],
                      *[model.joint(n).dofadr[0] for n in POSITION_JOINTS],
                      *[model.actuator(n).id for n in POSITION_JOINTS],
                      *[model.actuator(n).id for n in REAR_WHEELS],
                      *[model.joint(n).dofadr[0] for n in REAR_WHEELS],
                      *[model.body(n).id for n in FRONT_WHEELS+REAR_WHEELS],model.body('base_link').id],dtype=np.int32)
        config=np.array([*[POSITION_KP[n] for n in POSITION_JOINTS],*[POSITION_KD[n] for n in POSITION_JOINTS],
                         *POSITION_LOW,*POSITION_HIGH],dtype=np.float64)
        error=ct.create_string_buffer(2048)
        self.handle=self.library.sequence_create(str(xml).encode(),physics_dt,ids.ctypes.data,config.ctypes.data,error,len(error))
        if not self.handle:raise RuntimeError(error.value.decode())
        self.kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest()

    def run(self, initial, actions, record=False):
        initial=np.ascontiguousarray(initial,dtype=np.float64);actions=np.ascontiguousarray(actions,dtype=np.float32)
        if actions.ndim!=3 or actions.shape[2]!=7 or not 1<=actions.shape[1]<=800:raise ValueError('Expected episode x 1..800 control steps x 7 targets')
        count,steps,_=actions.shape
        if count<1 or initial.shape!=(count,self.nq):raise ValueError('Initial state shape mismatch')
        if not np.isfinite(initial).all() or not np.isfinite(actions).all() or np.abs(actions).max()>1:raise ValueError('Nonfinite states or out-of-range actions')
        summary=np.zeros((count,10),dtype=np.float64)
        trace=np.empty((count,steps,self.nq+self.nv+5),dtype=np.float64) if record else None
        self.library.sequence_rollout(self.handle,initial.ctypes.data,actions.ctypes.data,count,steps,self.threads,
                                     summary.ctypes.data,trace.ctypes.data if record else None)
        if not np.isfinite(summary).all() or summary[:,8].any():raise RuntimeError('Numerical failure during native search')
        return summary,trace

    def close(self):
        if self.handle:self.library.sequence_destroy(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
