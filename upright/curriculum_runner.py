"""Checkpoint the task curriculum and RNG alongside RSL's optimizer.

A forced restart starts fresh physical episodes; it does not pretend to restore
PhysX contacts. Completed update indices are advanced once, never replayed.
"""
import os
import random
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

class CurriculumRunner(OnPolicyRunner):
    def save(self,path,infos=None):
        from .continuous_task import ensure
        saved=self.alg.save()
        saved.update(iter=self.current_learning_iteration,infos=infos,task_state=dict(
            curriculum=ensure(self.env.env).curriculum_state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            physical_episode_restore=False))
        temporary=str(path)+'.tmp';torch.save(saved,temporary);os.replace(temporary,path)
        self.logger.save_model(path,self.current_learning_iteration)

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        from .continuous_task import ensure
        loaded=torch.load(path,weights_only=False,map_location=map_location)
        resume=self.alg.load(loaded,load_cfg,strict)
        state=loaded['task_state'];p=ensure(self.env.env)
        if resume:self.current_learning_iteration=int(loaded['iter'])+1
        p.load_curriculum_state_dict(state['curriculum'],restore_streams=not p.evaluating)
        if not p.evaluating:
            # Fresh physical episodes use the next counter generation.
            self.env.env.reset()
            torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        return loaded.get('infos')
