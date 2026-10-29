"""Same-run recovery, with an explicit random-initialization origin in every save."""
import os
import random
import numpy as np
import torch
from .curriculum_runner import CurriculumRunner


class ScratchRunner(CurriculumRunner):
    def save(self,path,infos=None):
        from .continuous_task import ensure
        saved=self.alg.save()
        saved.update(iter=self.current_learning_iteration,infos=infos,scratch_origin=self.scratch_origin,
            task_state=dict(curriculum=ensure(self.env.env).curriculum_state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),physical_episode_restore=False))
        tmp=str(path)+'.tmp';torch.save(saved,tmp);os.replace(tmp,path)
        self.logger.save_model(path,self.current_learning_iteration)

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        saved=torch.load(path,weights_only=False,map_location='cpu')
        if saved.get('scratch_origin')!=self.scratch_origin:
            raise ValueError('Cross-run initialization is forbidden; recovery requires the same scratch origin')
        forbidden=('parent_balance.','balance_reflex.','adapter.','rise_adapter.','yaw_adapter.')
        if any(k.startswith(forbidden) for k in saved['actor_state_dict']):
            raise ValueError('Historical parent/adapter weights are not a scratch policy')
        del saved
        return super().load(path,load_cfg,strict,map_location)
