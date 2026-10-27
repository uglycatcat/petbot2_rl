"""A fixed current-parent mean supplies physical feedback during PPO rollouts.

The trainable outer actor still decides every60ms. Its sampled action offset
is held constant while this checkpointed, frozen parent refreshes frame/common
means at20ms. The live PPO actor is never called from the environment substeps.
"""
from .rise_adapter import RiseAdapterModel, RiseAdapterInference

PARENT_FEEDBACK_CONTRACT = dict(version='frozen_full_parent_offset_v1',
    parent='explicit_initialization_checkpoint_after_equivalent_normalization',
    period=.02, gate_after=1.8, frozen=True, clock='outer60ms',
    physical_features_only=True, outer_action='sampled_action_minus_parent_mean',
    refreshed_channels=['frame_position','common_wheel_speed'],
    yaw='unchanged2ms_with_outer60ms_residual', history='four_unchanged60ms_frames')


class ParentFeedbackModel(RiseAdapterModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.parent_balance=RiseAdapterInference(self).eval().requires_grad_(False)

    def initialize_outer(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and missing and all(k.startswith('parent_balance.') for k in missing)

    def snapshot_parent(self):
        self.parent_balance.load_state_dict(RiseAdapterInference(self).state_dict(),strict=True)
        self.parent_balance.eval().requires_grad_(False)
