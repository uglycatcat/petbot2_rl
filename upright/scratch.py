"""Bootstrap a complete upright policy without pretrained networks or datasets."""
import torch
from .split_curriculum import SplitCurriculumProgress

CONTRACT=dict(version='scratch_curriculum_v1',pretrained_models=[],
    policy='single_trainable_mlp_20ms',history='five_58_feature_frames_20ms',
    ordinary_and_yaw='explicit_source_control_no_learned_parent',
    bootstrap=[dict(frame=[0.,0.],motion=[0.,0.],hold=4.),
               dict(frame=[-.1,.1],motion=[.05,.15],hold=8.)],
    minimum_bootstrap_trials=128,bootstrap_success=.8,bootstrap_windows=2,
    later='split_capability_v1',normalization='fresh_running_statistics',
    initialization='random_hidden_weights_physical_default_output_bias',
    transfer_from_other_runs=False)


class ScratchProgress(SplitCurriculumProgress):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.bootstrap_stage=0
        self.bootstrap_trials=0
        self.bootstrap_success=0
        self.bootstrap_streak=0
        self.bootstrap_history=[]
        self.bootstrap_at_reset=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.bootstrap_seen=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.split_levels=[0,0,0,0,0]

    def _limits(self,ids):
        limits=super()._limits(ids)
        from .split_curriculum import SPEED,YAW
        retention=self.cohort[ids]<3
        limits[retention]=torch.minimum(limits[retention],limits.new_tensor(
            [SPEED[self.split_levels[2]],YAW[self.split_levels[3]]]))
        stage=self.bootstrap_at_reset[ids]
        limits[stage==0]=0.
        limits[stage==1]=torch.minimum(limits[stage==1],limits.new_tensor([.05,.15]))
        return limits

    def reset(self,ids,pos,heading):
        self.bootstrap_at_reset[ids]=self.bootstrap_stage
        self.bootstrap_seen[ids]=False
        super().reset(ids,pos,heading)
        if not self.evaluating:
            if self.bootstrap_stage==0:self.frame[ids]=0.;self.hold_duration[ids]=4.
            elif self.bootstrap_stage==1:
                self.frame[ids]=-.1+.2*self.uniform(ids,3);self.hold_duration[ids]=8.
            self.entry_goal[ids,0]=self.frame[ids]

    def _record(self,*args):
        if self.bootstrap_stage>=2:super()._record(*args)

    def update(self,s,fixed):
        m=super().update(s,fixed)
        if not self.evaluating and self.bootstrap_stage<2:
            terminal=m['failure']|m['completed']|(self.active&(self.since>=self.hold_range[1]+18.))
            new=terminal&~self.bootstrap_seen
            self.bootstrap_seen|=terminal
            eligible=new&(self.bootstrap_at_reset==self.bootstrap_stage)
            self.bootstrap_trials+=int(eligible.sum())
            self.bootstrap_success+=int((eligible&m['completed']&self.rise_settled).sum())
            if self.bootstrap_trials>=128:
                rate=self.bootstrap_success/self.bootstrap_trials
                self.bootstrap_streak=self.bootstrap_streak+1 if rate>=.8 else 0
                previous=self.bootstrap_stage
                if self.bootstrap_streak>=2:self.bootstrap_stage+=1;self.bootstrap_streak=0
                self.bootstrap_history.append(dict(previous=previous,stage=self.bootstrap_stage,
                    trials=self.bootstrap_trials,success=self.bootstrap_success,rate=rate))
                self.bootstrap_trials=self.bootstrap_success=0
        m['bootstrap_stage']=self.bootstrap_at_reset.clone()
        return m

    def curriculum_state_dict(self):
        return dict(version=3,design='scratch_curriculum_v1',split=super().curriculum_state_dict(),
            bootstrap_stage=self.bootstrap_stage,bootstrap_trials=self.bootstrap_trials,
            bootstrap_success=self.bootstrap_success,bootstrap_streak=self.bootstrap_streak,
            bootstrap_history=self.bootstrap_history)

    def load_curriculum_state_dict(self,data,restore_streams=True):
        assert data['version']==3 and data['design']=='scratch_curriculum_v1'
        super().load_curriculum_state_dict(data['split'],restore_streams)
        for key in ('bootstrap_stage','bootstrap_trials','bootstrap_success','bootstrap_streak','bootstrap_history'):
            setattr(self,key,data[key])
