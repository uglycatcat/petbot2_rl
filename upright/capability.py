"""v25: continuous preparation, symmetric commands and two measured curricula.

Preparation is the physical, not-yet-lifted portion of phase 1. The existing
three actuator outputs already control common hip, common wheel and yaw.
There is no mandatory stop, prescribed hip pose, teleport or added network.
"""
import math
import torch
from .full_range import FullRangeProgress, project_command
from .moving import MovingProgress

CONTRACT=dict(version='continuous_minimum_adjustment_v1', rise_timeout=8.,
    preparation='phase1_before_lift_seen_policy_actuation', landing_frame=0.,
    landing_other_changes=False, settled_rise_seconds=3.,
    preparation_cost=dict(time=.03,speed_change=.06,hip_change=.03),
    ordinary='independent_source_limits_2ms_pi_phase0',
    balance_common_mapping='clip(latent/.99,-1,1)*25',
    tracking='continuous_ema_120ms_no_half_speed_plateau',
    curriculum=dict(version='two_axes_v1',retention=.3,boundary=.6,exploration=.1,
        minimum_per_side=32,promotion_windows=2,rise_success=.95,motion_success=.85,
        regression_below=.60, motion_tolerance=[.04,.15], relative_tolerance=.20,
        motion_fraction=.70, independent_evaluation=False))
SPEED=(.15,.30,.45,.60,.75)
YAW=(.5,1.5,3.,6.,15.)
FRAME=((- .2,.2),(-.35,.35),(-.6,.6),(-.8,.9),(-1.,1.2))

def speed_score(motion,command):
    # No flat tolerance band: even .74 -> .75 continues to improve the score.
    scale=command.new_tensor([.06,.20])+.20*command.abs()
    score=1./(1.+((motion-command)/scale).square())
    return .25*score.sum(-1)+.5*score.prod(-1)

def balance_common(latent):
    scaled=(latent/.99).clamp(-1.,1.)*25.
    return torch.where(latent.abs()>=.99,latent.sign()*25.,scaled)

class CapabilityProgress(FullRangeProgress):
    rise_timeout=8.
    def __init__(self,*args,mirror_training=False,adaptive_curriculum=False,evaluating=False,evaluation_pattern='mixed',**kwargs):
        self.mirror_training=mirror_training
        self.adaptive_curriculum=adaptive_curriculum and not evaluating
        self.evaluating=evaluating
        self.evaluation_pattern=evaluation_pattern
        super().__init__(*args,**kwargs)
        self.levels=[0,0];self.streak=[0,0];self.course_history=[]
        self.course_trials=torch.zeros(2,2,device=self.device)
        self.course_success=torch.zeros_like(self.course_trials)
        ids=torch.arange(self.n,device=self.device)
        self.cohort=(ids//2 if mirror_training else ids)%10
        if mirror_training and self.training_mixture!='off':self.training_level=(ids//2)*5//((self.n+1)//2)
        self.course_at_reset=torch.zeros(self.n,2,device=self.device,dtype=torch.long)
        self.captured=torch.zeros(self.n,2,device=self.device) # actual vx, common hip
        self.preparation_elapsed=torch.zeros(self.n,device=self.device)
        self.preparation_speed_change=torch.zeros(self.n,device=self.device)
        self.preparation_hip_change=torch.zeros(self.n,device=self.device)
        self.rise_settled=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.motion_count=torch.zeros(self.n,2,device=self.device)
        self.motion_good=torch.zeros_like(self.motion_count)
        self.terminal_seen=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.curriculum_enabled=self.adaptive_curriculum

    def uniform(self,ids,stream):
        if not self.mirror_training:return super().uniform(ids,stream)
        # Asynchronous resets never couple another environment's RNG. A pair
        # has identical streams only while generation and command count match.
        paired=ids//2
        x=(paired*1103515245+self.generation[ids]*12345+self.command_count[ids]*214013+self.seed*2531011+stream*134775813)&0x7fffffff
        x=((x^(x>>16))*1103515245+12345)&0x7fffffff
        x=((x^(x>>13))*214013+2531011)&0x7fffffff
        return x.to(torch.float32)/2147483648.

    def sampling_limits(self,ids):
        limits=super().sampling_limits(ids)
        if not self.adaptive_curriculum:return limits
        boundary=(self.cohort[ids]>=3)&(self.cohort[ids]<9)
        retention=self.cohort[ids]<3
        level=torch.where(self.phase[ids]==2,self.course_at_reset[ids,1],self.course_at_reset[ids,0])
        v=self.raw.new_tensor(SPEED)[level];w=self.raw.new_tensor(YAW)[level]
        result=self.raw.new_tensor([self.speed_limit,self.yaw_limit]).expand(len(ids),2).clone()
        result[boundary]=torch.minimum(result[boundary],torch.stack((v,w),-1)[boundary])
        result[retention]=torch.minimum(result[retention],result.new_tensor([.15,.5]))
        return result

    def sample_commands(self,ids):
        # Eight classes include both forward arcs. Reverse arcs are sampled
        # too; exact yaw reflection is applied to paired environments in B/C.
        logical=ids//2 if self.mirror_training else ids
        mode=(logical+self.command_count[ids])%8
        v=self.uniform(ids,0);w=self.uniform(ids,1)
        limits=self.sampling_limits(ids)
        # Explicit ceiling samples ensure the task includes +/- .75, rather
        # than expecting a continuous random draw to hit an endpoint.
        boundary=self.uniform(ids,15)<.25
        v=torch.where(boundary,torch.ones_like(v),.25+.75*v)*limits[:,0]
        w=torch.where(boundary,torch.ones_like(w),.25+.75*w)*limits[:,1]
        reverse=(mode==1)|((mode>=6)&(self.uniform(ids,17)<.5))
        v=torch.where(reverse,-v,v)
        v=torch.where((mode==2)|(mode==3)|(mode==5),0.,v)
        w=torch.where((mode==0)|(mode==1)|(mode==5),0.,w)
        w=torch.where((mode==3)|(mode==7),-w,w)
        if self.mirror_training:w*=torch.where(ids%2==0,1.,-1.)
        self.raw[ids]=project_command(torch.stack((v,w),-1))
        if self.evaluation_pattern in ('paired_speed','paired_arcs'):self.raw[ids]=0.
        if self.evaluation_pattern=='rise_grid':
            grid=self.raw.new_tensor([-.75,-.3,0.,.3,.75])
            self.raw[ids,0]=grid[(logical//5)%5];self.raw[ids,1]=0.
        self.mode[ids]=torch.where(mode>=6,4,mode)
        self.command_count[ids]+=1
        ramp=((self.raw[ids]-self.command[ids]).abs()/self.raw.new_tensor([.6,6.])).amax(-1)
        self.next_command[ids]=self.since[ids]+ramp+3.+3.*self.uniform(ids,2)

    def reset(self,ids,pos,heading):
        # At construction all task buffers exist before the first reset.
        self.course_at_reset[ids]=torch.tensor(self.levels,device=self.device)
        super().reset(ids,pos,heading)
        self.destination[ids]=0.
        if self.adaptive_curriculum:
            bounds=self.raw.new_tensor(FRAME)[self.course_at_reset[ids,0]]
            retention=self.cohort[ids]<3;full=self.cohort[ids]>=9
            bounds[retention]=bounds.new_tensor(FRAME[0]);bounds[full]=bounds.new_tensor(FRAME[-1])
            lo=bounds[:,0].clamp_min(self.frame_range[0]);hi=bounds[:,1].clamp_max(self.frame_range[1])
            self.frame[ids]=lo+(hi-lo)*self.uniform(ids,3)
        if self.evaluation_pattern=='rise_grid':
            grid=self.frame.new_tensor([-1.,-.5,0.,.6,1.2]);self.frame[ids]=grid[(ids//2)%5]
        elif self.evaluation_pattern in ('paired_speed','paired_arcs'):self.frame[ids]=0.
        for x in (self.captured,self.preparation_elapsed,self.preparation_speed_change,
                  self.preparation_hip_change,self.rise_settled,self.motion_count,self.motion_good,self.terminal_seen):x[ids]=0

    def before_step(self):
        if self.evaluation_pattern in ('paired_speed','paired_arcs'):
            ids=torch.arange(self.n,device=self.device);self.external[:]=True
            balancing=self.phase==2;seconds=self.phase_time
            if self.evaluation_pattern=='paired_speed':
                direction=torch.where((ids//2)%2==0,1.,-1.)
                target=torch.where((seconds>=3.)&(seconds<13.),.75,torch.where((seconds>=17.)&(seconds<27.),-.75,0.))
                self.raw[:,0]=torch.where(balancing,target*direction,0.);self.raw[:,1]=0.
            else:
                level=((seconds-3.).clamp_min(0.)/10.).long().clamp_max(3)
                v=self.raw.new_tensor([.15,.3,.45,.6])[level];w=self.raw.new_tensor([.5,1.5,3.,3.])[level]
                direction=torch.where(ids%2==0,1.,-1.);reverse=torch.where((ids//2)%2==0,1.,-1.)
                self.raw[:]=project_command(torch.stack((v*reverse,w*direction),-1))*((seconds>=3.)&balancing)[:,None]
        self.requested[self.external]=self.raw[self.external]
        super().before_step()
        # Ordinary drive's requested acceleration is not a policy capability.
        self.command[~self.active]=project_command(self.raw[~self.active])

    def request_mode(self,mode,frame=0.,ids=None):
        super().request_mode(mode,0.,ids)

    def update(self,s,fixed):
        m=super().update(s,fixed)
        switch=m['switched']
        self.captured[switch,0]=s['forward'][switch]
        self.captured[switch,1]=s['q'][switch,3:].mean(-1)
        preparing=(self.phase==1)&~self.lift_seen
        dv=(s['forward']-self.captured[:,0]).abs()
        dq=(s['q'][:,3:].mean(-1)-self.captured[:,1]).abs()
        self.preparation_elapsed+=preparing*self.dt
        self.preparation_speed_change=torch.maximum(self.preparation_speed_change,preparing*dv)
        self.preparation_hip_change=torch.maximum(self.preparation_hip_change,preparing*dq)
        m['preparation_cost']=preparing*(.03+.06*dv/.75+.03*dq/2.2)
        new_settled=(self.phase==2)&(self.phase_time>=3.)&~self.rise_settled&~m['failure']
        self.rise_settled|=new_settled
        m['settled_rise_reward']=new_settled.float()/self.dt
        m['track_reward']=(self.phase==2)*m['support']*speed_score(self.motion,self.command)
        steady=(self.phase==2)&(self.phase_time>=2.)
        tolerance=self.raw.new_tensor([.04,.15])+.2*self.command.abs()
        good=((self.motion-self.command).abs()<=tolerance).all(-1)&m['support']
        # Straight forward/reverse occupy the two side buckets; turns use yaw.
        side=torch.where(self.command[:,1].abs()>.015,self.command[:,1],self.command[:,0])<0
        for i in range(2):
            mask=steady&(side==bool(i));self.motion_count[:,i]+=mask;self.motion_good[:,i]+=mask&good
        m.update(preparing=preparing,preparation_elapsed=self.preparation_elapsed.clone(),
            preparation_speed_change=self.preparation_speed_change.clone(),preparation_hip_change=self.preparation_hip_change.clone(),
            initial_motion_frame=self.captured.clone(),rise_settled=self.rise_settled.clone(),
            curriculum_levels=self.course_at_reset.clone())
        terminal=m['failure']|m['completed']|(self.active&(self.since>=self.hold_range[1]+18.))
        new=terminal&~self.terminal_seen;self.terminal_seen|=terminal
        if self.adaptive_curriculum and new.any():self.record_outcomes(new,m)
        return m

    def record_outcomes(self,new,m):
        boundary=new&(self.cohort>=3)&(self.cohort<9)
        for axis in range(2):
            # Do not mix completed older-tier episodes into the current window.
            eligible=boundary&(self.course_at_reset[:,axis]==self.levels[axis])
            for side in range(2):
                if axis==0:
                    direction=(self.captured[:,0]<0).long()
                    mask=eligible&(direction==side);success=self.rise_settled
                else:
                    mask=eligible&self.rise_settled&(self.motion_count[:,side]>=8)
                    success=m['completed']&(self.motion_good[:,side]/self.motion_count[:,side].clamp_min(1)>=.70)
                self.course_trials[axis,side]+=mask.sum()
                self.course_success[axis,side]+=(mask&success).sum()
            if self.course_trials[axis].min()>=32:
                rate=float((self.course_success[axis]/self.course_trials[axis]).min())
                self.streak[axis]=self.streak[axis]+1 if rate>=(.95 if axis==0 else .85) else 0
                previous=self.levels[axis]
                if self.streak[axis]>=2:self.levels[axis]=min(4,previous+1);self.streak[axis]=0
                elif rate<.6:self.levels[axis]=max(0,previous-1)
                self.course_history.append(dict(axis=axis,previous=previous,level=self.levels[axis],worst_rate=rate,
                    trials=self.course_trials[axis].tolist(),success=self.course_success[axis].tolist()))
                self.course_trials[axis]=0;self.course_success[axis]=0

    def curriculum_state_dict(self):
        return dict(version=1,levels=self.levels.copy(),streak=self.streak.copy(),history=self.course_history,
            trials=self.course_trials.cpu(),success=self.course_success.cpu(),generation=self.generation.cpu())

    def load_curriculum_state_dict(self,data,restore_streams=True):
        assert data['version']==1 and len(data['levels'])==2
        self.levels=list(data['levels']);self.streak=list(data['streak']);self.course_history=list(data['history'])
        self.course_trials.copy_(data['trials']);self.course_success.copy_(data['success'])
        if restore_streams:
            assert data['generation'].shape==self.generation.shape
            self.generation.copy_(data['generation'])
