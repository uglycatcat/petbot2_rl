"""Independent capability curricula; deployment dynamics remain v25.

Only training requests and progression statistics change. Resets still run the
ordinary controller into the chosen entry state. Motion trials end at a command
change, failure or landing request, independently of later landing success.
"""
import torch
from .capability import CapabilityProgress
from .full_range import project_command
from .moving import slew

SPEED=(.05,.10,.15,.20,.30,.45,.60,.75)
YAW=(.15,.30,.50,.75,1.,1.5,3.,6.,15.)
FRAME=((- .2,.2),(-.3,.3),(-.4,.4),(-.5,.5),(-.65,.65),(-.8,.9),(-1.,1.2))
ARCS=((.10,.30),(.15,.50),(.20,.75),(.30,1.),(.30,1.5),(.45,2.),(.60,3.),(.60,6.),(.75,15.))
AXES=('frame','takeover','straight','turn','arc')
SIDES=(2,2,2,2,4)
CONTRACT=dict(version='split_capability_v1',axes=list(AXES),sides=list(SIDES),
    initial_levels=[0,2,2,2,1],speed=list(SPEED),yaw=list(YAW),
    frame=[list(x) for x in FRAME],arcs=[list(x) for x in ARCS],
    cohorts=['retention']*3+['frame','takeover','straight','straight','turn','arc','exploration'],
    command_amplitude='uniform_zero_to_limit_with_20percent_endpoint',
    gentle_request_fraction=.25,gentle_request_slew=[.15,.75],hold_seconds=[4.,6.],
    steady_delay_seconds=.5,minimum_steady_seconds=3.,minimum_per_side=32,
    promotion_windows=2,rise_success=.95,motion_success=.85,regression_below=.60,
    tolerance=[.04,.15],relative_tolerance=.20,motion_good_fraction=.70,
    motion_trial='held_target_until_next_command_or_landing_or_failure',
    physical_state_assignment=False,execution_limits_changed=False)


class SplitCurriculumProgress(CapabilityProgress):
    def __init__(self,*args,adaptive_curriculum=False,evaluating=False,**kwargs):
        # Keep the old progression disabled, including its terminal accounting.
        super().__init__(*args,adaptive_curriculum=False,evaluating=evaluating,**kwargs)
        self.split_enabled=adaptive_curriculum and not evaluating
        self.curriculum_enabled=self.split_enabled
        self.split_levels=list(CONTRACT['initial_levels'])
        self.split_streak=[0]*5
        self.split_history=[]
        self.split_trials=torch.zeros(5,4,device=self.device)
        self.split_success=torch.zeros_like(self.split_trials)
        self.split_at_reset=torch.zeros(self.n,5,device=self.device,dtype=torch.long)
        self.entry_recorded=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.entry_goal=torch.zeros(self.n,2,device=self.device)  # requested hip, speed
        self.request_target=torch.zeros_like(self.raw)
        self.request_slew=torch.zeros_like(self.raw)
        self.segment_open=torch.zeros_like(self.entry_recorded)
        self.segment_axis=torch.full((self.n,),-1,device=self.device,dtype=torch.long)
        self.segment_side=torch.zeros_like(self.segment_axis)
        self.segment_level=torch.zeros_like(self.segment_axis)
        self.segment_ready=torch.zeros(self.n,device=self.device)
        self.segment_count=torch.zeros(self.n,device=self.device)
        self.segment_good=torch.zeros_like(self.segment_count)
        self.segment_error=torch.zeros_like(self.raw)
        self.category_samples=torch.zeros(9,device=self.device)

    def _limits(self,ids):
        cap=self.raw.new_tensor([self.speed_limit,self.yaw_limit])
        limits=self.raw.new_tensor([.15,.5]).expand(len(ids),2).clone()
        cohort=self.cohort[ids]
        balancing=self.phase[ids]==2
        # Easy entry for motion/pose specialists. Only takeover and exploration
        # enlarge entry speed; turn commands are allowed in the retention set.
        limits[(cohort>=3)&~balancing]=limits.new_tensor([.05,0.])
        levels=self.split_at_reset[ids]
        entry=(cohort==4)&~balancing
        limits[entry,0]=self.raw.new_tensor(SPEED)[levels[entry,1]]
        straight=((cohort==5)|(cohort==6))&balancing
        limits[straight,0]=SPEED[self.split_levels[2]];limits[straight,1]=0.
        turn=(cohort==7)&balancing
        limits[turn,0]=0.;limits[turn,1]=YAW[self.split_levels[3]]
        arc=(cohort==8)&balancing
        arc_cap=self.raw.new_tensor(ARCS[self.split_levels[4]])
        mastered=self.raw.new_tensor([SPEED[self.split_levels[2]],YAW[self.split_levels[3]]])
        limits[arc]=torch.minimum(arc_cap,mastered)
        limits[cohort==9]=cap
        return torch.minimum(limits,cap)

    def sample_commands(self,ids):
        if not self.split_enabled:
            return super().sample_commands(ids)
        self._finish_segments(ids,failed=False)
        count=self.command_count[ids]
        logical=ids//2 if self.mirror_training else ids
        cohort=self.cohort[ids];balancing=self.phase[ids]==2
        category=(logical+count)%9
        seq=(count+3*(logical%2))%6
        straight=(cohort==5)|(cohort==6)
        category=torch.where(straight&balancing,category.new_tensor([0,0,8,1,1,8])[seq],category)
        category=torch.where((cohort==7)&balancing,category.new_tensor([2,2,8,3,3,8])[seq],category)
        category=torch.where((cohort==8)&balancing,4+(logical+count)%4,category)
        category=torch.where((cohort==8)&balancing&(count%5==4),8,category)
        # Pose specialists frequently start stationary. Other specialists enter
        # straight ahead/backwards, without also increasing turn difficulty.
        entry=(cohort>=3)&~balancing
        category=torch.where(entry,(logical+self.generation[ids])%2,category)
        category=torch.where((cohort==3)&~balancing&(self.uniform(ids,32)<.5),8,category)
        limit=self._limits(ids)
        amplitude=torch.stack((self.uniform(ids,0),self.uniform(ids,1)),-1)
        endpoint=self.uniform(ids,15)<.20
        amplitude[endpoint]=1.
        target=amplitude*limit
        reverse=(category==1)|(category==6)|(category==7)
        target[:,0]*=torch.where(reverse,-1.,1.)
        target[(category==2)|(category==3)|(category==8),0]=0.
        target[(category==0)|(category==1)|(category==8),1]=0.
        target[:,1]*=torch.where((category==3)|(category==5)|(category==7),-1.,1.)
        if self.mirror_training:target[:,1]*=torch.where(ids%2==0,1.,-1.)
        # One third of requests keep direction and make a small speed change;
        # the other requests retain full interior and endpoint coverage.
        small=balancing&(count%3==1)&(target.sign()==self.request_target[ids].sign()).all(-1)&(category!=8)
        bounded=self.request_target[ids]+(target-self.request_target[ids]).clamp(-.2*limit,.2*limit)
        target=torch.where(small[:,None],bounded,target)
        target=project_command(target)
        self.request_target[ids]=target
        gentle=balancing&(self.uniform(ids,33)<.25)
        rate=target.new_tensor([.6,6.]).expand(len(ids),2).clone()
        rate[gentle]=rate.new_tensor([.15,.75])
        self.request_slew[ids]=torch.where(gentle[:,None],rate,0.)
        # Immediate requests still pass through the original execution slew.
        self.raw[ids]=torch.where(gentle[:,None],self.raw[ids],target)
        ramp=((target-self.command[ids]).abs()/rate).amax(-1)
        hold=4.+2.*self.uniform(ids,2)
        self.next_command[ids]=self.since[ids]+ramp+hold
        self.segment_ready[ids]=self.since[ids]+ramp+.5
        self.mode[ids]=torch.where(category==8,5,torch.where(category>=4,4,category))
        self.command_count[ids]+=1
        actual=torch.where(target[:,0]<0,1,0)
        actual=torch.where(target[:,0]==0,torch.where(target[:,1]<0,3,2),actual)
        actual=torch.where((target!=0).all(-1),4+2*(target[:,0]<0).long()+(target[:,1]<0).long(),actual)
        actual=torch.where((target==0).all(-1),8,actual)
        self.category_samples+=torch.bincount(actual,minlength=9)
        axis=torch.where(straight,2,torch.where(cohort==7,3,torch.where(cohort==8,4,-1)))
        side=torch.where(axis==2,(target[:,0]<0).long(),(target[:,1]<0).long())
        side=torch.where(axis==4,2*(target[:,0]<0).long()+(target[:,1]<0).long(),side)
        self.segment_axis[ids]=axis;self.segment_side[ids]=side
        self.segment_level[ids]=self.segment_level.new_tensor(self.split_levels)[axis.clamp_min(0)]
        edge=torch.where(axis==2,target[:,0].abs()>=.75*limit[:,0],target[:,1].abs()>=.75*limit[:,1])
        arc_edge=(target.abs()>=.75*project_command(limit)).all(-1)
        arc_unclipped=bool(ARCS[self.split_levels[4]][0]<=SPEED[self.split_levels[2]] and
                          ARCS[self.split_levels[4]][1]<=YAW[self.split_levels[3]])
        edge=torch.where(axis==4,arc_edge&arc_unclipped,edge)
        self.segment_open[ids]=balancing&(axis>=2)&edge&(target.abs().sum(-1)>0)
        for value in (self.segment_count,self.segment_good,self.segment_error):value[ids]=0.

    def reset(self,ids,pos,heading):
        if not self.split_enabled:
            return super().reset(ids,pos,heading)
        self.split_at_reset[ids]=self.split_at_reset.new_tensor(self.split_levels)
        self.segment_open[ids]=False
        self.entry_recorded[ids]=False
        # MovingProgress.reset dispatches sample_commands before CycleProgress
        # clears the previous phase, so clear it before choosing entry requests.
        self.phase[ids]=0
        super().reset(ids,pos,heading)
        bounds=self.raw.new_tensor(FRAME[0]).expand(len(ids),2).clone()
        focus=self.cohort[ids]==3
        bounds[focus]=self.raw.new_tensor(FRAME)[self.split_at_reset[ids[focus],0]]
        bounds[self.cohort[ids]==9]=bounds.new_tensor(FRAME[-1])
        lo=bounds[:,0].clamp_min(self.frame_range[0]);hi=bounds[:,1].clamp_max(self.frame_range[1])
        self.frame[ids]=lo+(hi-lo)*self.uniform(ids,3)
        endpoint=focus&(self.uniform(ids,34)<.2)
        self.frame[ids]=torch.where(endpoint,torch.where(self.uniform(ids,35)<.5,lo,hi),self.frame[ids])
        self.destination[ids]=0.
        self.entry_goal[ids]=torch.stack((self.frame[ids],self.raw[ids,0]),-1)

    def before_step(self):
        if self.split_enabled:
            ids=torch.where(self.active&~self.external&(self.since>=self.next_command))[0]
            if len(ids):self.sample_commands(ids)
            gentle=(self.phase==2)&~self.external&(self.request_slew[:,0]>0)
            self.raw[gentle]=project_command(slew(self.raw[gentle],self.request_target[gentle],self.request_slew[gentle]*self.dt))
        super().before_step()

    def update(self,s,fixed):
        m=super().update(s,fixed)
        if self.split_enabled:
            entered=m['entered_balance']
            self.next_command[entered]=self.since[entered]+2.
            steady=self.segment_open&(self.phase==2)&(self.since>=self.segment_ready)
            error=self.motion-self.request_target
            tolerance=self.raw.new_tensor([.04,.15])+.2*self.request_target.abs()
            good=(error.abs()<=tolerance).all(-1)&m['support']
            self.segment_count+=steady
            self.segment_good+=steady&good
            self.segment_error+=error*steady[:,None]
            # An entry success is recorded when it actually settles, without
            # waiting for a later independent motion/landing failure.
            entry_end=(self.rise_settled|m['failure']|m['completed'])&~self.entry_recorded
            self.entry_recorded|=entry_end
            for axis,cohort in ((0,3),(1,4)):
                ids=torch.where(entry_end&(self.cohort==cohort))[0]
                if len(ids):
                    goal=self.entry_goal[ids,axis]
                    side=(goal<0).long()
                    level=self.split_at_reset[ids,axis]
                    if axis==0:
                        bounds=self.raw.new_tensor(FRAME)[level]
                        limit=torch.where(goal<0,-bounds[:,0],bounds[:,1])
                    else:limit=self.raw.new_tensor(SPEED)[level]
                    edge=goal.abs()>=.75*limit
                    actual=self.captured[ids,1 if axis==0 else 0]*goal.sign()
                    success=self.rise_settled[ids]&(actual>=.65*limit)
                    self._record(axis,side[edge],level[edge],success[edge])
            ended=m['failure']|m['entered_landing']|m['completed']
            ids=torch.where(ended&self.segment_open)[0]
            if len(ids):self._finish_segments(ids,failed=m['failure'][ids])
        m.update(split_curriculum_levels=self.split_at_reset.clone(),
                 training_request_target=self.request_target.clone())
        return m

    def _finish_segments(self,ids,failed):
        opened=self.segment_open[ids]
        if not opened.any():return
        failed=torch.as_tensor(failed,device=self.device,dtype=torch.bool).expand(len(ids))
        count=self.segment_count[ids]
        enough=count*self.dt+1e-5>=3.
        tolerance=self.raw.new_tensor([.04,.15])+.2*self.request_target[ids].abs()
        bias=self.segment_error[ids].abs()/count.clamp_min(1)[:,None]
        success=enough&~failed&(self.segment_good[ids]/count.clamp_min(1)>=.7)&(bias<=tolerance).all(-1)
        # A short aborted segment is a failure; an intentionally truncated
        # nonfailed segment carries insufficient evidence and is not promoted.
        eligible=opened&(enough|failed)
        for axis in (2,3,4):
            selected=eligible&(self.segment_axis[ids]==axis)
            if selected.any():self._record(axis,self.segment_side[ids[selected]],self.segment_level[ids[selected]],success[selected])
        self.segment_open[ids]=False

    def _record(self,axis,side,level,success):
        eligible=level==self.split_levels[axis]
        for i in range(SIDES[axis]):
            mask=eligible&(side==i)
            self.split_trials[axis,i]+=mask.sum()
            self.split_success[axis,i]+=(mask&success).sum()
        trials=self.split_trials[axis,:SIDES[axis]]
        if trials.min()<32:return
        rate=float((self.split_success[axis,:SIDES[axis]]/trials).min())
        threshold=.95 if axis<2 else .85
        self.split_streak[axis]=self.split_streak[axis]+1 if rate>=threshold else 0
        previous=self.split_levels[axis]
        maximum=(len(FRAME),len(SPEED),len(SPEED),len(YAW),len(ARCS))[axis]-1
        if self.split_streak[axis]>=2:
            self.split_levels[axis]=min(maximum,previous+1);self.split_streak[axis]=0
        elif rate<.6:self.split_levels[axis]=max(0,previous-1)
        self.split_history.append(dict(axis=AXES[axis],previous=previous,level=self.split_levels[axis],
            worst_rate=rate,trials=trials.tolist(),success=self.split_success[axis,:SIDES[axis]].tolist()))
        self.split_trials[axis]=0.;self.split_success[axis]=0.

    def curriculum_state_dict(self):
        return dict(version=2,design='split_capability_v1',levels=list(self.split_levels),
            streak=list(self.split_streak),history=self.split_history,
            trials=self.split_trials.cpu(),success=self.split_success.cpu(),
            generation=self.generation.cpu(),category_samples=self.category_samples.cpu())

    def load_curriculum_state_dict(self,data,restore_streams=True):
        assert data['version']==2 and data['design']=='split_capability_v1' and len(data['levels'])==5
        self.split_levels=list(data['levels']);self.split_streak=list(data['streak']);self.split_history=list(data['history'])
        self.split_trials.copy_(data['trials']);self.split_success.copy_(data['success'])
        self.category_samples.copy_(data['category_samples'])
        if restore_streams:
            assert data['generation'].shape==self.generation.shape
            self.generation.copy_(data['generation'])
