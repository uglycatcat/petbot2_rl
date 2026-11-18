// Original MuJoCo model, external capped PD, independent data per episode.
// This only moves the existing Python stepping loop into C++; no surrogate dynamics.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

struct Engine {
  mjModel* m;
  int jp[5], jd[5], pa[5], wa[2], wd[2], wb[4], base;
  double kp[5], kd[5], low[5], high[5];
  ~Engine() { mj_deleteModel(m); }
};

extern "C" void* sequence_create(const char* xml, double dt, const int* ids,
                                 const double* config, char* error, int capacity) {
  mjModel* m = mj_loadXML(xml, nullptr, error, capacity);
  if (!m) return nullptr;
  if (dt <= 0 || dt > m->opt.timestep || std::abs(std::round(.02/dt)*dt-.02)>1e-10) {
    std::snprintf(error, capacity, "Invalid integration timestep"); mj_deleteModel(m); return nullptr;
  }
  auto* e = new Engine; e->m=m;
  m->opt.timestep=dt; m->opt.integrator=mjINT_IMPLICITFAST;
  int k=0;
  for (auto* a : {e->jp, e->jd, e->pa}) for(int j=0;j<5;j++) a[j]=ids[k++];
  for (auto* a : {e->wa, e->wd}) for(int j=0;j<2;j++) a[j]=ids[k++];
  for(int j=0;j<4;j++) e->wb[j]=ids[k++];
  e->base=ids[k];
  k=0;
  for(auto* a : {e->kp,e->kd,e->low,e->high}) for(int j=0;j<5;j++) a[j]=config[k++];
  return e;
}
extern "C" void sequence_destroy(void* p) { delete static_cast<Engine*>(p); }

// summary: score, first_success, max_hold, max_up, four_steps, first_invalid,
// terminal_up, terminal_contacts, numerical_warning, max_quality.
// Optional trace: qpos, qvel, up, contact mask, other force, hold, quality.
extern "C" void sequence_rollout(void* handle, const double* initial,
                                 const float* actions, int count, int steps,
                                 int threads, double* summary, double* trace) {
  const auto& e=*static_cast<Engine*>(handle); const mjModel* m=e.m;
  const int decimation=std::lround(.02/m->opt.timestep);
  const int width=m->nq+m->nv+5;
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for(int episode=0;episode<count;episode++) {
    std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(m),mj_deleteData);
    mjData* d=data.get();
    std::copy(initial+episode*m->nq,initial+(episode+1)*m->nq,d->qpos);
    mj_forward(m,d);
    double hold=0,best=0,first=-1,maxup=-2,invalid=-1,qualitymax=0;
    double tail=0,up=-2; int foursteps=0,contacts=0,warning=0;
    std::vector<double> forces(m->nbody*3);
    for(int t=0;t<steps;t++) {
      const float* a=actions+(episode*steps+t)*7;
      double target[5];
      for(int j=0;j<5;j++) target[j]=e.low[j]+double((a[j]+1.f)*.5f)*(e.high[j]-e.low[j]);
      int hits[4]={0,0,0,0}; double other=0;
      for(int sub=0;sub<decimation;sub++) {
        for(int j=0;j<5;j++) {
          double cap=m->actuator_forcerange[e.pa[j]*2+1];
          d->ctrl[e.pa[j]]=std::clamp(e.kp[j]*(target[j]-d->qpos[e.jp[j]])-e.kd[j]*d->qvel[e.jd[j]],-cap,cap);
        }
        for(int j=0;j<2;j++) d->ctrl[e.wa[j]]=double(25.f*a[5+j]);
        mj_step(m,d);
        std::fill(forces.begin(),forces.end(),0.);
        for(int c=0;c<d->ncon;c++) {
          double local[6],world[3]; const auto& contact=d->contact[c];
          mj_contactForce(m,d,c,local); mju_mulMatTVec(world,contact.frame,local,3,3);
          int b1=m->geom_bodyid[contact.geom[0]],b2=m->geom_bodyid[contact.geom[1]];
          for(int j=0;j<3;j++) { forces[b1*3+j]-=world[j]; forces[b2*3+j]+=world[j]; }
        }
        for(int j=0;j<4;j++) hits[j]+=forces[e.wb[j]*3+2]>.05;
        for(int b=1;b<m->nbody;b++) {
          if(std::find(e.wb,e.wb+4,b)!=e.wb+4) continue;
          other=std::max(other,mju_norm3(forces.data()+b*3));
        }
      }
      mj_forward(m,d);
      for(int j=mjWARN_BADQPOS;j<=mjWARN_BADCTRL;j++) warning|=d->warning[j].number>0;
      for(int j=0;j<m->nq;j++) warning|=!std::isfinite(d->qpos[j]);
      for(int j=0;j<m->nv;j++) warning|=!std::isfinite(d->qvel[j]);
      if(invalid<0 && (warning || std::hypot(d->qpos[0],d->qpos[1])>3 || d->qpos[2]<-.1)) invalid=(t+1)*.02;
      float x=d->qpos[4], y=d->qpos[5]; up=1.f-2.f*(x*x+y*y);
      contacts=0;int mask=0;
      for(int j=0;j<4;j++) if(double(hits[j])/decimation>=.8) {contacts++;mask|=1<<j;}
      double vel[6];mj_objectVelocity(m,d,mjOBJ_BODY,e.base,vel,0);
      bool quiet=mju_norm3(vel)<.4 && mju_norm3(vel+3)<.15;
      for(int j=0;j<2;j++) quiet &= std::abs(d->qvel[e.wd[j]])<1.5;
      bool stable=up>std::cos(25.*mjPI/180.) && contacts==4 && quiet && other<.5;
      hold=stable?hold+.02:0.;
      // Fitness uses simultaneous posture and support, with a tail preference.
      // This search bias does not change the strict success criterion above.
      double quality=2*(up+1)+.5*contacts;
      if(invalid<0) {
        best=std::max(best,hold);maxup=std::max(maxup,up);foursteps+=contacts==4;
        qualitymax=std::max(qualitymax,quality);
        if(first<0 && hold>=.5) first=(t+1)*.02;
        if(t>=steps-100) tail+=quality/std::min(steps,100);
      }
      if(trace) {
        double* row=trace+(episode*steps+t)*width;
        std::copy(d->qpos,d->qpos+m->nq,row);std::copy(d->qvel,d->qvel+m->nv,row+m->nq);
        row+=m->nq+m->nv;row[0]=up;row[1]=mask;row[2]=other;row[3]=hold;row[4]=quality;
      }
    }
    double* s=summary+episode*10;
    s[0]=tail+.25*qualitymax+10*std::min(best,.5)+(first>=0?100:0)-(invalid>=0?10:0);
    s[1]=first;s[2]=best;s[3]=maxup;s[4]=foursteps;s[5]=invalid;
    s[6]=up;s[7]=contacts;s[8]=warning;s[9]=qualitymax;
  }
}
