"""Keep policy/physics work off the window thread; publish coherent render copies."""
import json
import queue
import threading
import time


class SimulationWorker:
    def __init__(self,sim,trace=None):
        self.sim=sim;self.trace=trace;self.commands=queue.Queue();self.lock=threading.Lock()
        self.events=[];self.error=None;self.began=time.monotonic();self.steps=0;self.compute_seconds=0.;self.active_seconds=0.
        self._publish();self.thread=threading.Thread(target=self._run,name='step-physics',daemon=True)

    def _publish(self):
        # Only this worker owns the live MuJoCo data. The viewer gets copies.
        state=(self.sim.snapshot(),self.sim.data.qpos.copy(),self.sim.data.qvel.copy())
        with self.lock:self.latest=state

    def read(self):
        with self.lock:return self.latest

    def start(self):self.thread.start()

    def key(self,key):self.commands.put(key)

    def stop(self):
        self.commands.put(None);self.thread.join()
        if self.error is not None:raise RuntimeError('Simulation worker failed') from self.error

    def _run(self):
        deadline=time.monotonic()
        try:
            while True:
                interval_began=time.monotonic()
                timeout=max(0.,deadline-time.monotonic()) if self.sim.state=='RUNNING' else .1
                try:
                    key=self.commands.get(timeout=timeout)
                except queue.Empty:
                    if self.sim.state!='RUNNING':continue
                else:
                    if key is None:return
                    if self.sim.key(key):
                        self.events.append(dict(key=key,wall_elapsed=time.monotonic()-self.began,after=self.sim.snapshot()))
                        self._publish();deadline=time.monotonic()
                    continue
                began=time.monotonic();snap=self.sim.step()
                self.compute_seconds+=time.monotonic()-began;self.steps+=1
                if self.trace:self.trace.write(json.dumps(snap)+'\n')
                self._publish()
                self.active_seconds+=time.monotonic()-interval_began
                deadline+=self.sim.metadata['policy_dt']
                # A slow machine runs slower in wall time; never skip physical
                # substeps or change policy dt to manufacture a higher FPS.
                if time.monotonic()-deadline>.1:deadline=time.monotonic()
        except BaseException as exc:self.error=exc
