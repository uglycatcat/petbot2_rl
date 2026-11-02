import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import mujoco
from .runtime import Preview
from .viewer_worker import SimulationWorker


def show(sim,seconds=None,trace=None,capture=None,render_fps=60.):
    import glfw
    if not glfw.init():raise RuntimeError('Cannot open display. On a desktop use a graphical terminal; use --headless for diagnostics.')
    window=glfw.create_window(1100,720,'PetBot Step | Space: start | R: reset',None,None)
    if not window:glfw.terminate();raise RuntimeError('Cannot create OpenGL window')
    glfw.make_context_current(window);glfw.swap_interval(1)
    scene=mujoco.MjvScene(sim.model,maxgeom=2000);context=mujoco.MjrContext(sim.model,mujoco.mjtFontScale.mjFONTSCALE_150)
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
    camera=mujoco.MjvCamera();camera.lookat[:]=[.55,0,.10];camera.distance=1.7;camera.azimuth=130;camera.elevation=-22
    option=mujoco.MjvOption();perturb=mujoco.MjvPerturb();mouse=[0.,0.]
    worker=SimulationWorker(sim,trace);render_data=mujoco.MjData(sim.model)
    def key(win,k,scan,action,mods):
        if action!=glfw.PRESS:return
        if k==glfw.KEY_ESCAPE:glfw.set_window_should_close(win,True);return
        if k in (glfw.KEY_SPACE,glfw.KEY_R):worker.key(k)
    def cursor(win,x,y):
        dx=x-mouse[0];dy=y-mouse[1];mouse[:]=[x,y]
        height=max(1,glfw.get_window_size(win)[1])
        if glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:action=mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS:action=mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:return
        mujoco.mjv_moveCamera(sim.model,action,dx/height,dy/height,scene,camera)
    def scroll(win,x,y):mujoco.mjv_moveCamera(sim.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,scene,camera)
    glfw.set_key_callback(window,key);glfw.set_cursor_pos_callback(window,cursor);glfw.set_scroll_callback(window,scroll)
    began=time.monotonic();last_state=sim.state;last_failure=0;frame_times=[];last_render=None
    worker.start()
    try:
        while not glfw.window_should_close(window) and (seconds is None or time.monotonic()-began<seconds):
            now=time.monotonic();glfw.poll_events()
            if worker.error is not None:raise RuntimeError('Simulation worker failed') from worker.error
            packet=worker.read();snap,qpos,qvel=packet
            if packet is not last_render:
                render_data.qpos[:]=qpos;render_data.qvel[:]=qvel;render_data.time=snap['time']
                mujoco.mj_forward(sim.model,render_data);last_render=packet
            if snap['state']!=last_state:print(json.dumps(snap),flush=True);last_state=snap['state']
            if snap['failure']!=last_failure:
                last_failure=snap['failure']
                if last_failure:print(f'Task failure {last_failure} recorded. Preview motion is not a successful task.',flush=True)
            width,height=glfw.get_framebuffer_size(window);viewport=mujoco.MjrRect(0,0,width,height)
            mujoco.mjv_updateScene(sim.model,render_data,option,perturb,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            mujoco.mjr_render(viewport,scene,context)
            recent=min(60,len(frame_times))
            fps=(recent-1)/(frame_times[-1]-frame_times[-recent]) if recent>=2 else 0.
            speed=worker.steps*sim.metadata['policy_dt']/max(worker.active_seconds,1e-9) if snap['state']=='RUNNING' else 0.
            text=f"{snap['state']}   t={snap['time']:.2f}s   phase={snap['phase']}\nSpace: start    R: reset    Esc: quit\nPolicy: {sim.metadata['policy_id']}\nFailure code: {snap['failure']}   Complete: {snap['completed']}\nDisplay: {fps:.0f} FPS   Simulation: {speed:.2f}x"
            if snap['failure']:text+='\nTASK FAILED - motion shown for inspection only'
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,text,'',context)
            if capture is not None:capture(viewport,context,SimpleNamespace(state=snap['state'],data=render_data,model=sim.model))
            glfw.swap_buffers(window);frame_times.append(time.monotonic())
            time.sleep(max(0.,1./render_fps-(time.monotonic()-now)))
    finally:
        try:worker.stop()
        finally:context.free();glfw.destroy_window(window);glfw.terminate()
    elapsed=time.monotonic()-began
    sim.viewer_stats=dict(frames=len(frame_times),wall_seconds=elapsed,render_fps=(len(frame_times)-1)/max(frame_times[-1]-frame_times[0],1e-9) if len(frame_times)>1 else 0.,
        physics_control_steps=worker.steps,physics_compute_seconds=worker.compute_seconds,physics_active_wall_seconds=worker.active_seconds,
        physics_seconds=worker.steps*sim.metadata['policy_dt'],target_render_fps=render_fps)
    return worker.events


def main():
    p=argparse.ArgumentParser(description='Step ONNX/MuJoCo: stationary on launch, Space to execute, R to reset.')
    p.add_argument('--policy',type=Path,help='Directory with policy.onnx and policy.json; default is the reviewed bundled candidate')
    p.add_argument('--model-dir',type=Path);p.add_argument('--headless',action='store_true');p.add_argument('--start',action='store_true',help='Start immediately, useful for headless validation')
    p.add_argument('--seconds',type=float,help='Simulation duration in headless mode, wall duration in GUI mode')
    p.add_argument('--render-fps',type=float,default=60.,help='Window refresh target (default60); physics and policy rates remain fixed')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--stop-on-failure',dest='continue_on_failure',action='store_false',help='Strict mode: pause at the first recorded task failure')
    modes.add_argument('--continue-on-failure',dest='continue_on_failure',action='store_true',help='Default preview mode: keep showing the policy after a recorded task failure; failure remains latched')
    p.set_defaults(continue_on_failure=True)
    p.add_argument('--output',type=Path,help='Write result.json and trace.jsonl in this directory')
    args=p.parse_args()
    if args.seconds is not None and args.seconds<=0:p.error('--seconds must be positive')
    if not 1<=args.render_fps<=240:p.error('--render-fps must be between1 and240')
    sim=Preview(args.policy,args.model_dir,args.continue_on_failure)
    print('Policy:',sim.metadata['policy_id']);print('Space: start policy; R: reset; Esc: exit. COMPLETE is only the strict task monitor result.',flush=True)
    if args.start:sim.start()
    trace=None
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False);trace=(args.output/'trace.jsonl').open('w')
    try:
        if args.headless:
            events=[];steps=round((args.seconds or 15)/sim.metadata['policy_dt'])
            for _ in range(steps):
                snap=sim.step()
                if trace:trace.write(json.dumps(snap)+'\n')
                if sim.state!='RUNNING':break
        else:events=show(sim,args.seconds,trace,render_fps=args.render_fps)
        result=dict(policy_id=sim.metadata['policy_id'],onnx_sha256=sim.metadata['onnx_sha256'],mujoco_version=mujoco.__version__,
            final=sim.snapshot(),first_failure=sim.failure_snapshot,keyboard_events=events,continue_on_failure=args.continue_on_failure,
            viewer=getattr(sim,'viewer_stats',None))
        if args.output:(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        if trace:trace.close()
