import argparse,json,time
from pathlib import Path
import mujoco
from .runtime import Preview


def show(sim,seconds=None,trace=None,capture=None):
    import glfw
    if not glfw.init():raise RuntimeError('Cannot open display. On a desktop use a graphical terminal; use --headless for diagnostics.')
    window=glfw.create_window(1100,720,'PetBot Step | Space: start | R: reset',None,None)
    if not window:glfw.terminate();raise RuntimeError('Cannot create OpenGL window')
    glfw.make_context_current(window);glfw.swap_interval(1)
    scene=mujoco.MjvScene(sim.model,maxgeom=2000);context=mujoco.MjrContext(sim.model,mujoco.mjtFontScale.mjFONTSCALE_150)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.55,0,.10];camera.distance=1.7;camera.azimuth=130;camera.elevation=-22
    option=mujoco.MjvOption();perturb=mujoco.MjvPerturb();events=[];mouse=[0.,0.]
    def key(win,k,scan,action,mods):
        if action!=glfw.PRESS:return
        if k==glfw.KEY_ESCAPE:glfw.set_window_should_close(win,True);return
        if sim.key(k):
            events.append(dict(key=k,wall_elapsed=time.monotonic()-began,after=sim.snapshot()))
            print(sim.state,flush=True)
    def cursor(win,x,y):
        dx=x-mouse[0];dy=y-mouse[1];mouse[:]=[x,y]
        height=max(1,glfw.get_window_size(win)[1])
        if glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:action=mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS:action=mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:return
        mujoco.mjv_moveCamera(sim.model,action,dx/height,dy/height,scene,camera)
    def scroll(win,x,y):mujoco.mjv_moveCamera(sim.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,scene,camera)
    glfw.set_key_callback(window,key);glfw.set_cursor_pos_callback(window,cursor);glfw.set_scroll_callback(window,scroll)
    began=time.monotonic();next_step=began;last_state=sim.state;last_failure=0
    try:
        while not glfw.window_should_close(window) and (seconds is None or time.monotonic()-began<seconds):
            now=time.monotonic()
            caught_up=0
            while now>=next_step and caught_up<5:
                snap=sim.step()
                if trace and sim.state!='IDLE':trace.write(json.dumps(snap)+'\n')
                next_step+=sim.metadata['policy_dt'];caught_up+=1
            if now-next_step>.1:next_step=now
            if sim.state!=last_state:print(json.dumps(sim.snapshot()),flush=True);last_state=sim.state
            if sim.monitor.failure!=last_failure:
                last_failure=sim.monitor.failure
                if last_failure:print(f'Task failure {last_failure} recorded. Preview motion is not a successful task.',flush=True)
            width,height=glfw.get_framebuffer_size(window);viewport=mujoco.MjrRect(0,0,width,height)
            mujoco.mjv_updateScene(sim.model,sim.data,option,perturb,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            mujoco.mjr_render(viewport,scene,context)
            text=f"{sim.state}   t={sim.data.time:.2f}s   phase={sim.monitor.phase}\nSpace: start    R: reset    Esc: quit\nPolicy: {sim.metadata['policy_id']}\nFailure code: {sim.monitor.failure}   Complete: {sim.monitor.completed}"
            if sim.monitor.failure:text+='\nTASK FAILED - motion shown for inspection only'
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,text,'',context)
            if capture is not None:capture(viewport,context,sim)
            glfw.swap_buffers(window);glfw.poll_events()
    finally:
        context.free();glfw.destroy_window(window);glfw.terminate()
    return events


def main():
    p=argparse.ArgumentParser(description='Step ONNX/MuJoCo: stationary on launch, Space to execute, R to reset.')
    p.add_argument('--policy',type=Path,help='Directory with policy.onnx and policy.json; default is the reviewed bundled candidate')
    p.add_argument('--model-dir',type=Path);p.add_argument('--headless',action='store_true');p.add_argument('--start',action='store_true',help='Start immediately, useful for headless validation')
    p.add_argument('--seconds',type=float,help='Simulation duration in headless mode, wall duration in GUI mode')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--stop-on-failure',dest='continue_on_failure',action='store_false',help='Strict mode: pause at the first recorded task failure')
    modes.add_argument('--continue-on-failure',dest='continue_on_failure',action='store_true',help='Default preview mode: keep showing the policy after a recorded task failure; failure remains latched')
    p.set_defaults(continue_on_failure=True)
    p.add_argument('--output',type=Path,help='Write result.json and trace.jsonl in this directory')
    args=p.parse_args()
    if args.seconds is not None and args.seconds<=0:p.error('--seconds must be positive')
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
        else:events=show(sim,args.seconds,trace)
        result=dict(policy_id=sim.metadata['policy_id'],onnx_sha256=sim.metadata['onnx_sha256'],mujoco_version=mujoco.__version__,
            final=sim.snapshot(),first_failure=sim.failure_snapshot,keyboard_events=events,continue_on_failure=args.continue_on_failure)
        if args.output:(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        if trace:trace.close()
