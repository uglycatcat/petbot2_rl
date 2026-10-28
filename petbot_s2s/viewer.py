"""GLFW viewer with press/release controls (no key-repeat dependence)."""
import time

import glfw
import mujoco


PHASES = ['FOUR WHEELS','RISING','TWO WHEELS','LANDING','HANDOFF']
FAILURES = {1:'invalid state',2:'body too low',4:'overturned',8:'excessive speed',
            16:'excessive rotation',32:'takeover not ready',64:'rise timed out',128:'landing timed out',
            256:'travel boundary',512:'front wheels too low',1024:'rear support lost',
            2048:'body contact',4096:'persistent overspeed'}


class Keyboard:
    def __init__(self):
        self.down = set()
        self.toggle = False
        self.reset = False

    def event(self, key, action):
        if action == glfw.PRESS:
            self.down.add(key)
            if key == glfw.KEY_SPACE:
                self.toggle = True
            if key == glfw.KEY_R:
                self.reset = True
        elif action == glfw.RELEASE:
            self.down.discard(key)

    def command(self, speed, yaw):
        return ((int(glfw.KEY_W in self.down)-int(glfw.KEY_S in self.down))*speed,
                (int(glfw.KEY_A in self.down)-int(glfw.KEY_D in self.down))*yaw)


def run_viewer(sim, smoke_seconds=None, forward_speed=None, turn_rate=None):
    forward_speed = sim.controller.speed_limit if forward_speed is None else forward_speed
    turn_rate = sim.controller.yaw_limit if turn_rate is None else turn_rate
    if not glfw.init():
        raise RuntimeError('GLFW could not open a display. Run on a desktop, or use --headless.')
    window = glfw.create_window(1280,800,'PetBot upright - MuJoCo',None,None)
    if window is None:
        glfw.terminate()
        raise RuntimeError('Unable to create an OpenGL window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    context = mujoco.MjrContext(sim.model,mujoco.mjtFontScale.mjFONTSCALE_150)
    scene = mujoco.MjvScene(sim.model,maxgeom=3000)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .85,135,-22
    option = mujoco.MjvOption()
    keys = Keyboard()
    mouse = [None,None]

    def key_callback(_window,key,scancode,action,mods):
        keys.event(key,action)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window,True)

    def cursor_callback(_window,x,y):
        if mouse[0] is not None and glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:
            width,height = glfw.get_window_size(window)
            mujoco.mjv_moveCamera(sim.model,mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                                 (x-mouse[0])/height,(y-mouse[1])/height,camera)
        mouse[:] = x,y

    glfw.set_key_callback(window,key_callback)
    glfw.set_cursor_pos_callback(window,cursor_callback)
    glfw.set_scroll_callback(window,lambda w,x,y: mujoco.mjv_moveCamera(
        sim.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0.,-.05*y,camera))
    glfw.set_window_focus_callback(window,lambda w,focused: keys.down.clear() if not focused else None)
    next_step = start = time.monotonic()
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if keys.reset:
                sim.reset()
                next_step = time.monotonic()
                keys.reset = keys.toggle = False
            c = sim.controller
            c.set_command(*keys.command(forward_speed,turn_rate))
            if keys.toggle:
                c.toggle()
                keys.toggle = False
            now = time.monotonic()
            # Render between physics substeps rather than showing a new pose
            # only after a complete 60 ms policy block. Bound catch-up work so
            # a slow machine stays responsive; never enlarge the physics dt.
            for _ in range(30):
                if next_step > now or c.failure_code:
                    break
                sim.physics_step()
                next_step += .002
            if next_step < now-.06 or c.failure_code:
                next_step = now
            camera.lookat[:] = sim.data.xpos[sim.root]
            width,height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0,0,width,height)
            mujoco.mjv_updateScene(sim.model,sim.data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            mujoco.mjr_render(viewport,scene,context)
            r = sim.record()
            status = PHASES[r['phase']]
            if r['armed'] and r['phase']==0:
                status += ' / REQUEST PENDING'
            if r['failure_code']:
                labels = dict(FAILURES)
                if c.transition_contract == 'linear_heading':
                    labels.update({1024:'rear axle height mismatch',2048:'overturned'})
                reason=', '.join(text for bit,text in labels.items() if r['failure_code']&bit)
                status += f" / PAUSED: {reason} (R to reset)"
            if c.full_range and c.interruptions and not c.armed:
                status += ' / CALL ENDED: Space for a new request'
            left = 'W/S: forward/reverse | A/D: turn\nSpace: rise / front wheels down | R: reset | Esc: exit\n'
            left += 'Mouse drag: orbit | Wheel: zoom\n\n'+status
            right = (f"Command {r['raw'][0]:+.3f} m/s  {r['raw'][1]:+.3f} rad/s\n"
                     f"Measured {r['motion'][0]:+.3f} m/s  {r['motion'][1]:+.3f} rad/s\n"
                     f"Cycles {r['cycles']} | Simulation {r['time']:.1f}s\n"
                     f"Policy: {sim.bundle.name}")
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport,left,right,context)
            glfw.swap_buffers(window)
            if smoke_seconds and time.monotonic()-start >= smoke_seconds:
                break
            time.sleep(.001)
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
