"""Finite Linux GUI check using an isolated Xvfb and actual X11 Space/R events.

Validation-only dependency: python-xlib. Rendering uses Mesa software, no GPU.
"""
import argparse,json,os,struct,subprocess,sys,time,zlib
from pathlib import Path


def png(path,rgb,width,height):
    def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    rows=b''.join(b'\0'+rgb[i*width*3:(i+1)*width*3] for i in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b''))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    from Xlib import X,XK,display
    from Xlib.ext import xtest
    import numpy as np
    number=next(i for i in range(190,220) if not Path(f'/tmp/.X11-unix/X{i}').exists() and not Path(f'/tmp/.X{i}-lock').exists())
    env=os.environ.copy();env.update(DISPLAY=f':{number}',LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa')
    xvlog=(out/'xvfb.log').open('w');xv=subprocess.Popen(['Xvfb',f':{number}','-screen','0','1100x720x24','-nolisten','tcp'],stdout=xvlog,stderr=subprocess.STDOUT)
    proc=None;d=None
    try:
        deadline=time.monotonic()+10
        while not Path(f'/tmp/.X11-unix/X{number}').exists():
            if xv.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Xvfb failed')
            time.sleep(.1)
        log=(out/'app.log').open('w');proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--render-child',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
        d=display.Display(env['DISPLAY']);window=None;deadline=time.monotonic()+10
        while window is None and time.monotonic()<deadline:
            for w in d.screen().root.query_tree().children:
                name=w.get_wm_name()
                if name and name.startswith('PetBot Step'):window=w;break
            if proc.poll() is not None:raise RuntimeError('GUI exited before window opened')
            time.sleep(.1)
        if window is None:raise RuntimeError('GUI window missing')
        window.set_input_focus(X.RevertToParent,X.CurrentTime);d.sync();time.sleep(.5)
        def press(name):
            code=d.keysym_to_keycode(XK.string_to_keysym(name));xtest.fake_input(d,X.KeyPress,code);xtest.fake_input(d,X.KeyRelease,code);d.sync()
        press('space');time.sleep(.5)
        geo=window.get_geometry();shot=window.get_image(0,0,geo.width,geo.height,X.ZPixmap,0xffffffff)
        raw=np.frombuffer(shot.data,np.uint8).reshape(geo.height,geo.width,4);png(out/'x11.png',raw[:,:,[2,1,0]].tobytes(),geo.width,geo.height)
        time.sleep(1.);press('r');time.sleep(.5);press('space');time.sleep(1.5);press('r');time.sleep(.3);press('Escape')
        proc.wait(timeout=30);assert proc.returncode==0
        result=json.loads((out/'runtime/result.json').read_text());events=result['keyboard_events'];assert [e['key'] for e in events]==[32,82,32,82],events
        for e in [events[0],events[2]]:assert e['after']['state']=='RUNNING' and e['after']['time']==0 and e['after']['packet_count']==1
        for e in [events[1],events[3]]:assert e['after']['state']=='IDLE' and e['after']['time']==0 and e['after']['policy_calls']==0 and e['after']['packet_count']==0
        trace=[json.loads(x) for x in (out/'runtime/trace.jsonl').read_text().splitlines()]
        assert any(x['policy_calls']>0 for x in trace)
        frames=json.loads((out/'frame_review.json').read_text());assert set(frames)=={'idle','running'}
        for frame in frames.values():assert frame['nonblack_pixels']>10000 and frame['pixel_std']>10,frames
        report=dict(status='passed',actual_x11_key_sequence=['Space','R','Space','R'],policy_executed=True,reset_twice=True,
            software_rendering=True,screenshot='running.png',rendered_frames=frames,final_state=result['final']['state'])
        (out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
    finally:
        if proc is not None and proc.poll() is None:proc.terminate();proc.wait(timeout=10)
        if d is not None:d.close()
        xv.terminate();xv.wait(timeout=10);xvlog.close()


def render_child(out):
    """Read the actual displayed framebuffer: Xvfb XGetImage can miss GLX buffers."""
    import functools
    import mujoco
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from step_s2s import app
    frames={}
    def capture(viewport,context,sim):
        key='idle' if sim.state=='IDLE' else 'running' if sim.data.time>=.2 else None
        if key is None or key in frames:return
        rgb=np.empty((viewport.height,viewport.width,3),np.uint8)
        mujoco.mjr_readPixels(rgb,None,viewport,context)
        png(out/(key+'.png'),rgb[::-1].tobytes(),viewport.width,viewport.height)
        frames[key]=dict(nonblack_pixels=int(np.count_nonzero(rgb.max(2))),pixel_std=float(rgb.std()),simulation_time=float(sim.data.time))
        (out/'frame_review.json').write_text(json.dumps(frames,indent=2)+'\n')
    app.show=functools.partial(app.show,capture=capture)
    sys.argv=[str(Path(__file__).resolve().parents[1]/'run.py'),'--seconds','12','--output',str(out/'runtime')]
    app.main()


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--render-child':render_child(Path(sys.argv[2]))
    else:main()
