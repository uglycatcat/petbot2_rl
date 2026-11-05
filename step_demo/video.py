"""One unmodified policy episode, streamed to H.264 MP4 inside Isaac."""
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


class PolicyVideo:
    fps = 25
    lead_in_s = .5
    tail_s = 1.

    def __init__(self, env, path, label):
        self.env,self.path,self.label=env,path,label
        self.stride=round(1./(self.fps*env.step_dt))
        if abs(self.stride*env.step_dt-1./self.fps)>1e-6:
            raise ValueError('Video frame rate must divide the control rate')
        self.writer=imageio.get_writer(str(path),format='FFMPEG',fps=self.fps,
            codec='libx264',pixelformat='yuv420p',macro_block_size=2,
            output_params=['-movflags','+faststart'])
        self.font=ImageFont.load_default(size=22)
        self.frames=0;self.last=None;self.terminal=None
        env.terminal_frame_callback=self.capture_terminal

    def frame(self):
        # Renderer warmup only; no policy action or physics step is inserted.
        for _ in range(12):
            rgb=self.env.render()
            if rgb is not None and np.asarray(rgb).max()>0:
                return np.asarray(rgb,dtype=np.uint8).copy()
        raise RuntimeError('Isaac renderer did not produce a nonblank RGB frame')

    def append(self, frame, text):
        image=Image.fromarray(frame);draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,42),fill=(15,18,24))
        draw.text((12,8),text,font=self.font,fill='white')
        self.writer.append_data(np.asarray(image));self.frames+=1

    def begin(self):
        self.last=self.frame()
        for _ in range(round(self.lead_in_s*self.fps)):
            self.append(self.last,f'{self.label} | Initial pose (paused)')

    def capture_terminal(self, records):
        if self.terminal is None:
            self.terminal=(self.frame(),records[0])

    def step(self, step):
        if self.terminal is not None:return
        if step % self.stride == 0:
            self.last=self.frame()
            self.append(self.last,f'{self.label} | t={step*self.env.step_dt:.2f}s')

    def finish(self, record):
        if self.terminal is None:raise RuntimeError('Missing frame before terminal reset')
        frame,_=self.terminal
        reasons={1:'front collision',2:'rear airborne',3:'fall',4:'path departure',
                 5:'invalid front flight',6:'platform exit',7:'timeout'}
        outcome='SUCCESS' if record['success'] else 'FAILED: '+reasons.get(record['failure'],'unknown')
        for _ in range(1+round(self.tail_s*self.fps)):
            self.append(frame,f'{self.label} | {outcome} | t={record["elapsed_s"]:.2f}s')
        self.close()
        return dict(status='complete',file=self.path.name,format='mp4',codec='h264',fps=self.fps,
            frames=self.frames,duration_s=self.frames/self.fps,simulation_s=record['elapsed_s'],
            initial_pause_s=round(self.lead_in_s*self.fps)/self.fps,final_pause_s=self.tail_s,
            success=record['success'],failure=record['failure'],deterministic_policy=True,
            terminal_frame_before_reset=True,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def close(self):
        self.env.terminal_frame_callback=None
        if self.writer is not None:self.writer.close();self.writer=None
