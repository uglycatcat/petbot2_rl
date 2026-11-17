"""Small desktop control panel; simulation actions run on the main loop."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk


class SnapshotPanel:
    def __init__(self, commands, directory: Path):
        self.commands = commands
        self.directory = directory
        self.alive = True
        self.reset_clock = False
        self.root = tk.Tk()
        self.root.title("PetBot 姿态快照")
        self.root.geometry("480x540+40+60")
        self.root.minsize(480, 540)
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#f3f5f7")
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f7")
        style.configure("TLabel", background="#f3f5f7", font=("Noto Sans CJK SC", 10))
        style.configure("TButton", font=("Noto Sans CJK SC", 11), padding=(8, 10))
        style.configure("Save.TButton", background="#2361d9", foreground="white")
        style.map("Save.TButton", background=[("active", "#184cb3")])

        content = ttk.Frame(self.root, padding=20)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="姿态快照", font=("Noto Sans CJK SC", 19, "bold")).pack(anchor="w")
        self.state = ttk.Label(content, foreground="#24754b")
        self.state.pack(anchor="w", pady=(2, 14))

        self.save_button = ttk.Button(content, text="保存快照", style="Save.TButton",
                                      command=lambda: self.send("save"))
        self.save_button.pack(fill="x")
        buttons = ttk.Frame(content)
        buttons.pack(fill="x", pady=(8, 14))
        buttons.columnconfigure((0, 1), weight=1, uniform="button")
        self.latest_button = ttk.Button(buttons, text="加载最新快照", command=lambda: self.send("load"))
        self.open_button = ttk.Button(buttons, text="选择快照…", command=self.choose_snapshot)
        self.reset_button = ttk.Button(buttons, text="回到默认位置", command=lambda: self.send("reset"))
        self.pause_button = ttk.Button(buttons, text="暂停仿真", command=lambda: self.send("pause"))
        for index, button in enumerate((self.latest_button, self.open_button, self.reset_button, self.pause_button)):
            button.grid(row=index // 2, column=index % 2, sticky="ew",
                        padx=(0, 4) if index % 2 == 0 else (4, 0), pady=4)

        ttk.Separator(content).pack(fill="x")
        readings = ttk.Frame(content)
        readings.pack(fill="x", pady=12)
        readings.columnconfigure(1, weight=1)
        self.readings = {}
        for row, (key, title) in enumerate((
                ("rpy", "横滚 / 俯仰 / 偏航 (°)"), ("tilt", "倾斜角 (°)"),
                ("height", "机身基座高度 (cm)"),
                ("accel", "加速度 (m/s²)"), ("gyro", "角速度 (rad/s)"),
                ("error", "关节最大默认角误差 (°)"))):
            ttk.Label(readings, text=title, foreground="#596575").grid(row=row, column=0, sticky="w", pady=4)
            value = ttk.Label(readings, font=("DejaVu Sans Mono", 10))
            value.grid(row=row, column=1, sticky="e", padx=(10, 0), pady=4)
            self.readings[key] = value

        ttk.Separator(content).pack(fill="x")
        self.message = ttk.Label(content, text="已加载默认位置，可以在机器人窗口中手动拨动。",
                                 wraplength=430, foreground="#39485b")
        self.message.pack(fill="x", pady=(12, 6))
        self.file_label = ttk.Label(content, text="尚未保存快照", wraplength=430, foreground="#596575")
        self.file_label.pack(fill="x")
        ttk.Label(content, text="在机器人窗口双击机身，Ctrl + 鼠标拖动。\n"
                  "加载后自动暂停；关闭任一窗口即可退出。",
                  foreground="#596575", wraplength=430).pack(side="bottom", anchor="w", pady=(12, 0))
        self.root.update_idletasks()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def send(self, action, path=None):
        self.commands.put((action, path))

    def choose_snapshot(self):
        directory = self.directory if self.directory.is_dir() else self.directory.parent
        try:
            selected = filedialog.askopenfilename(
                parent=self.root, title="选择姿态快照", initialdir=str(directory),
                filetypes=[("姿态快照", "*.json")])
            if selected:
                self.send("load", Path(selected))
        finally:
            # The dialog pauses the main loop. Do not catch up that wall time.
            self.reset_clock = True

    def request_close(self):
        self.send("quit")
        self.alive = False

    def pump(self):
        if self.alive:
            self.root.update()

    def show_message(self, text, *, path=None, error=False):
        self.message.configure(text=text, foreground="#b42318" if error else "#39485b")
        if path is not None:
            self.file_label.configure(text=f"文件：{path.name}")

    def render(self, imu, paused, sim_time, home_error, height):
        self.state.configure(text=f"{'已暂停' if paused else '仿真运行中'}  ·  {sim_time:.2f} s",
                             foreground="#a66b15" if paused else "#24754b")
        self.pause_button.configure(text="继续仿真" if paused else "暂停仿真")
        self.readings["rpy"].configure(text=" / ".join(f"{x:+.1f}" for x in imu["rpy_deg"]))
        self.readings["tilt"].configure(text=f"{imu['tilt_deg']:.1f}")
        self.readings["height"].configure(text=f"{height * 100:.2f}")
        self.readings["accel"].configure(text=" / ".join(f"{x:+.2f}" for x in imu["accel_m_s2"]))
        self.readings["gyro"].configure(text=" / ".join(f"{x:+.2f}" for x in imu["gyro_rad_s"]))
        self.readings["error"].configure(text=f"{home_error:.2f}")

    def close(self):
        self.alive = False
        self.root.destroy()
