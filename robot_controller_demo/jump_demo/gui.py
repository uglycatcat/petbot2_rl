"""Swiss-style joint and IMU traces. A 10 s window, redrawn live."""

import tkinter as tk
import tkinter.font as tkfont

import numpy as np

WHITE = "#ffffff"
INK = "#111111"
MUTED = "#5c5c5c"
HAIR = "#e4e4e4"
RULE = "#111111"
RED = "#e10600"
BLUE = "#0047bb"
WINDOW = 10.0

FRAME_SERIES = (
    ("left_frame_target", "LEFT TARGET", RED, None),
    ("right_frame_target", "RIGHT TARGET", RED, (5, 3)),
    ("left_frame", "LEFT", INK, None),
    ("right_frame", "RIGHT", INK, (5, 3)),
)
WHEEL_SERIES = (
    ("left_wheel_target", "LEFT TARGET", RED, None),
    ("right_wheel_target", "RIGHT TARGET", RED, (5, 3)),
    ("left_wheel", "LEFT", INK, None),
    ("right_wheel", "RIGHT", INK, (5, 3)),
)
IMU_SERIES = (
    ("pitch", "PITCH", INK, None),
    ("roll", "ROLL", RED, None),
    ("yaw", "YAW", BLUE, None),
)


def _family(root, names, fallback):
    have = set(tkfont.families(root))
    for name in names:
        if name in have:
            return name
    return fallback


class TraceLog:
    """Fixed-rate samples kept for one rolling window."""

    def __init__(self, horizon, dt, names):
        self.horizon = float(horizon)
        self.names = tuple(names)
        self.capacity = int(round(self.horizon / dt)) + 8
        self.time = np.full(self.capacity, np.nan)
        self.value = np.full((len(self.names), self.capacity), np.nan)
        self._index = {name: i for i, name in enumerate(self.names)}
        self._write = 0
        self._count = 0

    def push(self, time, values):
        slot = self._write
        self.time[slot] = time
        for name, value in values.items():
            self.value[self._index[name], slot] = value
        self._write = (slot + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def window(self, now):
        if self._count == 0:
            empty = np.empty((len(self.names), 0))
            return np.empty(0), empty
        if self._count < self.capacity:
            time = self.time[: self._count]
            value = self.value[:, : self._count]
        else:
            time = np.concatenate((self.time[self._write :], self.time[: self._write]))
            value = np.concatenate(
                (self.value[:, self._write :], self.value[:, : self._write]), axis=1)
        keep = time >= now - self.horizon
        return time[keep], value[:, keep]


class _Plot(tk.Canvas):
    def __init__(self, parent, series, unit, y_span, digits, unwrap=False):
        super().__init__(
            parent, width=640, height=168, bg=WHITE,
            highlightthickness=1, highlightbackground=RULE, bd=0)
        self._series = series
        self._unit = unit
        self._y_span = y_span
        self._digits = digits
        self._unwrap = unwrap
        self._font = ("TkFixedFont", 8)
        self._index = {}

    def draw(self, time, value, names, now):
        self.delete("all")
        width = max(self.winfo_width(), int(self["width"]))
        height = max(self.winfo_height(), int(self["height"]))
        left, right, top, bottom = 52, width - 12, 10, height - 22
        self.create_rectangle(left, top, right, bottom, outline=HAIR)
        t0 = now - WINDOW
        t1 = now
        columns = {}
        if len(time):
            for name, _, _, _ in self._series:
                column = value[names.index(name)]
                if self._unwrap:
                    column = np.degrees(np.unwrap(np.radians(column)))
                columns[name] = column
        finite = np.concatenate([column[np.isfinite(column)] for column in columns.values()]) if columns else np.empty(0)
        if finite.size == 0:
            y0, y1 = -self._y_span / 2, self._y_span / 2
        else:
            lo = float(finite.min())
            hi = float(finite.max())
            span = max(hi - lo, self._y_span)
            mid = 0.5 * (lo + hi)
            pad = 0.08 * span
            y0 = mid - 0.5 * span - pad
            y1 = mid + 0.5 * span + pad
        self._grid(left, right, top, bottom, t0, t1, y0, y1)
        if len(time):
            stride = max(1, len(time) // 500)
            index = np.arange(0, len(time), stride)
            if index[-1] != len(time) - 1:
                index = np.append(index, len(time) - 1)
            for name, _, color, dash in self._series:
                self._stroke(time[index], columns[name][index], color, dash, left, right, top, bottom, t0, t1, y0, y1)
        self._index = {}
        for name, _, _, _ in self._series:
            if name not in columns or not np.any(np.isfinite(columns[name])):
                self._index[name] = None
            else:
                good = columns[name][np.isfinite(columns[name])]
                self._index[name] = float(good[-1])

    def _grid(self, left, right, top, bottom, t0, t1, y0, y1):
        for fraction in (0.0, 0.5, 1.0):
            y = bottom - fraction * (bottom - top)
            value = y0 + fraction * (y1 - y0)
            self.create_line(left, y, right, y, fill=HAIR)
            self.create_text(left - 6, y, text=f"{value:.{self._digits}f}", anchor="e", fill=MUTED, font=self._font)
        if y0 < 0 < y1:
            y = bottom - (0 - y0) / (y1 - y0) * (bottom - top)
            self.create_line(left, y, right, y, fill=RULE)
        for mark in range(-10, 1, 2):
            x = left + (mark + WINDOW) / WINDOW * (right - left)
            self.create_line(x, bottom, x, bottom + 4, fill=RULE)
            self.create_text(x, bottom + 12, text=str(mark), fill=MUTED, font=self._font)
        self.create_line(left, top, left, bottom, fill=RULE)
        self.create_line(left, bottom, right, bottom, fill=RULE)

    def _stroke(self, time, values, color, dash, left, right, top, bottom, t0, t1, y0, y1):
        span_t = t1 - t0
        span_y = y1 - y0
        coords = []
        for stamp, value in zip(time, values):
            if not np.isfinite(stamp) or not np.isfinite(value):
                self._emit(coords, color, dash)
                coords = []
                continue
            coords.extend((
                float(left + (stamp - t0) / span_t * (right - left)),
                float(bottom - (value - y0) / span_y * (bottom - top)),
            ))
        self._emit(coords, color, dash)

    def _emit(self, coords, color, dash):
        if len(coords) < 4:
            return
        options = {"fill": color, "width": 1 if dash else 2, "capstyle": tk.ROUND}
        if dash:
            options["dash"] = dash
        self.create_line(*coords, **options)


class JumpGui:
    def __init__(self, dt):
        self.alive = True
        self.root = tk.Tk()
        self.root.title("PetBot Joint Data")
        self.root.configure(bg=WHITE)
        self.root.geometry("700x900+32+32")
        self._jump_click = False
        self._cross_click = False
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        sans = _family(self.root, ("Nimbus Sans", "Liberation Sans", "FreeSans", "DejaVu Sans"), "TkDefaultFont")
        mono = _family(self.root, ("DejaVu Sans Mono", "Liberation Mono", "FreeMono"), "TkFixedFont")
        names = [name for series in (FRAME_SERIES, WHEEL_SERIES, IMU_SERIES) for name, _, _, _ in series]
        self.log = TraceLog(WINDOW, dt, names)
        self._plots = {}
        self._values = {}
        self._build(sans, mono)
        for plot in self._plots.values():
            plot._font = (mono, 8)
        self.root.update_idletasks()

    def close(self):
        if not self.alive:
            return
        self.alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def pump(self):
        if not self.alive:
            return
        try:
            self.root.update()
        except tk.TclError:
            self.alive = False

    def push(self, time, values):
        if self.alive:
            self.log.push(time, values)

    def consume_jump(self):
        clicked = self._jump_click
        self._jump_click = False
        return clicked

    def consume_cross(self):
        clicked = self._cross_click
        self._cross_click = False
        return clicked

    def set_actions_enabled(self, enabled):
        if not self.alive:
            return
        state = "normal" if enabled else "disabled"
        self._jump.configure(state=state)
        self._cross.configure(state=state)

    def render(self, now, phase):
        if not self.alive:
            return
        self._phase.configure(text=phase.upper())
        time, value = self.log.window(now)
        names = self.log.names
        for key, plot in self._plots.items():
            plot.draw(time, value, list(names), now)
            for name, label in self._values[key]:
                reading = plot._index.get(name)
                text = "—" if reading is None else f"{reading:+.{plot._digits}f}"
                label.configure(text=text)

    def _build(self, sans, mono):
        pad = tk.Frame(self.root, bg=WHITE, padx=24, pady=20)
        pad.pack(fill="both", expand=True)
        head = tk.Frame(pad, bg=WHITE)
        head.pack(fill="x")
        tk.Label(head, text="PETBOT", bg=WHITE, fg=INK, font=(sans, 22, "bold")).pack(side="left")
        tk.Label(head, text="JOINT DATA", bg=WHITE, fg=MUTED, font=(sans, 11)).pack(side="left", padx=(12, 0))
        self._phase = tk.Label(head, text="DRIVE", bg=WHITE, fg=INK, font=(mono, 11))
        self._phase.pack(side="right")
        tk.Frame(pad, bg=RULE, height=1).pack(fill="x", pady=(12, 0))
        action = tk.Frame(pad, bg=WHITE)
        action.pack(fill="x", pady=(12, 0))
        self._jump = tk.Button(
            action, text="Rear Hop", command=self._request_jump,
            state="disabled", bg=WHITE, fg=INK, activebackground=WHITE,
            activeforeground=INK, relief="solid", bd=1, font=(sans, 11),
            padx=14, pady=4)
        self._jump.pack(side="left")
        self._cross = tk.Button(
            action, text="Cross", command=self._request_cross,
            state="disabled", bg=WHITE, fg=INK, activebackground=WHITE,
            activeforeground=INK, relief="solid", bd=1, font=(sans, 11),
            padx=14, pady=4)
        self._cross.pack(side="left", padx=(12, 0))
        self._figure(pad, "01", "FRAME POSITION", "rad", FRAME_SERIES, 0.8, 2, sans, mono, False)
        self._figure(pad, "02", "WHEEL SPEED", "rad/s", WHEEL_SERIES, 8.0, 1, sans, mono, False)
        self._figure(pad, "03", "IMU ATTITUDE", "deg", IMU_SERIES, 30.0, 1, sans, mono, True)
        tk.Label(
            pad, text="ROLLING 10 s    ·    RIGHT EDGE IS NOW    ·    DASHED LINES ARE THE RIGHT JOINT",
            bg=WHITE, fg=MUTED, font=(sans, 8)).pack(anchor="w", pady=(12, 0))

    def _request_jump(self):
        self._jump_click = True
        self._cross_click = False

    def _request_cross(self):
        self._cross_click = True
        self._jump_click = False

    def _figure(self, parent, index, title, unit, series, y_span, digits, sans, mono, unwrap):
        block = tk.Frame(parent, bg=WHITE)
        block.pack(fill="x", pady=(16, 0))
        title_row = tk.Frame(block, bg=WHITE)
        title_row.pack(fill="x")
        tk.Label(title_row, text=index, bg=WHITE, fg=RED, font=(sans, 14, "bold")).pack(side="left")
        tk.Label(title_row, text=title, bg=WHITE, fg=INK, font=(sans, 11, "bold")).pack(side="left", padx=(8, 0))
        tk.Label(title_row, text=unit, bg=WHITE, fg=MUTED, font=(sans, 10)).pack(side="right")
        plot = _Plot(block, series, unit, y_span, digits, unwrap)
        plot.pack(fill="x", pady=(6, 0))
        legend = tk.Frame(block, bg=WHITE)
        legend.pack(fill="x", pady=(6, 0))
        labels = []
        for name, text, color, _dash in series:
            item = tk.Frame(legend, bg=WHITE)
            item.pack(side="left", padx=(0, 14))
            tk.Label(item, text=text, bg=WHITE, fg=color, font=(sans, 8, "bold")).pack(side="left")
            value = tk.Label(item, text="—", bg=WHITE, fg=INK, font=(mono, 8))
            value.pack(side="left", padx=(6, 0))
            labels.append((name, value))
        self._plots[index] = plot
        self._values[index] = labels
