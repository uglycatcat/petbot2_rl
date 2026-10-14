"""Swiss-style status window: mode, IMU axes, and the fall flag."""

import tkinter as tk
import tkinter.font as tkfont

import numpy as np

from attitude import FALLEN_ENTER_DEG, FALLEN_EXIT_DEG
from control import EAR, FRAME, PASSIVE, RECOVER, TRACK, WHEEL, RecoveryController

WHITE = "#ffffff"
INK = "#111111"
MUTED = "#5c5c5c"
RULE = "#111111"
RED = "#e10600"
WORLD = "#c6c6c6"
AXIS_COLOR = {"x": "#d00000", "y": "#0b7a32", "z": "#0047bb"}


def _family(root, names, fallback):
    have = set(tkfont.families(root))
    for name in names:
        if name in have:
            return name
    return fallback


class AxisView(tk.Canvas):
    """Fixed camera, world triad in gray, IMU triad in red / green / blue."""

    def __init__(self, parent, sans):
        super().__init__(
            parent, width=320, height=168, bg=WHITE,
            highlightthickness=1, highlightbackground=RULE, bd=0)
        self._sans = sans

    def draw(self, rotation: np.ndarray):
        self.delete("all")
        width = max(self.winfo_width(), int(self["width"]))
        height = max(self.winfo_height(), int(self["height"]))
        segments = _frame_segments(rotation, width, height)
        for segment in sorted(segments, key=lambda item: item["depth"]):
            x0, y0 = segment["start"]
            x1, y1 = segment["end"]
            color = segment["color"]
            self.create_line(x0, y0, x1, y1, fill=color, width=segment["width"], capstyle=tk.ROUND)
            if segment["arrow"]:
                _arrow_head(self, (x0, y0), (x1, y1), color)
                self.create_text(
                    segment["label_at"][0], segment["label_at"][1],
                    text=segment["label"], fill=color, font=(self._sans, 11, "bold"))
        ox, oy = segments[0]["origin"]
        self.create_oval(ox - 3, oy - 3, ox + 3, oy + 3, fill=INK, outline=INK)


class RecoveryGui:
    def __init__(self, on_mode, on_bar, on_drag, on_stand, limits):
        self._on_mode = on_mode
        self._on_bar = on_bar
        self._on_drag = on_drag
        self._on_stand = on_stand
        self._limits = limits
        self.alive = True
        self.root = tk.Tk()
        self.root.title("PetBot Pose Recovery")
        self.root.configure(bg=WHITE)
        self.root.geometry("+40+40")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        sans = _family(self.root, ("Nimbus Sans", "Liberation Sans", "FreeSans", "DejaVu Sans"), "TkDefaultFont")
        mono = _family(self.root, ("DejaVu Sans Mono", "Liberation Mono", "FreeMono"), "TkFixedFont")
        self._sans = sans
        self._build(sans, mono)
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

    def render(self, state):
        if not self.alive:
            return
        live = not state.standing
        self._passive.set_active(live and state.mode == PASSIVE)
        self._track.set_active(live and state.mode == TRACK)
        self._recover.set_active(live and state.mode == RECOVER)
        for key in (self._passive, self._track, self._recover):
            key.set_enabled(live)
        if state.standing:
            done = state.progress * state.stand_total
            status = f"STAND  {done:3.1f} / {state.stand_total:.1f} s    ·    THEN {state.mode.upper()}"
        elif state.mode == PASSIVE:
            status = "MOTORS OFF"
        elif state.mode == TRACK:
            status = "BARS LIVE"
        elif state.progress < 1.0:
            done = state.progress * RecoveryController.DURATION
            status = f"HOMING    {done:3.1f} / {RecoveryController.DURATION:.1f} s"
        else:
            status = "HOLDING HOME    ·    WHEEL SPEED 0"
        self._status.configure(text=status)
        locked = state.standing or state.mode != TRACK
        self._frames.show(state.frame, locked)
        self._ears.show(state.ear, locked)
        self._wheels.show(state.wheel, locked)
        self._stand.set_enabled(not state.dragging and not state.standing)
        self._axes.draw(state.rotation)
        roll, pitch, yaw = state.rpy
        self._roll.configure(text=_signed_deg(roll))
        self._pitch.configure(text=_signed_deg(pitch))
        self._yaw.configure(text=_signed_deg(yaw))
        fallen = state.fallen
        band = RED if fallen else WHITE
        word = WHITE if fallen else INK
        self._band.configure(bg=band)
        self._state.configure(text="FALLEN" if fallen else "UPRIGHT", bg=band, fg=word)
        self._tilt.configure(text=f"{np.degrees(state.tilt):05.1f}°")

    def _build(self, sans, mono):
        pad = tk.Frame(self.root, bg=WHITE)
        pad.pack(fill="both", expand=True, padx=22, pady=16)

        head = tk.Frame(pad, bg=WHITE)
        head.pack(fill="x")
        tk.Frame(head, bg=RED, width=10, height=10).pack(side="left", pady=4)
        tk.Label(head, text="PETBOT", bg=WHITE, fg=INK, font=(sans, 18, "bold")).pack(side="left", padx=(10, 0))
        tk.Label(head, text="POSE RECOVERY", bg=WHITE, fg=MUTED, font=(sans, 10)).pack(side="left", padx=(12, 0))

        body = tk.Frame(pad, bg=WHITE)
        body.pack(fill="both", expand=True, pady=(14, 0))
        left = tk.Frame(body, bg=WHITE)
        right = tk.Frame(body, bg=WHITE)
        left.pack(side="left", fill="both", expand=True)
        tk.Frame(body, bg=RULE, width=1).pack(side="left", fill="y", padx=18)
        right.pack(side="left", fill="both")

        tk.Label(left, text="MODE", bg=WHITE, fg=INK, font=(sans, 9, "bold")).pack(anchor="w")
        row = tk.Frame(left, bg=WHITE)
        row.pack(fill="x", pady=(8, 0))
        self._passive = _ModeKey(row, "PASSIVE", lambda: self._on_mode(PASSIVE), sans)
        self._track = _ModeKey(row, "TRACK", lambda: self._on_mode(TRACK), sans)
        self._recover = _ModeKey(row, "RECOVER", lambda: self._on_mode(RECOVER), sans)
        self._passive.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._track.pack(side="left", fill="x", expand=True, padx=6)
        self._recover.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._status = tk.Label(left, text="", bg=WHITE, fg=INK, font=(mono, 10), anchor="w")
        self._status.pack(fill="x", pady=(10, 0))

        tk.Label(left, text="COMMAND", bg=WHITE, fg=INK, font=(sans, 9, "bold")).pack(anchor="w", pady=(16, 0))
        self._frames = TargetSlider(
            left, "FRAMES", "rad", self._limits[FRAME],
            lambda value: self._on_bar(FRAME, value), self._on_drag, sans, mono)
        self._ears = TargetSlider(
            left, "EARS", "rad", self._limits[EAR],
            lambda value: self._on_bar(EAR, value), self._on_drag, sans, mono)
        self._wheels = TargetSlider(
            left, "WHEELS", "rad/s", self._limits[WHEEL],
            lambda value: self._on_bar(WHEEL, value), self._on_drag, sans, mono)
        for slider in (self._frames, self._ears, self._wheels):
            slider.pack(fill="x", pady=(8, 0))
        self._stand = _ModeKey(left, "STAND UP", self._on_stand, sans)
        self._stand.pack(fill="x", pady=(14, 0))

        tk.Label(right, text="IMU", bg=WHITE, fg=INK, font=(sans, 9, "bold")).pack(anchor="w")
        tk.Label(
            right, text="GRAY IS WORLD",
            bg=WHITE, fg=MUTED, font=(sans, 9)).pack(anchor="w", pady=(2, 6))
        self._axes = AxisView(right, sans)
        self._axes.pack(fill="x")
        numbers = tk.Frame(right, bg=WHITE)
        numbers.pack(fill="x", pady=(8, 0))
        self._roll = _angle(numbers, "ROLL", sans, mono)
        self._pitch = _angle(numbers, "PITCH", sans, mono)
        self._yaw = _angle(numbers, "YAW", sans, mono)

        self._band = tk.Frame(right, bg=WHITE, highlightthickness=1, highlightbackground=RULE)
        self._band.pack(fill="x", pady=(12, 0))
        self._state = tk.Label(self._band, text="UPRIGHT", bg=WHITE, fg=INK, font=(sans, 20, "bold"), pady=8)
        self._state.pack(fill="x")
        meta = tk.Frame(right, bg=WHITE)
        meta.pack(fill="x", pady=(8, 0))
        tilt_col = tk.Frame(meta, bg=WHITE)
        tilt_col.pack(side="left")
        tk.Label(tilt_col, text="TILT", bg=WHITE, fg=MUTED, font=(sans, 9)).pack(anchor="w")
        self._tilt = tk.Label(tilt_col, text="000.0°", bg=WHITE, fg=INK, font=(mono, 14))
        self._tilt.pack(anchor="w")
        limit = tk.Frame(meta, bg=WHITE)
        limit.pack(side="right")
        tk.Label(limit, text=f"FALLEN AT ≥ {FALLEN_ENTER_DEG:.0f}°", bg=WHITE, fg=INK, font=(sans, 9)).pack(anchor="e")
        tk.Label(limit, text=f"CLEARS BELOW {FALLEN_EXIT_DEG:.0f}°", bg=WHITE, fg=MUTED, font=(sans, 9)).pack(anchor="e")
        tk.Label(
            right, text="DOUBLE-CLICK A BODY, THEN CTRL-DRAG",
            bg=WHITE, fg=MUTED, font=(sans, 8)).pack(anchor="w", pady=(10, 0))


class _ModeKey(tk.Frame):
    def __init__(self, parent, text, command, sans):
        super().__init__(parent, bg=WHITE, highlightthickness=1, highlightbackground=RULE, cursor="hand2")
        self._command = command
        self._enabled = True
        self._active = False
        self._label = tk.Label(
            self, text=text, bg=WHITE, fg=INK, font=(sans, 12, "bold"),
            pady=12, cursor="hand2")
        self._label.pack(fill="both", expand=True)
        self.bind("<Button-1>", self._click)
        self._label.bind("<Button-1>", self._click)

    def _click(self, _event):
        if self._enabled:
            self._command()

    def set_active(self, active):
        self._active = active
        self._paint()

    def set_enabled(self, enabled):
        self._enabled = enabled
        cursor = "hand2" if enabled else "arrow"
        self.configure(cursor=cursor)
        self._label.configure(cursor=cursor)
        self._paint()

    def _paint(self):
        if not self._enabled:
            bg, fg = WHITE, "#b0b0b0"
        elif self._active:
            bg, fg = INK, WHITE
        else:
            bg, fg = WHITE, INK
        self.configure(bg=bg)
        self._label.configure(bg=bg, fg=fg)


class TargetSlider(tk.Frame):
    """One shared target for a joint pair, or one shared wheel speed."""

    def __init__(self, parent, title, unit, limits, on_value, on_drag, sans, mono):
        super().__init__(parent, bg=WHITE)
        self.low, self.high = limits
        self.unit = unit
        self._on_value = on_value
        self._on_drag = on_drag
        self._mono = mono
        self.dragging = False
        self.locked = False
        head = tk.Frame(self, bg=WHITE)
        head.pack(fill="x")
        tk.Label(head, text=title, bg=WHITE, fg=INK, font=(sans, 9, "bold")).pack(side="left")
        self._value = tk.Label(head, text="OFF", bg=WHITE, fg=INK, font=(mono, 11))
        self._value.pack(side="right")
        self._track = tk.Canvas(self, height=22, bg=WHITE, highlightthickness=0, bd=0, cursor="hand2")
        self._track.pack(fill="x", pady=(4, 0))
        self._shown = 0.0
        self._track.bind("<Button-1>", self._press)
        self._track.bind("<B1-Motion>", self._motion)
        self._track.bind("<ButtonRelease-1>", self._release)

    def show(self, value, locked: bool):
        self.locked = locked
        self._track.configure(cursor="arrow" if locked else "hand2")
        if isinstance(value, tuple):
            self._value.configure(text=_pair_readout(value[0], value[1], self.unit))
            shown = 0.5 * (value[0] + value[1])
        elif value is None:
            self._value.configure(text="OFF")
            shown = None
        else:
            self._value.configure(text=_scalar_readout(value, self.unit))
            shown = value
        if self.dragging:
            return
        self._shown = 0.0 if shown is None else shown
        self._draw(shown)

    def _press(self, event):
        if self.locked:
            return
        self.dragging = True
        self._on_drag(True)
        self._apply(event.x)

    def _motion(self, event):
        if self.dragging:
            self._apply(event.x)

    def _release(self, event):
        if not self.dragging:
            return
        self._apply(event.x)
        self.dragging = False
        self._on_drag(False)

    def _apply(self, x):
        value = bar_value(x, self._width(), self.low, self.high)
        self._shown = value
        self._draw(value)
        self._value.configure(text=_scalar_readout(value, self.unit))
        self._on_value(value)

    def _width(self):
        current = self._track.winfo_width()
        return current if current > 40 else 420

    def _draw(self, value):
        canvas = self._track
        canvas.delete("all")
        width = self._width()
        y = 11
        inset = 7
        canvas.create_line(inset, y, width - inset, y, fill="#b5b5b5" if self.locked else INK, width=2)
        zero = inset + bar_fraction(0.0, self.low, self.high) * (width - 2 * inset)
        canvas.create_line(zero, y - 6, zero, y + 6, fill="#d0d0d0", width=1)
        if value is None:
            return
        x = inset + bar_fraction(value, self.low, self.high) * (width - 2 * inset)
        color = "#b5b5b5" if self.locked else INK
        canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill=color, outline=color)


def bar_fraction(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def bar_value(x: float, width: int, low: float, high: float) -> float:
    inset = 7
    span = max(width - 2 * inset, 1)
    fraction = float(np.clip((x - inset) / span, 0.0, 1.0))
    return low + fraction * (high - low)


def _pair_readout(left: float, right: float, unit: str) -> str:
    if abs(left - right) < 5e-3:
        return _scalar_readout(left, unit)
    if unit == "rad":
        return f"L {left:+.3f}   R {right:+.3f} rad"
    return f"L {left:+.2f}   R {right:+.2f} {unit}"


def _scalar_readout(value: float, unit: str) -> str:
    if unit == "rad":
        return f"{value:+.3f} rad    {_signed_deg(value)}"
    return f"{value:+.2f} {unit}"


def _angle(parent, caption, sans, mono):
    column = tk.Frame(parent, bg=WHITE)
    column.pack(side="left", expand=True, fill="x")
    tk.Label(column, text=caption, bg=WHITE, fg=MUTED, font=(sans, 9)).pack(anchor="w")
    value = tk.Label(column, text="+000.0°", bg=WHITE, fg=INK, font=(mono, 14))
    value.pack(anchor="w", pady=(2, 0))
    return value


def _signed_deg(radians: float) -> str:
    degrees = round(float(np.degrees(radians)), 1)
    if degrees == 0:
        degrees = 0.0
    return f"{degrees:+06.1f}°"


def _frame_segments(rotation: np.ndarray, width: int, height: int):
    origin = _project(np.zeros(3), width, height)
    segments = []
    specs = (
        (np.array([0.70, 0.0, 0.0]), WORLD, 1, False, ""),
        (np.array([0.0, 0.70, 0.0]), WORLD, 1, False, ""),
        (np.array([0.0, 0.0, 0.70]), WORLD, 1, False, ""),
        (rotation @ np.array([0.96, 0.0, 0.0]), AXIS_COLOR["x"], 3, True, "X"),
        (rotation @ np.array([0.0, 0.96, 0.0]), AXIS_COLOR["y"], 3, True, "Y"),
        (rotation @ np.array([0.0, 0.0, 0.96]), AXIS_COLOR["z"], 3, True, "Z"),
    )
    for tip, color, stroke, arrow, label in specs:
        end = _project(tip, width, height)
        direction = np.array([end[0] - origin[0], end[1] - origin[1]], dtype=float)
        norm = np.linalg.norm(direction) or 1.0
        label_at = (end[0] + 14 * direction[0] / norm, end[1] + 14 * direction[1] / norm)
        segments.append({
            "start": (origin[0], origin[1]),
            "end": (end[0], end[1]),
            "origin": (origin[0], origin[1]),
            "color": color,
            "width": stroke,
            "arrow": arrow,
            "label": label,
            "label_at": label_at,
            "depth": end[2],
        })
    return segments


def _project(point: np.ndarray, width: int, height: int):
    eye = np.array([1.35, -1.10, 0.82])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = point - eye
    depth = float(np.dot(relative, forward))
    focal = 0.62 * min(width, height)
    x = width / 2 + focal * float(np.dot(relative, right)) / depth
    y = height / 2 - focal * float(np.dot(relative, up)) / depth
    return x, y, depth


def _arrow_head(canvas, start, end, color):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head, spread = 12.0, 4.5
    left = (end[0] - ux * head + px * spread, end[1] - uy * head + py * spread)
    right = (end[0] - ux * head - px * spread, end[1] - uy * head - py * spread)
    canvas.create_polygon(*end, *left, *right, fill=color, outline=color)
