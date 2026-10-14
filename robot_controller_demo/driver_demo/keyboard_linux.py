"""X11 physical key state, gated by the terminal window and focus reports.

Terminal bytes bind the input surface; they never determine how long a driving
key is held. Only the configured control keys and modifiers are queried.
"""

import ctypes as C
from ctypes.util import find_library
import os
import select
import sys
import termios
import tty


class X11KeyboardState:
    KEY_NAMES = {'up': 'Up', 'down': 'Down', 'left': 'Left', 'right': 'Right',
                 'space': 'space', 'c': 'c', 'q': 'q', 'w': 'w',
                 'e': 'e', 'r': 'r', 'escape': 'Escape'}
    MODIFIERS = ('Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Meta_L', 'Meta_R',
                 'Super_L', 'Super_R', 'ISO_Level3_Shift')

    def __init__(self):
        try:
            self.lib = C.CDLL(find_library('X11') or 'libX11.so.6')
        except OSError as error:
            raise RuntimeError('当前运行环境缺少 X11 共享库（libX11.so.6）。') from error
        signatures = {
            'XInitThreads': ([], C.c_int),
            'XOpenDisplay': ([C.c_char_p], C.c_void_p),
            'XCloseDisplay': ([C.c_void_p], C.c_int),
            'XStringToKeysym': ([C.c_char_p], C.c_ulong),
            'XKeysymToKeycode': ([C.c_void_p, C.c_ulong], C.c_ubyte),
            'XQueryKeymap': ([C.c_void_p, C.POINTER(C.c_ubyte)], C.c_int),
            'XGetInputFocus': ([C.c_void_p, C.POINTER(C.c_ulong),
                               C.POINTER(C.c_int)], C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.lib.XInitThreads()  # The MuJoCo viewer also uses X11 on another thread.
        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError(
                '无法连接 X11 桌面。请检查 DISPLAY、XAUTHORITY 和 /tmp/.X11-unix '
                '挂载，并从同一桌面的本地 X11 终端进入容器。')
        self.keycodes = {name: self._keycode(symbol)
                         for name, symbol in self.KEY_NAMES.items()}
        self.modifiers = {self._keycode(symbol) for symbol in self.MODIFIERS}
        if not all(self.keycodes.values()):
            self.close()
            raise RuntimeError('当前 X11 键盘映射缺少必要的方向键或控制键。')

    def _keycode(self, symbol):
        return int(self.lib.XKeysymToKeycode(
            self.display, self.lib.XStringToKeysym(symbol.encode('ascii'))))

    def snapshot(self):
        window, revert = C.c_ulong(), C.c_int()
        self.lib.XGetInputFocus(self.display, C.byref(window), C.byref(revert))
        state = (C.c_ubyte * 32)()
        self.lib.XQueryKeymap(self.display, state)

        def held(code):
            return bool(code and state[code // 8] & (1 << (code % 8)))

        modified = any(held(code) for code in self.modifiers)
        keys = {name for name, code in self.keycodes.items() if held(code)}
        return window.value, keys, modified

    def close(self):
        if self.display:
            self.lib.XCloseDisplay(self.display)
            self.display = None


class LinuxTerminalKeyboard:
    # Normal and application-cursor modes, plus xterm-compatible focus reports.
    SEQUENCES = {b'\x1b[A': 'up', b'\x1b[B': 'down',
                 b'\x1b[C': 'right', b'\x1b[D': 'left',
                 b'\x1bOA': 'up', b'\x1bOB': 'down',
                 b'\x1bOC': 'right', b'\x1bOD': 'left',
                 b'\x1b[I': 'focus_in', b'\x1b[O': 'focus_out'}

    def __init__(self, backend=None):
        self.backend = backend
        self.fd = sys.stdin.fileno()
        self.saved = None
        self.target = None
        self.previous = set()
        self.focused = False
        self.terminal_focused = True
        self.pending = b''

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('需要交互式终端；请使用 docker exec -it 进入容器。')
        self.backend = self.backend or X11KeyboardState()
        try:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)  # Preserve Ctrl+C / SIGINT.
            sys.stdout.write('\x1b[?1004h')
            sys.stdout.flush()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _consume(self, chunk):
        self.pending += chunk
        received = set()
        while self.pending:
            sequence = next((s for s in self.SEQUENCES
                             if self.pending.startswith(s)), None)
            if sequence is not None:
                name = self.SEQUENCES[sequence]
                self.pending = self.pending[len(sequence):]
                if name.startswith('focus_'):
                    self.terminal_focused = name == 'focus_in'
                else:
                    received.add(name)
                continue
            if any(s.startswith(self.pending) for s in self.SEQUENCES):
                break  # A report / arrow sequence can span multiple reads.
            if self.pending.startswith(b'\x1b['):
                end = next((i for i, value in enumerate(self.pending[2:], 2)
                            if 0x40 <= value <= 0x7e), None)
                if end is None:
                    if len(self.pending) > 64:
                        self.pending = b''
                    break
                self.pending = self.pending[end + 1:]
                continue  # Ignore modified keys and other terminal reports.
            if self.pending.startswith(b'\x1b'):
                self.pending = self.pending[2:]  # Alt+key is not a command.
                continue
            char, self.pending = self.pending[:1], self.pending[1:]
            if char in (b' ', b'c', b'q', b'w', b'e', b'r'):
                received.add('space' if char == b' ' else char.decode('ascii'))
        return received

    def poll(self):
        received = set()
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 4096)
            if not chunk:
                raise EOFError('终端输入已关闭。')
            received.update(self._consume(chunk))
        window, keys, modified = self.backend.snapshot()
        # A physical Escape distinguishes the standalone key from the prefix of
        # an arrow/focus sequence, and permits exiting before a driving key binds.
        if 'escape' in keys and self.pending == b'\x1b':
            self.pending = b''
            received.add('escape')
        # A real stdin control key identifies the originating terminal. Opening
        # the viewer may take focus; never bind just to whatever window is active.
        if (self.target is None and received and not modified
                and self.terminal_focused and window not in (0, 1)):
            self.target = window
        self.focused = bool(self.target is not None and window == self.target
                            and self.terminal_focused)
        if not self.focused or modified:
            keys = set()
            received = set()
        # Preserve brief command taps, but use physical state for driving/release.
        pressed = (keys | (received & {'escape', 'c'})) - self.previous
        self.previous = keys
        return keys, pressed, self.focused

    def __exit__(self, *_):
        try:
            sys.stdout.write('\x1b[?1004l')
            sys.stdout.flush()
        finally:
            try:
                if self.saved is not None:
                    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
                    termios.tcflush(self.fd, termios.TCIFLUSH)
                    self.saved = None
            finally:
                if self.backend is not None:
                    self.backend.close()
        print()
