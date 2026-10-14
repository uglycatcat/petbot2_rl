"""Physical key state, gated by the originating terminal's focused AX element.

Only the small set of driving keys is queried; no keyboard events are logged.
Terminal bytes identify the input surface, not the duration of key presses.
"""

import os
import select
import subprocess
import sys
import termios
import tty


class MacTerminalKeyboard:
    KEY_CODES = {'up': 126, 'down': 125, 'left': 123, 'right': 124,
                 'c': 8, 'space': 49, 'q': 12, 'w': 13,
                 'e': 14, 'r': 15, 'escape': 53}

    def __init__(self):
        import AppKit
        import ApplicationServices as AX
        import CoreFoundation as CF
        import Quartz
        self.AppKit, self.AX, self.CF, self.CG = AppKit, AX, CF, Quartz
        self.fd = sys.stdin.fileno()
        self.saved = None
        self.target = None
        self.previous = set()
        self.focused = False
        self.terminal_pid = self._terminal_ancestor()
        self.app = AX.AXUIElementCreateApplication(self.terminal_pid)
        AX.AXUIElementSetMessagingTimeout(self.app, 0.02)

    def _terminal_ancestor(self):
        pid = os.getppid()
        while pid > 1:
            app = self.AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if app is not None and app.bundleIdentifier():
                # An IDE can share one AX element between terminal tabs. Fail closed.
                if str(app.bundleIdentifier()) not in ('com.apple.Terminal', 'com.googlecode.iterm2'):
                    raise RuntimeError('请从 macOS 终端.app 或 iTerm2 启动，以可靠识别终端焦点。')
                return pid
            pid = int(subprocess.check_output(
                ['/bin/ps', '-o', 'ppid=', '-p', str(pid)], text=True).strip() or '1')
        raise RuntimeError('没有找到启动程序的终端，请在终端.app 中运行 python main.py。')

    def _focused_element(self):
        front = self.AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if front is None or front.processIdentifier() != self.terminal_pid:
            return None
        error, element = self.AX.AXUIElementCopyAttributeValue(
            self.app, self.AX.kAXFocusedUIElementAttribute, None)
        return element if error == 0 else None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('需要交互式终端，不能重定向标准输入。')
        ax_ok = self.AX.AXIsProcessTrusted()
        input_ok = self.CG.CGPreflightListenEventAccess()
        if not ax_ok or not input_ok:
            if not ax_ok:
                self.AX.AXIsProcessTrustedWithOptions({self.AX.kAXTrustedCheckOptionPrompt: True})
            if not input_ok:
                self.CG.CGRequestListenEventAccess()
            raise RuntimeError(
                '精确键盘控制需要系统权限：系统设置 → 隐私与安全性 → '
                '辅助功能 / 输入监控。请允许系统提示的终端或 MuJoCo (mjpython)，'
                '然后退出并重新打开终端，再运行本程序。')
        self.target = self._focused_element()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self):
        received = False
        while select.select([self.fd], [], [], 0)[0]:
            if not os.read(self.fd, 4096):
                raise EOFError('终端输入已关闭。')
            received = True
        element = self._focused_element()
        # If the viewer took focus during startup, bind on the first real stdin key.
        if self.target is None and received and element is not None:
            self.target = element
        self.focused = bool(element is not None and self.target is not None
                            and self.CF.CFEqual(element, self.target))
        keys = set()
        if self.focused:
            state = self.CG.kCGEventSourceStateCombinedSessionState
            # Don't drive on terminal shortcuts such as Cmd+Left or Ctrl+C.
            modifiers = (54, 55, 58, 61, 59, 62)
            if not any(self.CG.CGEventSourceKeyState(state, code) for code in modifiers):
                keys = {name for name, code in self.KEY_CODES.items()
                        if self.CG.CGEventSourceKeyState(state, code)}
        pressed = keys - self.previous
        self.previous = keys
        return keys, pressed, self.focused

    def __exit__(self, *_):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            termios.tcflush(self.fd, termios.TCIFLUSH)
        print()
