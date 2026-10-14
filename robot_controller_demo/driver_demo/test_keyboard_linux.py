"""Linux keyboard tests; no X server, root input access or GUI is required."""

import ctypes as C
import unittest
from unittest.mock import Mock, patch

from keyboard_linux import LinuxTerminalKeyboard, X11KeyboardState


class LinuxKeyboardTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.backend.snapshot.return_value = (42, set(), False)
        with patch('keyboard_linux.sys.stdin') as stdin:
            stdin.fileno.return_value = 0
            self.keyboard = LinuxTerminalKeyboard(self.backend)

    def poll(self, chunk=b''):
        ready = [([0], [], []), ([], [], [])] if chunk else [( [], [], [])]
        with patch('keyboard_linux.select.select', side_effect=ready), \
                patch('keyboard_linux.os.read', return_value=chunk):
            return self.keyboard.poll()

    def test_bind_only_on_stdin_control_and_detect_physical_release(self):
        self.backend.snapshot.return_value = (42, {'up', 'left'}, False)
        self.assertEqual(self.poll(), (set(), set(), False))
        self.assertEqual(self.poll(b'\x1b[A'), ({'up', 'left'}, {'up', 'left'}, True))
        self.assertEqual(self.poll(), ({'up', 'left'}, set(), True))
        self.backend.snapshot.return_value = (42, set(), False)
        self.assertEqual(self.poll(), (set(), set(), True))

    def test_focus_loss_viewer_other_window_and_return(self):
        self.backend.snapshot.return_value = (42, {'up'}, False)
        self.poll(b'\x1b[A')
        self.backend.snapshot.return_value = (99, {'up', 'q'}, False)
        self.assertEqual(self.poll(), (set(), set(), False))
        self.backend.snapshot.return_value = (42, {'up'}, False)
        self.assertEqual(self.poll(), ({'up'}, {'up'}, True))

    def test_terminal_tab_focus_reports_and_split_sequences(self):
        self.backend.snapshot.return_value = (42, {'up'}, False)
        self.poll(b'\x1b[A')
        self.poll(b'\x1b[')
        self.assertEqual(self.poll(b'O'), (set(), set(), False))
        self.assertEqual(self.poll(), (set(), set(), False))
        self.assertEqual(self.poll(b'\x1b[I'), ({'up'}, {'up'}, True))

    def test_focus_reports_do_not_bind_a_viewer(self):
        self.poll(b'\x1b[I')
        self.assertIsNone(self.keyboard.target)
        self.backend.snapshot.return_value = (1, {'up'}, False)
        self.assertFalse(self.poll(b'\x1b[A')[2])

    def test_shortcuts_do_not_drive_or_quit(self):
        self.backend.snapshot.return_value = (42, {'up'}, False)
        self.poll(b'\x1b[A')
        self.backend.snapshot.return_value = (42, {'up', 'q', 'c'}, True)
        self.assertEqual(self.poll(b'\x1bq'), (set(), set(), True))

    def test_short_command_taps_and_application_cursor_sequences(self):
        self.assertEqual(self.poll(b'\x1bOAc'), (set(), {'c'}, True))
        self.assertEqual(self.poll(b'c'), (set(), {'c'}, True))

    def test_leg_keys_use_physical_state_and_can_bind_terminal(self):
        self.backend.snapshot.return_value = (42, {'q', 'w', 'e', 'r'}, False)
        self.assertEqual(self.poll(b'qwer')[0], {'q', 'w', 'e', 'r'})
        self.backend.snapshot.return_value = (42, set(), False)
        self.assertEqual(self.poll(b'qwer'), (set(), set(), True))

    def test_escape_exits_before_binding_and_does_not_poison_next_sequence(self):
        self.backend.snapshot.return_value = (42, {'escape'}, False)
        self.assertEqual(self.poll(b'\x1b'), ({'escape'}, {'escape'}, True))
        self.assertEqual(self.keyboard.pending, b'')

    def test_held_mode_key_toggles_once_despite_terminal_repeat(self):
        self.backend.snapshot.return_value = (42, {'c'}, False)
        self.assertEqual(self.poll(b'c')[1], {'c'})
        self.assertEqual(self.poll(b'cc')[1], set())
        self.assertEqual(self.poll()[1], set())
        self.backend.snapshot.return_value = (42, set(), False)
        self.poll()
        self.backend.snapshot.return_value = (42, {'c'}, False)
        self.assertEqual(self.poll(b'c')[1], {'c'})

    def test_unknown_escape_sequences_do_not_become_commands(self):
        self.assertEqual(self.keyboard._consume(b'\x1b[1;5A\x1bq'), set())
        self.assertEqual(self.keyboard._consume(b'\x1b'), set())
        self.assertEqual(self.keyboard._consume(b'[D '), {'left', 'space'})

    def test_eof_terminates_instead_of_leaving_commands_active(self):
        with patch('keyboard_linux.select.select', return_value=([0], [], [])), \
                patch('keyboard_linux.os.read', return_value=b''):
            with self.assertRaises(EOFError):
                self.keyboard.poll()

    def test_terminal_restored_and_display_closed_after_exception(self):
        with patch('keyboard_linux.sys.stdin') as stdin, \
                patch('keyboard_linux.sys.stdout') as stdout, \
                patch('keyboard_linux.termios.tcgetattr', return_value=['saved']), \
                patch('keyboard_linux.termios.tcsetattr') as restore, \
                patch('keyboard_linux.termios.tcflush'), \
                patch('keyboard_linux.tty.setcbreak'):
            stdin.isatty.return_value = True
            with self.assertRaisesRegex(RuntimeError, 'viewer failed'):
                with self.keyboard:
                    raise RuntimeError('viewer failed')
            restore.assert_called_once_with(0, unittest.mock.ANY, ['saved'])
            self.backend.close.assert_called_once()
            stdout.write.assert_any_call('\x1b[?1004h')
            stdout.write.assert_any_call('\x1b[?1004l')


class X11StateTests(unittest.TestCase):
    def test_bitmask_keys_modifiers_and_focus(self):
        backend = X11KeyboardState.__new__(X11KeyboardState)
        backend.lib = Mock()
        backend.display = 7
        backend.keycodes = {'up': 111, 'left': 113, 'q': 24}
        backend.modifiers = {37, 0}

        def focus(display, window, revert):
            C.cast(window, C.POINTER(C.c_ulong))[0] = 42

        def keymap(display, state):
            for code in (111, 113, 37):
                state[code // 8] |= 1 << (code % 8)

        backend.lib.XGetInputFocus.side_effect = focus
        backend.lib.XQueryKeymap.side_effect = keymap
        self.assertEqual(backend.snapshot(), (42, {'up', 'left'}, True))
        backend.close()
        backend.close()
        backend.lib.XCloseDisplay.assert_called_once_with(7)


if __name__ == '__main__':
    unittest.main()
