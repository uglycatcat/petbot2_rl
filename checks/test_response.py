import unittest
from upright.response import basic_response


def frames(mode='responsive', commands=None):
    commands = commands or [[.1, 0.], [-.1, 0.], [0., .3], [0., -.3], [0., 0.]]
    result = []
    for command in commands:
        for step in range(34):
            actual = [v*.8 for v in command]
            if mode == 'no_turn': actual[1] = 0.
            if mode == 'wrong_turn': actual[1] *= -1
            if mode == 'transient' and step < 8: actual = [1., 2.]
            result.append(dict(t=(len(result)+1)*.06, active=[True], phase_before=[2],
                               failure=[False], requested_command=[command], motion=[actual]))
    return result


class BasicResponseContracts(unittest.TestCase):
    def test_completion_without_turning_is_not_basic_control(self):
        self.assertEqual(basic_response(frames('no_turn'), 1, [True], .06)['basic_control_pass'], 0)
        self.assertEqual(basic_response(frames('wrong_turn'), 1, [True], .06)['basic_control_pass'], 0)

    def test_signed_response_with_settling_is_sufficient(self):
        for mode in ('responsive', 'transient'):
            self.assertEqual(basic_response(frames(mode), 1, [True], .06)['basic_control_pass'], 1)

    def test_missing_coverage_or_incomplete_cycle_is_not_passed(self):
        self.assertEqual(basic_response(frames(), 1, [False], .06)['basic_control_pass'], 0)
        self.assertEqual(basic_response(frames(commands=[[.1, 0.]]), 1, [True], .06)['basic_control_pass'], 0)

    def test_short_window_cannot_prove_response(self):
        result = basic_response(frames()[:12], 1, [True], .06)
        self.assertEqual(result['basic_control_pass'], 0)
        self.assertFalse(result['basic_response']['cases'][0]['windows'])
