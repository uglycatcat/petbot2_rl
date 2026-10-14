"""Run `python main.py` in a Linux/X11 or macOS graphical terminal."""

import argparse
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='PetBot 终端键盘控制（Linux / macOS）')
    parser.add_argument('--max-speed', type=float, default=0.35, help='最高目标线速度 m/s')
    parser.add_argument('--max-yaw-rate', type=float, default=1.5, help='最高目标角速度 rad/s')
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_speed, args.max_yaw_rate)):
        parser.error('速度上限必须是大于零的有限数值')
    if sys.platform not in ('linux', 'darwin'):
        parser.error('终端键盘控制支持 Linux / X11 和 macOS')
    if not sys.stdin.isatty():
        parser.error('请在交互式终端中启动')

    import mujoco
    import mujoco.viewer

    # macOS must keep the Cocoa main thread free. Reuse this Python environment.
    if sys.platform == 'darwin' and mujoco.viewer._MJPYTHON is None:
        launcher = Path(sys.executable).parent / 'mjpython'
        if not launcher.exists():
            raise RuntimeError('当前环境缺少 mjpython，请安装 mujoco。')
        os.execv(str(launcher), [str(launcher), str(Path(__file__).resolve()), *sys.argv[1:]])

    from control import DriveConfig, RobotController, configure_simulation
    if sys.platform == 'linux':
        from keyboard_linux import LinuxTerminalKeyboard as TerminalKeyboard
    else:
        from keyboard_macos import MacTerminalKeyboard as TerminalKeyboard

    model_path = Path(__file__).resolve().parents[2] / 'robot_description/mjcf/scene.xml'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    configure_simulation(model)
    data = mujoco.MjData(model)
    controller = RobotController(model, DriveConfig(
        max_speed=args.max_speed, max_yaw_rate=args.max_yaw_rate))
    dt = model.opt.timestep
    mujoco.mj_forward(model, data)

    with TerminalKeyboard() as keyboard:
        print('↑↓ 前后 | ←→ 转向 | 空格 刹车 | c 普通/控腿 | Esc / Ctrl+C 退出')
        print('控腿：Q/W 左车架 −/+ | E/R 右车架 −/+；松键保持，切回普通模式平滑回零。')
        print('点击启动终端并按控制键操控；离开本终端自动减速、停止调腿。')
        with mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.10]
                viewer.cam.distance = 0.65
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -22
            last = time.monotonic()
            accumulator = 0.0
            next_render = last
            next_status = last
            while viewer.is_running():
                now = time.monotonic()
                # Don't apply a long catch-up burst after sleep/debugger/permission UI.
                accumulator += min(now - last, 0.05)
                last = now
                keys, pressed, focused = keyboard.poll()
                if 'escape' in pressed:
                    break
                with viewer.lock():
                    controller.handle_pressed(pressed, active=focused)
                    while accumulator >= dt:
                        controller.step(data, keys, dt, active=focused)
                        mujoco.mj_step(model, data)
                        accumulator -= dt
                    # Follow position without overwriting mouse-controlled view angles.
                    viewer.cam.lookat[:] = data.xpos[model.body('lower_body').id]
                if now >= next_render:
                    viewer.sync()
                    next_render = now + 1 / 60
                if now >= next_status:
                    status = '控制中' if focused else '未聚焦：减速停车'
                    left, right = map(math.degrees, controller.frame_targets)
                    print(f'\r\033[2K{controller.mode_label} | {status} | '
                          f'目标 v={controller.drive.speed:+.2f} m/s '
                          f'ω={controller.drive.yaw_rate:+.2f} rad/s | '
                          f'车架目标 L={left:+.1f}° R={right:+.1f}°',
                          end='', flush=True)
                    next_status = now + 0.2
                time.sleep(0.004)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已退出。')
    except (RuntimeError, EOFError) as error:
        print(f'\n{error}', file=sys.stderr)
        sys.exit(1)
