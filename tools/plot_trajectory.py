"""Plot original-model posture, support, controls and measured joint motion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recover.contract import POSITION_LOW, POSITION_HIGH, TILT_COS
from recover.model import POSITION_JOINTS, source_fingerprint


def plot(prefix, output):
    report = json.loads(prefix.with_suffix('.json').read_text())
    if report.get('backend') != 'original_mujoco':
        raise ValueError('Use an original-MuJoCo evaluation')
    if report['source_hash'] != source_fingerprint():
        raise ValueError('The source model differs from the evaluation')
    z = np.load(prefix.with_suffix('.npz'))
    t = (np.arange(len(z['up'])) + 1) * report['control_dt']
    output.parent.mkdir(parents=True, exist_ok=True)
    low, high = np.array(POSITION_LOW), np.array(POSITION_HIGH)
    targets = low + (z['action'][:, :, :5] + 1) * .5 * (high - low)
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    for pose in range(4):
        ax = axes[pose, 0]
        ax.plot(t, z['up'][:, pose], label='body up')
        ax.axhline(TILT_COS, color='gray', linestyle=':', label='upright threshold')
        ax.plot(t, np.minimum(z['hold'][:, pose], .5) / .5,
                label='stable hold / 0.5 s', alpha=.65)
        ax.set_ylim(-1.05, 1.1)
        ax.set_ylabel(f'pose_{pose+1:02d}\nposture / hold')
        status = 'recovered' if report['success'][pose] else 'no recovery'
        ax.set_title(status)
        if pose == 0:
            ax.legend(fontsize=8, loc='lower right')

        ax = axes[pose, 1]
        contacts = z['contacts'][:, pose].T.astype(float)
        ax.imshow(contacts, extent=(0, t[-1], 3.5, -.5), aspect='auto',
                  interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_yticks(range(4), ['front L', 'front R', 'rear L', 'rear R'])
        ax.set_title('wheel support (at least 80% of physics samples)')

        ax = axes[pose, 2]
        for wheel, color in enumerate(('tab:blue', 'tab:orange')):
            side = ('L', 'R')[wheel]
            ax.plot(t, z['wheel'][:, pose, wheel], color=color, label=f'{side} actual')
            ax.plot(t, 25*z['action'][:, pose, 5+wheel], color=color,
                    linestyle='--', alpha=.6, label=f'{side} target')
        ax.set_title('rear wheel speed (rad/s)')
        if pose == 0:
            ax.legend(fontsize=8, ncol=2)
        for ax in axes[pose]:
            ax.set_xlim(0, t[-1])
            if report['first_invalid_s'][pose] >= 0:
                ax.axvspan(report['first_invalid_s'][pose], t[-1], color='red', alpha=.12)
    for ax in axes[-1]:
        ax.set_xlabel('time (s)')
    fig.suptitle(f'{prefix.name}: four recorded starts on original MuJoCo model')
    fig.tight_layout()
    overview = output.with_name(output.name + '_overview.png')
    fig.savefig(overview, dpi=140)
    plt.close(fig)

    paths = [str(overview)]
    for pose in range(4):
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
        for j, ax in enumerate(axes.flat):
            if j < 5:
                ax.plot(t, z['q'][:, pose, j], label='actual')
                ax.plot(t, targets[:, pose, j], linestyle='--', alpha=.65, label='target')
                ax.axhline(low[j], color='gray', linestyle=':')
                ax.axhline(high[j], color='gray', linestyle=':')
                ax.set_title(POSITION_JOINTS[j])
                ax.set_ylabel('angle (rad)')
            else:
                ax.plot(t, np.linalg.norm(z['root'][:, pose, 7:10], axis=-1), label='linear (m/s)')
                ax.plot(t, np.linalg.norm(z['root'][:, pose, 10:13], axis=-1), label='angular (rad/s)')
                ax.set_title('body speed')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
            ax.set_xlim(0, t[-1])
        for ax in axes[-1]:
            ax.set_xlabel('time (s)')
        fig.suptitle(f'{prefix.name}: pose_{pose+1:02d} requested targets and measured motion')
        fig.tight_layout()
        path = output.with_name(output.name + f'_pose_{pose+1:02d}_joints.png')
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    print(json.dumps(paths), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(Path(args.evaluation), Path(args.output))
