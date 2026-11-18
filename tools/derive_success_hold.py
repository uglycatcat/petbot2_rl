"""Preserve a searched prefix through success, then hold its last target."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from recover.contract import CONTROL_DT
from recover.model import source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--witness', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    witness, evaluation, output = map(Path, (args.witness, args.evaluation, args.output))
    metadata = json.loads(witness.with_suffix('.json').read_text())
    report = json.loads(evaluation.with_suffix('.json').read_text())
    if metadata['source_hash'] != source_fingerprint() or report['source_hash'] != metadata['source_hash']:
        raise ValueError('Model source differs')
    if report.get('pose_ids') != [metadata['pose_id']] or not report['success_during_recorded_sequence'][0]:
        raise ValueError('An independently successful recorded prefix is required')
    archive = np.load(witness.with_suffix('.npz'))
    actions = archive['actions']
    trace = np.load(evaluation.with_suffix('.npz'))
    source_sha = hashlib.sha256(actions.tobytes()).hexdigest()
    if source_sha != metadata['action_sha256'] or not np.array_equal(trace['action'][:len(actions), 0], actions):
        raise ValueError('Witness checksum or independent replay targets differ')
    source_first = float(metadata.get('metrics', [0, -1])[1])
    end = max(float(report['first_success_s'][0]), source_first)
    steps = round(end / CONTROL_DT)
    if steps < 1 or steps > len(actions) or abs(steps * CONTROL_DT - end) > 1e-5:
        raise ValueError('Success must be within the recorded control horizon')
    if output.with_suffix('.json').exists() or output.with_suffix('.npz').exists():
        raise ValueError('Never overwrite a candidate')
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = actions[:steps].copy()
    np.savez_compressed(output.with_suffix('.npz'), actions=selected,
                        initial_root=archive['initial_root'], initial_joint=archive['initial_joint'])
    metadata.update(steps=steps, action_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                    status='derived_candidate_requires_independent_replay', derived_from=str(witness),
                    derivation=dict(method='hold_last_target_after_success_in_search_and_native_replay',
                                    evidence=str(evaluation), cutoff_s=steps * CONTROL_DT,
                                    source_action_sha256=source_sha,
                                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                    metrics_scope='Original full searched sequence; derived prefix requires its own reports')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata['derivation']), flush=True)


if __name__ == '__main__':
    main()
