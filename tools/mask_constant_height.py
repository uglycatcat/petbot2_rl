"""Inference-only ablation of the zero-variance terrain-height scalar.
The scene and every relative-geometry input stay unchanged. No weights are trained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import torch


class ConstantHeightFeature(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self.register_buffer('training_height', raw.normalizer._mean[:, 25:26].clone())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.clone()
        x[:, 25:26] = self.training_height
        return self.raw(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = torch.jit.load(str(args.policy), map_location='cpu').eval()
    std = float(raw.normalizer._std[0, 25])
    if std > 1e-6:
        raise ValueError('Height varied during training; this ablation is not applicable')
    args.output.mkdir(parents=True, exist_ok=False)
    policy = torch.jit.script(ConstantHeightFeature(raw).eval())
    target = args.output / 'policy_raw.pt'
    policy.save(str(target))
    report = {'source_policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
              'policy_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'training_height_mean': float(raw.normalizer._mean[0, 25]),
              'training_height_std': std, 'normalization_epsilon': float(raw.normalizer.eps),
              'ignored_input_index': 25, 'weights_updated': False,
              'relative_geometry_inputs_unchanged': True, 'task_success_verified': False}
    (args.output / 'adaptation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
