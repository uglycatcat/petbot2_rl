"""Convert existing frozen bundles and check every ONNX output before publishing.

TorchScript files are conversion/reference inputs only, never runtime fallbacks.
Uses the ScriptModule exporter documented at https://docs.pytorch.org/docs/stable/onnx
and ONNX Runtime CPU: https://onnxruntime.ai/docs/api/python/api_summary.html
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import onnx
import onnxruntime
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from petbot_s2s.networks import OnnxNetwork


class ReflexOutputs(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, observation):
        return self.network(observation), self.network.base(observation)


def convert(bundle):
    manifest_path = bundle/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get('torchscript_sources', manifest['networks'])
    networks, errors = {}, {}
    generator = torch.Generator().manual_seed(20261028)
    for name, sha in sources.items():
        if not name.endswith('.pt'):
            raise ValueError(f'Expected frozen TorchScript input: {name}')
        source = bundle/name
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise ValueError(f'Source hash mismatch: {source}')
        original = torch.jit.load(str(source)).eval()
        for parameter in original.parameters():
            parameter.requires_grad_(False)
        reflex = name == 'balance_reflex.pt'
        exported = torch.jit.script(ReflexOutputs(original).eval()) if reflex else original
        output_names = ['action', 'base'] if reflex else ['action']
        path = source.with_suffix('.onnx')
        torch.onnx.export(exported, torch.zeros(1, 290), str(path),
            dynamo=False, opset_version=17, input_names=['observation'],
            output_names=output_names,
            dynamic_axes={key: {0: 'batch'} for key in ['observation', *output_names]})
        onnx.checker.check_model(str(path), full_check=True)
        converted = OnnxNetwork(path)
        maximum = {key: 0. for key in output_names}
        for observation in (torch.zeros(1, 290), torch.randn(1, 290, generator=generator),
                            torch.randn(1024, 290, generator=generator),
                            3.*torch.randn(1024, 290, generator=generator)):
            with torch.no_grad():
                expected = exported(observation)
                if not reflex:
                    expected = (expected,)
                for key, value in zip(output_names, expected):
                    actual = converted.evaluate(observation, key)
                    maximum[key] = max(maximum[key], float((actual-value).abs().max()))
                    # 1e-4 normalized action is <=0.0025 rad/s common-wheel
                    # target or 0.00011 rad frame target. Float32 GEMM fusion
                    # differs between JIT and ORT; closed-loop checks follow.
                    torch.testing.assert_close(actual, value, rtol=1e-5, atol=1e-4)
        errors[path.name] = maximum
        networks[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['format'] = 'petbot_cycle_onnx_bundle_v2'
    manifest['torchscript_sources'] = sources
    manifest['networks'] = networks
    manifest['onnx_conversion'] = dict(opset=17, torch=torch.__version__, onnx=onnx.__version__,
        onnxruntime=onnxruntime.__version__, provider='CPUExecutionProvider', samples_per_network=2050,
        max_abs_error=errors, tolerance=dict(atol=1e-4, rtol=1e-5), passed=True,
        scope='Numerical conversion only; not a closed-loop or operator acceptance result')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest['onnx_conversion']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, action='append')
    args = parser.parse_args()
    torch.set_num_threads(1)
    for bundle in args.policy or sorted((ROOT/'policies').glob('*/manifest.json')):
        if bundle.is_file():
            bundle = bundle.parent
        print(json.dumps(dict(bundle=bundle.name, result=convert(bundle)), indent=2))
