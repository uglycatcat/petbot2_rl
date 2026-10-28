"""ONNX-only CPU inference; tensors are retained for the shared control math."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class OnnxNetwork(torch.nn.Module):
    def __init__(self, path: Path):
        super().__init__()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or inputs[0].shape[1:] != [290]:
            raise ValueError(f'Unexpected ONNX observation contract: {path}')
        self.input_name = inputs[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]
        if self.output_names not in (['action'], ['action', 'base']):
            raise ValueError(f'Unexpected ONNX output contract: {path}')

    def evaluate(self, observation, output):
        if observation.device.type != 'cpu' or observation.ndim != 2 or observation.shape[1] != 290:
            raise ValueError('ONNX observations must be CPU [batch, 290] tensors')
        array = np.ascontiguousarray(observation.detach().numpy(), dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite ONNX observation')
        result, = self.session.run([output], {self.input_name: array})
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite ONNX output')
        return torch.from_numpy(result)

    def forward(self, observation):
        return self.evaluate(observation, 'action')

    def base(self, observation):
        if 'base' not in self.output_names:
            raise ValueError('This network has no frozen base output')
        return self.evaluate(observation, 'base')
