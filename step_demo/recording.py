"""Batched diagnostic transfer; no policy/task/control decisions live here."""
import torch


def records_from_tensors(fields):
    """Copy a batch of terminal records once, preserving integer/bool types.

    Float64 packing retains the original float32 values and exact episode
    counters. Python conversion happens only after the single CPU transfer.
    """
    if not fields:
        return []
    n = len(next(iter(fields.values())))
    if not n:
        return []
    packed = torch.cat([value.reshape(n, -1).to(torch.float64)
                        for value in fields.values()], -1).detach().cpu()
    columns = {}
    offset = 0
    for name, value in fields.items():
        width = value[0].numel()
        columns[name] = packed[:, offset:offset+width].to(value.dtype).reshape(value.shape).tolist()
        offset += width
    return [{name: column[i] for name, column in columns.items()} for i in range(n)]
