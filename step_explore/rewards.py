"""Episode-progress reward based on simultaneous physical wheel support."""
import torch

TOP_PEAK_VALUES = (0.0, 3.0, 6.0, 12.0, 18.0)

def simultaneous_top_bonus(top, previous_peak, levels):
    peak = torch.maximum(previous_peak, top.sum(dim=-1))
    return levels[peak] - levels[previous_peak], peak
