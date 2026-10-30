"""Inspect the contract, or dispatch to the container-only Isaac/RSL-RL runtime."""

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .control import map_action
from .model import read_limits, source_fingerprint
from .observations import OBSERVATION_FIELDS


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('train','evaluate'):
        from .runtime import main as run
        return run(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.model_dir)
        limits = read_limits(config.model_dir)
        standstill = map_action((0., -1.), limits, config.control)
        result = dict(status="implementation_present", physics_validated="see engineering evidence", training_implemented=True,
                      source_sha256=source_fingerprint(config.model_dir), limits=asdict(limits),
                      terrain=asdict(config.terrain), acceptance_provisional=asdict(config.acceptance),
                      action_dim=2, actor_observation_dim=len(OBSERVATION_FIELDS),
                      terrain_updates_per_episode=1, standstill_targets=asdict(standstill),
                      initialization="random_only", pretrained_models=[])
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
