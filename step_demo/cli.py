"""Dependency-free scaffold inspection; no simulator or training side effects."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .control import map_action
from .model import read_limits, source_fingerprint
from .observations import OBSERVATION_FIELDS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command != "check":
        parser.error("Framework only: simulation backend, sensor estimator and PPO are not implemented; no run started")
    try:
        config = load_config(args.config, args.model_dir)
        limits = read_limits(config.model_dir)
        standstill = map_action((0., -1.), limits, config.control)
        result = dict(status="framework_only", physics_validated=False, training_implemented=False,
                      source_sha256=source_fingerprint(config.model_dir), limits=asdict(limits),
                      terrain=asdict(config.terrain), acceptance_provisional=asdict(config.acceptance),
                      action_dim=2, actor_observation_dim=len(OBSERVATION_FIELDS),
                      terrain_updates_per_episode=1, standstill_targets=asdict(standstill),
                      initialization="random_only", pretrained_models=[])
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
