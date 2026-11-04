"""Export only an explicitly identified restart-run checkpoint; never deploy automatically."""
import argparse
import json
from pathlib import Path
from upright.config import Config, require_execution
from upright.experiment import identities, check_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_execution('engineering')
    manifest = json.loads((args.run_dir/'run.json').read_text())
    cfg = Config(**manifest['config']).validate()
    check_identity(manifest, dict(manifest, **identities(cfg)))
    if args.checkpoint.resolve().parent != args.run_dir.resolve()/'checkpoints':
        raise ValueError('Checkpoint must belong to this restart run')
    import torch
    from upright.export import make_actor, export_actor
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    check_identity(saved.get('restart', {}), manifest)
    actor = make_actor(cfg)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    result = export_actor(actor, cfg, args.output,
                          dict(run_id=manifest['run_id'], checkpoint=args.checkpoint.name,
                               updates_completed=saved['restart']['updates_completed'],
                               code_sha256=manifest['code_sha256'], model_sha256=manifest['model_sha256']))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
