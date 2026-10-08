import argparse
import json
from pathlib import Path
from .experiment import run
from .artifacts import Recommender

def main():
    parser = argparse.ArgumentParser(description='Paper-aligned OULAD recommendation')
    commands = parser.add_subparsers(dest='command', required=True)
    experiment = commands.add_parser('experiment')
    experiment.add_argument('--data', required=True)
    experiment.add_argument('--output', required=True)
    experiment.add_argument('--epochs', type=int, default=100)
    experiment.add_argument('--seeds', type=int, nargs='+', default=[42])
    experiment.add_argument('--ablations', action='store_true')
    experiment.add_argument('--registration-mode', choices=['relative', 'absolute'], default='relative')
    experiment.add_argument('--presentation-starts', help='JSON mapping item or presentation to ISO date')
    experiment.add_argument('--training-config', help='JSON rank, batch_size, learning_rate, weight_decay, patience')
    experiment.add_argument('--missing-registration', choices=['error', 'exclude'], default='error')
    predict = commands.add_parser('recommend')
    predict.add_argument('--model', required=True)
    predict.add_argument('--profile', required=True, help='Path to demographic JSON')
    predict.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    try:
        if args.command == 'experiment':
            starts = json.loads(Path(args.presentation_starts).read_text()) if args.presentation_starts else None
            config = json.loads(Path(args.training_config).read_text()) if args.training_config else None
            report = run(args.data, args.output, args.seeds, args.epochs, args.ablations, args.registration_mode, starts, config, args.missing_registration)
            print(json.dumps([r['counts'] for r in report['runs']], indent=2))
        else:
            profile = json.loads(Path(args.profile).read_text(encoding='utf-8'))
            print(json.dumps(Recommender(args.model).recommend(profile, args.k), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')

if __name__ == '__main__':
    main()
