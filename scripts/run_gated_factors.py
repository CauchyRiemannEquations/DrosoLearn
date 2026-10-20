"""Register or execute the read-only descriptive gated-factor accounting once."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drosolearn.gated_factors_envelope import register, require, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register', 'run'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--publication-proof', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.mode != 'run' or (args.registration is not None and args.publication_proof is not None),
            'Analysis requires a registration and published byte proof')
    if args.mode == 'register':
        require(args.registration is None and args.publication_proof is None,
                'Registration mode has no prior registration/publication inputs')
        result = register(args.root, args.config, args.out)
    else:
        result = run(args.root, args.config, args.registration, args.publication_proof, args.out)
    print(json.dumps({'status': result.get('status', 'registered'), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
