"""Freeze or execute one bounded follow-up without replacing completed studies."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from drosolearn.prospective import register, require, run

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['register', 'run'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--registration', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    require(not a.out.exists(), 'New output path required')
    require(a.mode != 'run' or a.registration is not None, 'Registration required')
    result = register(a.root.resolve(), a.config, a.cache, a.out) if a.mode == 'register' else run(
        a.root.resolve(), a.config, a.registration, a.cache, a.out)
    print(json.dumps({'status': result.get('status', 'registered'), 'out': str(a.out)}))

if __name__ == '__main__': main()
