from pathlib import Path
import argparse
import json
from .experiment import register, run
from .analysis import analyze

def main():
    parser = argparse.ArgumentParser(description="DrosoLearn reproducible research pipeline")
    parser.add_argument("command", choices=("register", "run", "analyze", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    for name in ("config", "registration", "out"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (value if value.is_absolute() else args.root / value).resolve())
    if args.command in ("register", "run") and args.config is None:
        parser.error("--config is required")
    if args.command == "register":
        result = register(args.root, args.config, args.out)
        print(json.dumps({"registered_at_utc": result["registered_at_utc"], "config_sha256": result["config_sha256"]}))
    elif args.command == "run":
        if args.registration is None:
            parser.error("--registration is required")
        result = run(args.root, args.config, args.registration, args.out)
        analyze(args.out)
        print(json.dumps({"completed_seed_conditions": len(result), "out": str(args.out)}))
    elif args.command == "analyze":
        print(json.dumps(analyze(args.out)))
    else:
        from .audit import verify
        result = verify(args.out)
        print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)

if __name__ == "__main__":
    main()
