"""Run a frozen matched-prefix component-ablation experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drosolearn.learning_ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
