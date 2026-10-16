"""Execute frozen read-only state decomposition without new learning."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drosolearn.state_decomposition import main

if __name__ == "__main__":
    raise SystemExit(main())
