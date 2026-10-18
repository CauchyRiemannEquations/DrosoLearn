"""Exclusive registration and exact local-event time-mixture calibration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drosolearn.time_mixture import main

if __name__ == "__main__":
    raise SystemExit(main())
