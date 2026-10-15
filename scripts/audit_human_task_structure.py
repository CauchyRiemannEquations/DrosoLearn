"""Freeze and audit source-coded task structure, without learning-performance estimates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/human_task_structure_v1.json"


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def safe_path(base, relative):
    require(type(relative) is str and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Unsafe relative input path")
    base = Path(base).resolve()
    path = (base / relative).resolve()
    require(path.is_relative_to(base), "Input escapes root")
    return path


def frozen_files(cfg):
    paths = cfg["frozen_files"]
    require(len(paths) == len(set(paths)), "Repeated frozen source path")
    evidence = json.loads(safe_path(ROOT, cfg["source_manifest"]).read_text(encoding="utf8"))
    evidence_root = (ROOT / cfg["source_manifest"]).parent
    result = {p: sha(safe_path(ROOT, p)) for p in paths}
    for entry in evidence["files"]:
        path = safe_path(evidence_root, entry["path"])
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "Primary evidence bytes differ")
        result[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(sorted(result.items()))


def inputs(cfg, cache):
    cache = Path(cache).resolve()
    require(not cache.is_relative_to(ROOT), "Human cache must remain external")
    origin = json.loads(safe_path(ROOT, cfg["origin_manifest"]).read_text(encoding="utf8"))
    parent = json.loads(safe_path(ROOT, cfg["origin_result"]).read_text(encoding="utf8"))
    require(parent["status"] == "pass" and parent["gate"]["passed"] is True
            and parent["total_values_compared"] == 75000 and parent["total_mismatches"] == 0,
            "Verified source-origin gate required")
    entries = [e for e in origin["source_files"] if e["role"] == "osf_raw"]
    require(len(entries) == 25 and len({e["path"] for e in entries}) == 25, "Fixed25 source selection required")
    checked = []
    for e in entries:
        p = safe_path(cache, e["path"])
        require(p.stat().st_size == e["bytes"] and sha(p) == e["sha256"], "Raw source bytes differ")
        checked.append({k: e[k] for k in ("path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")})
    return checked, origin


def inspect_raw(raw, trials=1000):
    shapes = {"A": (1, trials), "C": (1, trials), "S": (trials, 1), "R": (trials, 2)}
    require(all(k in raw and np.asarray(raw[k]).shape == shape for k, shape in shapes.items()), "Source shape differs")
    require(all(np.asarray(raw[k]).dtype.kind in "iuf" for k in shapes), "Numeric nonboolean arrays required")
    a, c, s, r = (np.asarray(raw[k], dtype=float) for k in ("A", "C", "S", "R"))
    require(all(not np.isinf(v).any() for v in (a, c, s, r)), "Infinite source values")
    a, c, s = a[0], c[0], s[:, 0]
    require(np.isin(a[~np.isnan(a)], [1, 2, 3]).all() and np.isin(c[~np.isnan(c)], [0, 1]).all()
            and np.isin(s, [1, 2]).all() and np.isin(r, [-1, 1]).all(), "Source codes differ")
    choice = np.isin(a, [1, 2])
    logic = bool(np.isfinite(c[choice]).all() and np.array_equal(c[choice], (a[choice] == s[choice]).astype(int)))
    other_missing = bool(np.isnan(c[~choice]).all())
    starts = [1, *(np.flatnonzero(s[1:] != s[:-1]) + 2).tolist()]
    lengths = np.diff([*starts, trials + 1]).tolist()
    signature = hashlib.sha256(s.astype("<i8").tobytes() + r.astype("<i8").tobytes()).hexdigest()
    return {"schema_passed": True, "c_equals_action_state_indicator": logic,
            "query_missing_c_remains_missing": other_missing,
            "state_segment_starts_1based": starts, "state_segment_lengths": lengths,
            "schedule_sha256_little_endian_int64_S_then_R": signature,
            "structural_passed": logic and other_missing}


def register(cache):
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    checked, _ = inputs(cfg, cache)
    return {"study_id": cfg["study_id"], "registration_type": "source_task_structure_audit",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": sha(CONFIG), "frozen_files": frozen_files(cfg), "inputs": checked,
            "notice": "Exploratory prospective private Git freeze; no behavioral estimates in registration"}


def run(cache, registration):
    started = datetime.now(timezone.utc).isoformat()
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    reg = json.loads(Path(registration).read_text(encoding="utf8"))
    require(reg["study_id"] == cfg["study_id"] and reg["config_sha256"] == sha(CONFIG)
            and reg["frozen_files"] == frozen_files(cfg), "Frozen task source/config differs")
    checked, origin = inputs(cfg, cache)
    require(checked == reg["inputs"], "Registered raw identities differ")
    rows = []
    for entry in checked:
        record = {"row_index": entry["derived_row_index"], "osf_file_id": entry["osf_file_id"],
                  "group": entry["derived_group"],
                  "schedule_group": "regular" if entry["derived_row_index"] < origin["groups"]["main"]["regular_rows"] else "irregular"}
        try:
            raw = loadmat(safe_path(cache, entry["path"]), variable_names=["A", "C", "S", "R"])
            record.update(inspect_raw(raw, cfg["trials_per_row"]))
        except Exception as exc:
            record.update(structural_passed=False, error_type=type(exc).__name__, error_message=str(exc))
        rows.append(record)
    schedules = {}
    for label in ("regular", "irregular"):
        selected = [r for r in rows if r["schedule_group"] == label]
        signatures = sorted({r.get("schedule_sha256_little_endian_int64_S_then_R", "FAILED") for r in selected})
        schedules[label] = {"rows": len(selected), "unique_schedules": len(signatures),
                            "identical_within_group": len(signatures) == 1 and "FAILED" not in signatures}
    passed = len(rows) == 25 and all(r["structural_passed"] for r in rows) and all(
        group["identical_within_group"] for group in schedules.values())
    return {"study_id": cfg["study_id"], "status": "pass" if passed else "failed",
            "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "registration_sha256": sha(registration), "source_git_commit": reg["source_git_commit"],
            "config_sha256": sha(CONFIG), "frozen_files": reg["frozen_files"], "inputs": checked,
            "rows": rows, "schedule_groups": schedules,
            "gate": {"source_coded_descriptive_metric_allowed": passed,
                     "exact_historical_task_execution_version_verified": False,
                     "direct_human_model_similarity_allowed": False},
            "human_accuracy_or_learning_curves_computed": False,
            "scope": "source-coded option/state structure, not behavioral success",
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "run"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Output already exists")
    require(args.mode != "run" or args.registration is not None, "Run requires registration")
    try:
        result = register(args.cache) if args.mode == "register" else run(args.cache, args.registration)
    except Exception as exc:
        result = {"study_id": "human_task_structure_v1", "status": "failed", "phase": args.mode,
                  "error_type": type(exc).__name__, "error_message": str(exc),
                  "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    write_new(args.out, result)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return int(result.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
