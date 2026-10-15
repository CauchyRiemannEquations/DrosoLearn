"""Inspect pinned behavioral MAT schema; do not join people or estimate learning."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import numpy as np
import scipy
from scipy.io import loadmat


def inspect_mats(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
    items = [item for item in manifest["files"] if item["path"].endswith(".mat")]
    if len(items) != 2:
        raise ValueError("Exactly the two selected behavioral MAT files required")
    matrices, records = [], []
    for item in items:
        path = (source / item["path"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Unsafe source path")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        binding = re.search(r"SHA256E-s(\d+)--([a-f0-9]{64})", item["git_pointer"])
        if len(raw) != item["size_bytes"] or sha != item["sha256"] or not binding or int(binding[1]) != len(raw) or binding[2] != sha:
            raise ValueError("MAT source or annex hash differs")
        loaded = {key: value for key, value in loadmat(path, simplify_cells=True).items() if not key.startswith("__")}
        variables = {}
        for name, values in loaded.items():
            if not isinstance(values, np.ndarray) or values.dtype != object or values.ndim != 1:
                raise ValueError("Expected one-dimensional cell vector")
            cells = [np.asarray(value) for value in values]
            if any(value.ndim != 1 or not np.issubdtype(value.dtype, np.number) for value in cells):
                raise ValueError("Expected numeric trial vector in each cell")
            if any(not np.isfinite(value).all() for value in cells):
                raise ValueError("Nonfinite MAT measurement; preserve and investigate")
            variables[name] = {"cells": len(cells), "trial_vector_lengths": sorted({len(value) for value in cells}),
                               "cell_dtypes": sorted({str(value.dtype) for value in cells})}
            if any(token in name.lower() for token in ("action", "feedback", "expect")):
                variables[name]["codes"] = sorted(float(value) for value in np.unique(np.concatenate(cells)))
        matrices.append(loaded)
        records.append({"path": item["path"], "sha256": sha, "size_bytes": len(raw), "variables": variables})
    common = sorted(matrices[0].keys() & matrices[1].keys())
    matches = {}
    for key in common:
        left, right = matrices[0][key], matrices[1][key]
        matches[key] = len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))
    if not common or not all(matches.values()):
        raise ValueError("Common MAT variables do not agree")
    return {"scope": "schema_and_duplicate_variable_check_only", "files": records,
            "common_variable_equality": matches, "participant_id_mapping": "not_assumed_or_performed",
            "performance_estimates_computed": False, "human_model_similarity_tested": False,
            "runtime": {"numpy": np.__version__, "scipy": scipy.__version__}}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/human/stolz_2022_openneuro")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new inspection output")
    report = inspect_mats(args.source)
    report.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_manifest_sha256=hashlib.sha256((args.source / "source-manifest.json").read_bytes()).hexdigest(),
                  inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "schema_checked", "files": len(report["files"]), "participant_mapping": "not_performed"}))


if __name__ == "__main__":
    main()
