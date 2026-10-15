"""Audit two public candidate caches without estimating human learning outcomes.

Raw files stay outside this repository. Upstream R/Python scripts are not run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
from scipy.io import loadmat

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY / "configs/human_candidate_source_manifest.json"
TEFITS = "cochrane_2019_tefits_subset"
PYBEFIT = "markovic_2022_pybefit"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def external_cache(path, repository=REPOSITORY):
    resolved = Path(path).resolve()
    require(not resolved.is_relative_to(Path(repository).resolve()),
            "Raw candidate cache must be outside the DrosoLearn repository")
    return resolved


def cache_file(cache, source_id, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "Manifest path must use nonempty relative POSIX paths")
    parts = Path(relative).parts
    require(not Path(relative).is_absolute() and ".." not in parts,
            "Manifest path escapes source cache")
    base = (Path(cache) / source_id).resolve()
    require(base.is_relative_to(Path(cache).resolve()), "Source cache symlink escapes cache")
    target = (base / relative).resolve()
    require(target.is_relative_to(base), "Cache symlink or path escapes source cache")
    return target


def verified_bytes(path, entry):
    raw = Path(path).read_bytes()
    require(len(raw) == entry["bytes"], f"Size mismatch: {entry['path']}")
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == entry["sha256"], f"SHA-256 mismatch: {entry['path']}")
    return raw


def prepare_cache(cache, manifest, fetch=False):
    records = []
    require([s["id"] for s in manifest["sources"]] == [TEFITS, PYBEFIT],
            "Unexpected candidate source IDs or order")
    for source in manifest["sources"]:
        seen = set()
        for entry in source["files"]:
            require(entry["path"] not in seen, "Duplicate manifest file")
            seen.add(entry["path"])
            prefix = ("https://raw.githubusercontent.com/" +
                      source["repository"].removeprefix("https://github.com/") +
                      "/" + source["commit"] + "/")
            require(entry["url"] == prefix + entry["path"],
                    "Only exact immutable upstream raw URLs are allowed")
            path = cache_file(cache, source["id"], entry["path"])
            if not path.exists():
                require(fetch, f"Missing cache file: {source['id']}/{entry['path']}; use --fetch")
                with urllib.request.urlopen(entry["url"], timeout=45) as response:
                    raw = response.read(entry["bytes"] + 1)
                require(len(raw) == entry["bytes"], f"Downloaded size mismatch: {entry['path']}")
                require(hashlib.sha256(raw).hexdigest() == entry["sha256"],
                        f"Downloaded SHA-256 mismatch: {entry['path']}")
                path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also protects a concurrent cache writer.
                with path.open("xb") as handle:
                    handle.write(raw)
            verified_bytes(path, entry)
            records.append({"source": source["id"], **entry})
    return records


def audit_tefits_frames(full, single, expected):
    columns = expected["columns"]
    require(list(map(str, full.columns)) == columns and
            list(map(str, single.columns)) == columns, "TEfits columns differ")
    identifiers = [str(x) for x in full["subID"].unique()]
    require(sorted(identifiers) == expected["subIDs"], "TEfits participant keys differ")
    n = expected["trials_per_participant"]
    require(len(full) == expected["participants"] * n and len(single) == n,
            "TEfits row count differs")
    require(not full.isna().to_numpy().any() and not single.isna().to_numpy().any(),
            "TEfits contains missing fields")
    for frame in (full, single):
        for col in columns[:-1]:
            values = frame[col].to_numpy(dtype=float)
            require(np.isfinite(values).all(), f"TEfits nonfinite {col}")
        require(np.isin(frame["resp"].to_numpy(), [0, 1]).all(), "TEfits response code")
        require(np.isin(frame["acc"].to_numpy(), [0, 1]).all(), "TEfits accuracy code")
        require((frame["ratio"].to_numpy() != 0).all(), "TEfits zero ratio needs codebook review")
        require((frame["sizeRat"].to_numpy() > 0).all(), "TEfits nonpositive size ratio")
        correct = frame["resp"].to_numpy() == (frame["ratio"].to_numpy() > 0)
        require(np.array_equal(correct, frame["acc"].to_numpy().astype(bool)),
                "TEfits response/ratio/accuracy coding mismatch")
    per_id = {}
    for identifier in expected["subIDs"]:
        group = full[full["subID"].astype(str) == identifier]
        require(np.array_equal(group["trialNum"].to_numpy(), np.arange(1, n + 1)),
                "TEfits trial order, duplicates or gaps differ")
        per_id[identifier] = {"rows": len(group), "trial_order_exact": True}
    first = full[full["subID"].astype(str) == "s1"]
    require(np.array_equal(first.to_numpy(), single.to_numpy()),
            "TEfits single participant file is not the exact s1 subset")
    return {"rows": len(full), "columns": columns, "participant_keys": identifiers,
            "per_id": per_id, "missing_fields": 0, "coding_mismatches": 0,
            "single_file_exact_s1_duplicate": True,
            "combine_single_as_extra_participant": False,
            "absent_fields": ["feedback", "session", "block", "RT", "reversal", "exact_counts"],
            "ratio_transform_and_color_key_mapping_confirmed": False}


def read_wide_csv(path, n_trials, responses=False):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = ["subject", *map(str, range(n_trials))]
    if responses:
        header.append("condition")
    require(rows and rows[0] == header, f"Unexpected CSV header: {Path(path).name}")
    require(all(len(row) == len(header) for row in rows[1:]), "Ragged CSV rows")
    identifiers = [row[0] for row in rows[1:]]
    require(all(identifiers) and len(set(identifiers)) == len(identifiers),
            "Missing or repeated subject keys within variant")
    return rows[1:]


def parse_values(tokens):
    values = []
    for token in tokens:
        if token == "":
            values.append(np.nan)
        else:
            value = float(token)
            require(math.isfinite(value), "Only empty cells may encode missing values")
            values.append(value)
    return np.array(values, dtype=float)


def duplicate_rows(rows, n_trials):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[1:n_trials + 1])].append(row[0])
    return [ids for ids in groups.values() if len(ids) > 1]


def audit_pybefit_group(directory, n_trials, expected):
    directory = Path(directory)
    tables = {name: read_wide_csv(directory / (name + ".csv"), n_trials,
                                  responses=name == "responses")
              for name in ("responses", "outcomes", "correct_responses")}
    responses = tables["responses"]
    require(len(responses) == expected["rows"], "pybefit row count differs")
    ids = [row[0] for row in responses]
    require(all([row[0] for row in values] == ids for values in tables.values()),
            "pybefit subject alignment differs")
    conditions = dict(Counter(row[-1] for row in responses))
    require(conditions == expected["conditions"], "pybefit condition counts differ")
    schedule = loadmat(directory / "states_and_rewards.mat", simplify_cells=True)
    for condition in conditions:
        require(condition in schedule and isinstance(schedule[condition], dict),
                "Missing MAT schedule condition")
        state = np.asarray(schedule[condition].get("S"))
        reward = np.asarray(schedule[condition].get("R"))
        require(state.shape == (n_trials,) and reward.shape == (n_trials, 2),
                "MAT schedule dimensions differ")
        require(np.isin(state, [1, 2]).all() and np.isin(reward, [-1, 1]).all(),
                "MAT state/reward code differs")
    mismatch = Counter({k: 0 for k in ("reward_vs_R", "C_vs_correctness", "hint_vs_state",
                                       "C_nan_for_nonbinary", "outcome_nan_for_missing")})
    codes = {name: set() for name in tables}
    for row, outcome, correctness in zip(responses, tables["outcomes"], tables["correct_responses"]):
        action, observed, correct = [parse_values(r[1:n_trials + 1])
                                     for r in (row, outcome, correctness)]
        require(np.isin(action[~np.isnan(action)], [0, 1, 2]).all(), "pybefit response code differs")
        require(np.isin(observed[~np.isnan(observed)], [0, 1, 2, 3]).all(), "pybefit outcome code differs")
        require(np.isin(correct[~np.isnan(correct)], [0, 1]).all(), "pybefit C code differs")
        binary, hint, missing = np.isin(action, [0, 1]), action == 2, np.isnan(action)
        state = np.asarray(schedule[row[-1]]["S"])
        reward = np.asarray(schedule[row[-1]]["R"])
        ix = np.flatnonzero(binary)
        mismatch["reward_vs_R"] += int(np.sum(observed[binary] != (reward[ix, action[binary].astype(int)] + 1) / 2))
        mismatch["C_vs_correctness"] += int(np.sum(correct[binary] != (action[binary] == state[binary] - 1)))
        mismatch["hint_vs_state"] += int(np.sum(observed[hint] != state[hint] + 1))
        mismatch["C_nan_for_nonbinary"] += int(np.sum(~np.isnan(correct[hint | missing])))
        mismatch["outcome_nan_for_missing"] += int(np.sum(~np.isnan(observed[missing])))
        for name, values in zip(tables, (action, observed, correct)):
            codes[name].update(int(v) for v in values if not np.isnan(v))
    require(not any(mismatch.values()), "pybefit reward/state/correctness semantics mismatch: " + str(dict(mismatch)))
    duplicates = {name: duplicate_rows(rows, n_trials) for name, rows in tables.items()}
    require(all(d == expected["duplicate_rows"] for d in duplicates.values()),
            "pybefit duplicate pattern differs")
    return {"rows": len(responses), "trials_per_row": n_trials,
            "conditions": conditions, "duplicate_rows": duplicates,
            "mismatch_counts": dict(mismatch), "code_sets": {k: sorted(v) for k, v in codes.items()},
            "C_is_correctness_indicator_not_correct_card_ID": True,
            "outcome_2_3_are_hint_state_codes_not_money": True,
            "unit_key": ["variant", "subject"]}


def audit(cache, manifest, rdata_deps=None, fetch=False):
    cache = external_cache(cache)
    records = prepare_cache(cache, manifest, fetch=fetch)
    if rdata_deps:
        sys.path.append(str(Path(rdata_deps).resolve()))
    try:
        import rdata
    except ImportError as exc:
        raise ValueError("Install requirements-human-candidate-audit.txt or provide --rdata-deps") from exc
    te, py = manifest["sources"]
    tbase = cache / TEFITS / "data"
    full = rdata.read_rda(tbase / "anstrain.RData")
    single = rdata.read_rda(tbase / "anstrain_s1.RData")
    require(list(full) == ["anstrain"] and list(single) == ["anstrain_s1"], "Unexpected RData objects")
    te_report = audit_tefits_frames(full["anstrain"], single["anstrain_s1"], te["expected"])
    pbase = cache / PYBEFIT / "examples/temporal_rev_learn/behavioral_data"
    groups = {group: audit_pybefit_group(pbase / group, py["expected"]["trials_per_row"], expected)
              for group, expected in py["expected"]["groups"].items()}
    equal_schedule = ((pbase / "main/states_and_rewards.mat").read_bytes() ==
                      (pbase / "pilot/states_and_rewards.mat").read_bytes())
    require(equal_schedule, "Main and pilot schedule bytes differ")
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("scipy", "rdata", "pandas", "xarray"):
        versions[name] = importlib.metadata.version(name)
    return {"schema_version": 1, "status": "pass", "scope": manifest["scope"],
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": versions, "source_files": records,
            "sources": {TEFITS: te_report, PYBEFIT: {"groups": groups, "schedule_files_byte_equal": True}},
            "performance_estimates_computed": False,
            "human_model_similarity_tested": False, "raw_data_redistributed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New JSON path; overwrite refused")
    parser.add_argument("--fetch", action="store_true", help="Fetch hash-pinned files into external cache only")
    parser.add_argument("--rdata-deps", type=Path, help="Append a separately installed RData dependency directory")
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Output already exists; choose a new path")
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    report = audit(args.cache, manifest, rdata_deps=args.rdata_deps, fetch=args.fetch)
    report["manifest_sha256"] = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    script = Path(__file__).resolve()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                         capture_output=True, text=True, check=False, timeout=15)
    report["code_identity"] = {
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "script": script.relative_to(REPOSITORY).as_posix(),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "manifest": DEFAULT_MANIFEST.relative_to(REPOSITORY).as_posix(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "files": len(report["source_files"]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
