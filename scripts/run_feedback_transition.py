"""Registered descriptive persistence counts on fixed human/model source streams."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drosolearn.feedback_transition import human_rows, measure_stream, model_row

CONFIG = ROOT / "configs/feedback_transition_v1.json"
LEDGER_FIELDS = ["stream_id", "source_position", "next_source_position", "trial", "next_trial",
                 "eligible", "reasons", "feedback", "stay"]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(base, relative):
    require(type(relative) is str and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Unsafe input path")
    base = Path(base).resolve()
    result = (base / relative).resolve()
    require(result.is_relative_to(base), "Input path escapes base")
    return result


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def config():
    cfg = json.loads(CONFIG.read_text(encoding="utf8"))
    require(cfg["study_id"] == "feedback_transition_v1" and cfg["human"]["trials"] == 1000
            and cfg["human"]["regime_boundary_after_trial"] == 800,
            "Fixed human study identity required")
    model = cfg["model"]
    require(model["seeds"] == list(range(7001, 7011)) and model["phase"] == "phase2"
            and model["trials"] == 400, "Fixed model scope required")
    prior = json.loads(safe_path(ROOT, cfg["model"]["parent_config"]).read_text(encoding="utf8"))
    require(model["conditions"] == prior["conditions"] and model["histories"] == prior["histories"],
            "All prior conditions/histories must be retained")
    return cfg


def source_freeze(cfg):
    require(len(cfg["frozen_files"]) == len(set(cfg["frozen_files"])), "Repeated source path")
    return {p: digest(safe_path(ROOT, p)) for p in sorted(cfg["frozen_files"])}


def verify_parent_gates(cfg):
    task = json.loads(safe_path(ROOT, cfg["human"]["structure_result"]).read_text(encoding="utf8"))
    require(task["status"] == "pass" and task["gate"]["source_coded_descriptive_metric_allowed"] is True,
            "Human source-coded semantic/structure gate is closed")
    task_verify = json.loads(safe_path(ROOT, cfg["human"]["structure_verification"]).read_text(encoding="utf8"))
    require(task_verify.get("status") == "pass" and task_verify["result_sha256"] == digest(
        safe_path(ROOT, cfg["human"]["structure_result"])), "Independent human structure verification required")
    model = json.loads(safe_path(ROOT, cfg["model"]["verification"]).read_text(encoding="utf8"))
    require(model["passed"] is True, "Prior actual model replay gate is closed")
    return task


def input_inventory(cfg, human_cache):
    human_cache = Path(human_cache).resolve()
    require(not human_cache.is_relative_to(ROOT), "Human raw source cache must remain external")
    task = verify_parent_gates(cfg)
    origin = json.loads(safe_path(ROOT, cfg["human"]["origin_manifest"]).read_text(encoding="utf8"))
    raw = [entry for entry in origin["source_files"] if entry["role"] == "osf_raw"]
    require(len(raw) == 25 and len({e["derived_row_index"] for e in raw}) == 25, "Fixed all25 scope required")
    linked = [{key: entry[key] for key in ("path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")}
              for entry in raw]
    require(all(entry["derived_group"] == "main" for entry in raw) and linked == task["inputs"],
            "Human raw bytes are not the exact sources covered by the structure gate")
    entries = [{"base": "human_cache", "role": "human_raw", **{k: e[k] for k in ("path", "bytes", "sha256", "osf_file_id")},
                "row_index": e["derived_row_index"]} for e in raw]
    artifacts = json.loads(safe_path(ROOT, cfg["model"]["artifact_hashes"]).read_text(encoding="utf8"))
    for seed in cfg["model"]["seeds"]:
        rel = f"seed_{seed}/trials.csv"
        path = cfg["model"]["root"] + "/" + rel
        target = safe_path(ROOT, path)
        require(digest(target) == artifacts[rel], "Actual model log differs from prior artifact freeze")
        entries.append({"base": "repo", "role": "model_actual", "path": path,
                        "bytes": target.stat().st_size, "sha256": artifacts[rel], "seed": seed})
    for path in cfg["parent_files"]:
        target = safe_path(ROOT, path)
        entries.append({"base": "repo", "role": "parent", "path": path,
                        "bytes": target.stat().st_size, "sha256": digest(target)})
    require(len({(e["base"], e["path"]) for e in entries}) == len(entries), "Repeated input identity")
    for e in entries:
        target = safe_path(human_cache if e["base"] == "human_cache" else ROOT, e["path"])
        require(target.stat().st_size == e["bytes"] and digest(target) == e["sha256"], "Frozen input bytes differ")
    return entries


def register(human_cache):
    cfg = config()
    checked = input_inventory(cfg, human_cache)
    return {"study_id": cfg["study_id"], "registration_type": "descriptive_feedback_transition",
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": digest(CONFIG), "config": cfg, "frozen_files": source_freeze(cfg), "inputs": checked,
            "notice": "Exploratory fixed descriptive analysis; private Git freeze, not blinded confirmatory preregistration"}


def streams_from_inputs(cfg, entries, human_cache):
    for entry in entries:
        if entry["role"] == "human_raw":
            raw = loadmat(safe_path(human_cache, entry["path"]), variable_names=["A", "C", "R", "S"])
            yield human_rows(raw, stream_id=f"human/main/{entry['row_index']}", trials=cfg["human"]["trials"],
                             regime_boundary_after_trial=cfg["human"]["regime_boundary_after_trial"])
        elif entry["role"] == "model_actual":
            grouped = defaultdict(list)
            seen_keys = set()
            with safe_path(ROOT, entry["path"]).open(encoding="utf8", newline="") as handle:
                reader = csv.DictReader(handle)
                require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
                        and {"seed", "condition", "history", "phase", "trial", "left_count", "right_count", "action", "reward", "p_right"} <= set(reader.fieldnames),
                        "Actual model header differs")
                for source_position, record in enumerate(reader, 2):
                    require(None not in record and all(value is not None for value in record.values()), "Ragged actual model log")
                    require(record["seed"] == str(entry["seed"]) and record["condition"] in cfg["model"]["conditions"]
                            and record["history"] in cfg["model"]["histories"] and record["phase"] in ("phase1", "phase2"),
                            "Unexpected actual model stream")
                    key = tuple(record[k] for k in ("seed", "condition", "history", "phase", "trial"))
                    require(key not in seen_keys, "Duplicate original model key")
                    seen_keys.add(key)
                    if record["phase"] == "phase2":
                        row = model_row(record, source_position=source_position)
                        grouped[row["stream_id"]].append(row)
            require(len(grouped) == len(cfg["model"]["conditions"]) * len(cfg["model"]["histories"]), "Missing model stream")
            for condition in cfg["model"]["conditions"]:
                for history in cfg["model"]["histories"]:
                    stream = f"model/{entry['seed']}/{condition}/{history}/phase2"
                    rows = grouped[stream]
                    require([r["trial"] for r in rows] == list(range(1, cfg["model"]["trials"] + 1)),
                            "Incomplete or reordered actual model stream")
                    yield rows


def run(human_cache, registration, out):
    cfg = config()
    reg = json.loads(Path(registration).read_text(encoding="utf8"))
    require(reg["study_id"] == cfg["study_id"] and reg["config"] == cfg
            and reg["config_sha256"] == digest(CONFIG) and reg["frozen_files"] == source_freeze(cfg),
            "Frozen analysis source/config differs")
    entries = input_inventory(cfg, human_cache)
    require(entries == reg["inputs"], "Registered analysis inputs differ")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"study_id": cfg["study_id"], "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "registration_sha256": digest(registration), "source_git_commit": reg["source_git_commit"],
                "config_sha256": digest(CONFIG), "frozen_files": reg["frozen_files"], "inputs": entries,
                "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
                "scope": cfg["scope"], "group_or_species_contrast_computed": False,
                "new_model_learning_executed": False, "level4_or_level5_supported": False}
    write_new(out / "metadata.json", metadata)
    try:
        summaries = []
        with (out / "ledger.csv").open("x", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            for rows in streams_from_inputs(cfg, entries, human_cache):
                summary, ledger = measure_stream(rows)
                summaries.append(summary)
                writer.writerows(ledger)
        require(len(summaries) == 305 and len({s["stream_id"] for s in summaries}) == 305,
                "All25 human and280 model streams must remain")
        write_new(out / "metrics.json", {"study_id": cfg["study_id"], "streams": summaries,
                                         "pooled_species_or_group_estimates": False})
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        streams=len(summaries), trials=sum(s["trials"] for s in summaries),
                        candidates=sum(s["candidates"] for s in summaries),
                        eligible=sum(s["eligible"] for s in summaries), excluded=sum(s["excluded"] for s in summaries))
    except Exception as exc:
        metadata.update(status="failed", failure_type=type(exc).__name__, failure=str(exc),
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
    # metadata.json is a run-state journal within this new output directory only.
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    write_new(out / "artifact_hashes.json", {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "run"])
    parser.add_argument("--human-cache", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Output already exists")
    require(args.mode != "run" or args.registration is not None, "Run requires registration")
    if args.mode == "register":
        result = register(args.human_cache)
        write_new(args.out, result)
    else:
        result = run(args.human_cache, args.registration, args.out)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return int(result.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
