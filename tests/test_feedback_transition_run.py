"""Synthetic registered-run integration; no real participant or model logs are read."""
from copy import deepcopy
import csv
from datetime import datetime
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat


SPEC = importlib.util.spec_from_file_location(
    "synthetic_feedback_transition_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_feedback_transition.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

FIELDS = ["seed", "condition", "history", "phase", "trial", "left_count", "right_count",
          "action", "reward", "p_right"]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def save_csv(path, records, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def csv_records(path):
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_fixture(tmp_path, monkeypatch):
    """Create 305 streams with 3 trials each, using the same production gates."""
    root, cache = tmp_path / "repo", tmp_path / "external_cache"
    root.mkdir()
    cache.mkdir()
    cfg = {
        "study_id": "feedback_transition_v1", "scope": "synthetic_integration_only",
        "human": {
            "origin_manifest": "parents/origin.json", "structure_result": "parents/human_structure.json",
            "structure_verification": "parents/human_verification.json", "trials": 3,
            "regime_boundary_after_trial": 2,
        },
        "model": {
            "root": "model", "parent_config": "parents/model_config.json",
            "verification": "parents/model_verification.json", "artifact_hashes": "parents/model_artifacts.json",
            "seeds": list(range(7001, 7011)),
            "conditions": ["linear_normalized", "fixed", "plastic", "rewired_fixed", "rewired_plastic",
                           "ann_fixed", "ann_plastic"],
            "histories": ["fresh_low", "high_to_low", "high_to_low_reset_head", "low_to_low"],
            "phase": "phase2", "trials": 3,
        },
        "parent_files": ["parents/origin.json", "parents/human_structure.json", "parents/human_verification.json",
                         "parents/model_verification.json", "parents/model_artifacts.json", "parents/model_config.json"],
        "frozen_files": ["configs/feedback_transition_v1.json", "source.py", "protocol.md"],
    }
    (root / "source.py").write_text("# synthetic source freeze marker\n", encoding="utf8")
    (root / "protocol.md").write_text("Synthetic protocol: retain every selected stream.\n", encoding="utf8")
    raw_entries = []
    for index in range(25):
        path = cache / f"raw/synthetic_{index:02d}.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        savemat(path, {
            "A": np.array([[1., 1., 2.]]), "S": np.array([[1], [1], [2]], dtype=np.int64),
            "R": np.array([[-1, 1], [1, -1], [-1, 1]], dtype=np.int64), "C": np.ones((1, 3)),
        })
        raw_entries.append({
            "role": "osf_raw", "path": path.relative_to(cache).as_posix(), "bytes": path.stat().st_size,
            "sha256": RUNNER.digest(path), "osf_file_id": f"synthetic{index:02d}",
            "derived_group": "main", "derived_row_index": index,
        })
    origin_path = root / cfg["human"]["origin_manifest"]
    save_json(origin_path, {"source_files": raw_entries})
    structure_path = root / cfg["human"]["structure_result"]
    structure_inputs = [{key: entry[key] for key in (
        "path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")} for entry in raw_entries]
    save_json(structure_path, {"status": "pass", "gate": {"source_coded_descriptive_metric_allowed": True},
                               "inputs": structure_inputs})
    save_json(root / cfg["human"]["structure_verification"],
              {"status": "pass", "result_sha256": RUNNER.digest(structure_path)})
    save_json(root / cfg["model"]["verification"], {"passed": True})
    save_json(root / cfg["model"]["parent_config"],
              {"conditions": cfg["model"]["conditions"], "histories": cfg["model"]["histories"]})
    artifacts, model_paths = {}, []
    for seed in cfg["model"]["seeds"]:
        records = []
        for condition in cfg["model"]["conditions"]:
            for history in cfg["model"]["histories"]:
                base = {"seed": str(seed), "condition": condition, "history": history}
                if history != "fresh_low":
                    records.append({**base, "phase": "phase1", "trial": "1", "left_count": "2",
                                    "right_count": "4", "action": "1", "reward": "1", "p_right": "0.5"})
                for trial, left, right, action, reward in [(1, 2, 4, 0, 1), (2, 2, 4, 1, 0), (3, 4, 2, 0, 0)]:
                    records.append({**base, "phase": "phase2", "trial": str(trial), "left_count": str(left),
                                    "right_count": str(right), "action": str(action), "reward": str(reward),
                                    "p_right": "0.9"})
        path = root / "model" / f"seed_{seed}" / "trials.csv"
        save_csv(path, records)
        artifacts[f"seed_{seed}/trials.csv"] = RUNNER.digest(path)
        model_paths.append(path)
    artifact_path = root / cfg["model"]["artifact_hashes"]
    save_json(artifact_path, artifacts)
    config_path = root / "configs/feedback_transition_v1.json"
    save_json(config_path, cfg)
    strict_config = RUNNER.config
    monkeypatch.setattr(RUNNER, "ROOT", root)
    monkeypatch.setattr(RUNNER, "CONFIG", config_path)
    # Only the production's fixed 1000/400-trial assertion is relaxed. All actual
    # byte, path, semantic-parent, all25 and 305-stream checks run unchanged.
    monkeypatch.setattr(RUNNER, "config", lambda: read_json(config_path))

    def synthetic_git(command, *, cwd, text):
        assert command == ["git", "rev-parse", "HEAD"]
        assert cwd == root and text is True
        return "a" * 40 + "\n"

    monkeypatch.setattr(RUNNER.subprocess, "check_output", synthetic_git)
    return {
        "runner": RUNNER, "root": root, "cache": cache, "cfg": cfg, "config_path": config_path,
        "strict_config": strict_config, "registration_path": root / "registration.json", "out": root / "new_run",
        "origin_path": origin_path, "artifact_path": artifact_path, "model_paths": model_paths,
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return make_fixture(tmp_path, monkeypatch)


def register_fixture(fixture):
    registration = RUNNER.register(fixture["cache"])
    RUNNER.write_new(fixture["registration_path"], registration)
    return registration


def refreeze_changed_model(fixture, records, *, fields=FIELDS):
    """Change synthetic input before registration to exercise parser errors."""
    path = fixture["model_paths"][0]
    save_csv(path, records, fields=fields)
    artifacts = read_json(fixture["artifact_path"])
    artifacts[path.relative_to(fixture["root"] / "model").as_posix()] = RUNNER.digest(path)
    save_json(fixture["artifact_path"], artifacts)


def test_synthetic_register_run_all305_streams_and_complete_ledger(fixture):
    registration = register_fixture(fixture)
    assert registration["source_git_commit"] == "a" * 40
    assert datetime.fromisoformat(registration["registered_at_utc"]).utcoffset() is not None
    assert len([entry for entry in registration["inputs"] if entry["role"] == "human_raw"]) == 25
    assert len([entry for entry in registration["inputs"] if entry["role"] == "model_actual"]) == 10
    result = RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert result["status"] == "completed"
    assert result["streams"] == 305 and result["trials"] == 915
    assert result["candidates"] == 610 and result["eligible"] == 585 and result["excluded"] == 25
    assert result["registration_sha256"] == RUNNER.digest(fixture["registration_path"])
    assert datetime.fromisoformat(result["started_at_utc"]) <= datetime.fromisoformat(result["completed_at_utc"])
    assert result["group_or_species_contrast_computed"] is False
    assert result["new_model_learning_executed"] is False
    assert result["level4_or_level5_supported"] is False
    metrics = read_json(fixture["out"] / "metrics.json")
    assert metrics["pooled_species_or_group_estimates"] is False
    assert len({entry["stream_id"] for entry in metrics["streams"]}) == 305
    human = [entry for entry in metrics["streams"] if entry["kind"] == "human"]
    model = [entry for entry in metrics["streams"] if entry["kind"] == "model"]
    assert len(human) == 25 and len(model) == 280
    assert all(entry["reason_counts"] == {"phase_boundary": 1, "state_boundary": 1} for entry in human)
    assert all(entry["trials"] == 3 and entry["eligible"] == 2 for entry in model)
    assert all(entry["by_feedback"]["0"]["stay"] == 1 and entry["by_feedback"]["1"]["switch"] == 1
               for entry in model)
    ledger = csv_records(fixture["out"] / "ledger.csv")
    assert len(ledger) == 610
    assert all(int(entry["next_trial"]) == int(entry["trial"]) + 1 for entry in ledger)
    assert len([entry for entry in ledger if entry["reasons"] == "phase_boundary|state_boundary"]) == 25
    first_model_pairs = [entry for entry in ledger if entry["stream_id"] == "model/7001/linear_normalized/fresh_low/phase2"]
    assert [(entry["source_position"], entry["next_source_position"]) for entry in first_model_pairs] == [
        ("2", "3"), ("3", "4")]
    hashes = read_json(fixture["out"] / "artifact_hashes.json")
    assert set(hashes) == {"metadata.json", "metrics.json", "ledger.csv"}
    assert all(RUNNER.digest(fixture["out"] / name) == sha for name, sha in hashes.items())
    assert RUNNER.input_inventory(fixture["cfg"], fixture["cache"]) == registration["inputs"]


@pytest.mark.parametrize("target", ["source", "config"])
def test_frozen_source_or_config_change_is_blocked_before_output(fixture, target):
    register_fixture(fixture)
    path = fixture["root"] / "source.py" if target == "source" else fixture["config_path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen analysis source/config"):
        RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert not fixture["out"].exists()


def test_changed_config_scope_is_rejected_even_if_config_loader_accepts_it(fixture):
    register_fixture(fixture)
    cfg = read_json(fixture["config_path"])
    cfg["model"]["histories"] = cfg["model"]["histories"][:-1]
    save_json(fixture["config_path"], cfg)
    with pytest.raises(ValueError, match="Frozen analysis source/config"):
        RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])


@pytest.mark.parametrize("target", ["human", "model", "parent"])
def test_registered_input_byte_tampering_is_rejected_before_output(fixture, target):
    register_fixture(fixture)
    if target == "human":
        path = fixture["cache"] / read_json(fixture["origin_path"])["source_files"][0]["path"]
    elif target == "model":
        path = fixture["model_paths"][0]
    else:
        path = fixture["root"] / fixture["cfg"]["model"]["parent_config"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen input bytes|artifact freeze|Registered analysis inputs"):
        RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert not fixture["out"].exists()


@pytest.mark.parametrize("gate", ["human_status", "human_permission", "human_verifier", "model_verifier"])
def test_parent_gate_blocks_registration(fixture, gate):
    cfg, root = fixture["cfg"], fixture["root"]
    if gate in ("human_status", "human_permission"):
        path = root / cfg["human"]["structure_result"]
        value = read_json(path)
        if gate == "human_status":
            value["status"] = "fail"
        else:
            value["gate"]["source_coded_descriptive_metric_allowed"] = False
    elif gate == "human_verifier":
        path = root / cfg["human"]["structure_verification"]
        value = {"status": "fail", "result_sha256": "0" * 64}
    else:
        path = root / cfg["model"]["verification"]
        value = {"passed": False}
    save_json(path, value)
    with pytest.raises(ValueError, match="gate|verification"):
        RUNNER.register(fixture["cache"])


def test_human_structure_verification_must_bind_exact_parent_result_bytes(fixture):
    path = fixture["root"] / fixture["cfg"]["human"]["structure_result"]
    value = read_json(path)
    value["synthetic_additional_field"] = True
    save_json(path, value)
    with pytest.raises(ValueError, match="Independent human structure verification"):
        RUNNER.register(fixture["cache"])


@pytest.mark.parametrize("key", ["path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id"])
def test_parent_gate_inputs_must_match_each_current_raw_identity_field(fixture, key):
    path = fixture["root"] / fixture["cfg"]["human"]["structure_result"]
    value = read_json(path)
    first = value["inputs"][0]
    first[key] = {
        "path": "raw/unrelated_synthetic.mat", "bytes": first["bytes"] + 1,
        "sha256": "0" * 64, "derived_group": "pilot", "derived_row_index": 99,
        "osf_file_id": "unrelated_synthetic",
    }[key]
    save_json(path, value)
    # The independent parent report still authenticates this synthetic result;
    # identity linking, rather than a stale parent-result hash, must reject it.
    check_path = fixture["root"] / fixture["cfg"]["human"]["structure_verification"]
    save_json(check_path, {"status": "pass", "result_sha256": RUNNER.digest(path)})
    with pytest.raises(ValueError, match="exact sources covered"):
        RUNNER.register(fixture["cache"])


@pytest.mark.parametrize("defect", ["reordered", "missing", "duplicated"])
def test_parent_gate_must_cover_whole_raw_inventory_in_frozen_order(fixture, defect):
    path = fixture["root"] / fixture["cfg"]["human"]["structure_result"]
    value = read_json(path)
    if defect == "reordered":
        value["inputs"].reverse()
    elif defect == "missing":
        value["inputs"].pop()
    else:
        value["inputs"][-1] = value["inputs"][0].copy()
    save_json(path, value)
    check_path = fixture["root"] / fixture["cfg"]["human"]["structure_verification"]
    save_json(check_path, {"status": "pass", "result_sha256": RUNNER.digest(path)})
    with pytest.raises(ValueError, match="exact sources covered"):
        RUNNER.register(fixture["cache"])


def test_rehashed_replacement_raw_cannot_reuse_old_parent_gate(fixture):
    origin = read_json(fixture["origin_path"])
    entry = origin["source_files"][0]
    path = fixture["cache"] / entry["path"]
    # Same source ID/shape/action/state/C; change only a potential actual result.
    # Updating the new manifest is insufficient because A verified other bytes.
    savemat(path, {"A": np.array([[1., 1., 2.]]), "S": np.array([[1], [1], [2]]),
                  "R": np.array([[1, 1], [1, -1], [-1, 1]]), "C": np.ones((1, 3))})
    entry.update(bytes=path.stat().st_size, sha256=RUNNER.digest(path))
    save_json(fixture["origin_path"], origin)
    with pytest.raises(ValueError, match="exact sources covered"):
        RUNNER.register(fixture["cache"])


def test_parent_gate_closed_after_registration_still_blocks_execution(fixture):
    register_fixture(fixture)
    save_json(fixture["root"] / fixture["cfg"]["model"]["verification"], {"passed": False})
    with pytest.raises(ValueError, match="gate is closed"):
        RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert not fixture["out"].exists()


@pytest.mark.parametrize("defect", ["only24", "duplicate_row", "duplicate_path"])
def test_all25_unique_raw_sources_must_remain(fixture, defect):
    value = read_json(fixture["origin_path"])
    entries = value["source_files"]
    if defect == "only24":
        entries.pop()
    elif defect == "duplicate_row":
        entries[-1]["derived_row_index"] = entries[0]["derived_row_index"]
    else:
        entries[-1]["path"] = entries[0]["path"]
    save_json(fixture["origin_path"], value)
    with pytest.raises(ValueError, match="all25|Repeated input|exact sources covered"):
        RUNNER.register(fixture["cache"])


def test_human_raw_cache_inside_repo_is_rejected(fixture):
    with pytest.raises(ValueError, match="external"):
        RUNNER.register(fixture["root"] / "raw")


@pytest.mark.parametrize("relative", ["../outside", "sub/../../outside", "sub\\outside"])
def test_unsafe_source_path_is_rejected(fixture, relative):
    with pytest.raises(ValueError, match="Unsafe input path"):
        RUNNER.safe_path(fixture["root"], relative)


def test_repeated_frozen_source_path_is_rejected(fixture):
    cfg = deepcopy(fixture["cfg"])
    cfg["frozen_files"].append(cfg["frozen_files"][0])
    with pytest.raises(ValueError, match="Repeated source"):
        RUNNER.source_freeze(cfg)


@pytest.mark.parametrize("defect,message", [
    ("missing_stream", "Missing model stream"),
    ("missing_trial", "Incomplete or reordered"),
    ("reordered_trial", "Incomplete or reordered"),
    ("duplicate_key", "Duplicate original model key"),
    ("wrong_seed", "Unexpected actual model stream"),
    ("unknown_condition", "Unexpected actual model stream"),
    ("unknown_phase", "Unexpected actual model stream"),
])
def test_invalid_registered_model_structure_is_retained_as_failed_result(fixture, defect, message):
    records = csv_records(fixture["model_paths"][0])
    selected = [index for index, record in enumerate(records)
                if record["phase"] == "phase2" and record["condition"] == "linear_normalized"
                and record["history"] == "fresh_low"]
    if defect == "missing_stream":
        records = [record for index, record in enumerate(records) if index not in selected]
    elif defect == "missing_trial":
        records.pop(selected[1])
    elif defect == "reordered_trial":
        records[selected[0]], records[selected[1]] = records[selected[1]], records[selected[0]]
    elif defect == "duplicate_key":
        records.insert(selected[0], records[selected[0]].copy())
    else:
        key, value = {"wrong_seed": ("seed", "9999"), "unknown_condition": ("condition", "unknown"),
                      "unknown_phase": ("phase", "unknown")}[defect]
        records[selected[0]][key] = value
    refreeze_changed_model(fixture, records)
    register_fixture(fixture)
    result = RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert result["status"] == "failed" and result["failure_type"] == "ValueError"
    assert message in result["failure"]
    assert read_json(fixture["out"] / "metadata.json") == result
    assert not (fixture["out"] / "metrics.json").exists()
    hashes = read_json(fixture["out"] / "artifact_hashes.json")
    assert all(RUNNER.digest(fixture["out"] / name) == sha for name, sha in hashes.items())


@pytest.mark.parametrize("defect", ["duplicate_header", "missing_header", "extra_cell", "missing_cell"])
def test_bad_model_csv_header_or_ragged_rows_fail_without_silent_repair(fixture, defect):
    path = fixture["model_paths"][0]
    records = csv_records(path)
    if defect == "duplicate_header":
        refreeze_changed_model(fixture, records, fields=FIELDS + ["trial"])
    elif defect == "missing_header":
        for record in records:
            record.pop("reward")
        refreeze_changed_model(fixture, records, fields=[field for field in FIELDS if field != "reward"])
    else:
        lines = path.read_text(encoding="utf8").splitlines()
        lines[1] = lines[1] + ",extra" if defect == "extra_cell" else ",".join(lines[1].split(",")[:-1])
        path.write_text("\n".join(lines) + "\n", encoding="utf8")
        artifacts = read_json(fixture["artifact_path"])
        artifacts[path.relative_to(fixture["root"] / "model").as_posix()] = RUNNER.digest(path)
        save_json(fixture["artifact_path"], artifacts)
    register_fixture(fixture)
    result = RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert result["status"] == "failed"
    assert "header" in result["failure"] or "Ragged" in result["failure"]


def test_existing_run_directory_and_registration_are_not_overwritten(fixture):
    register_fixture(fixture)
    registration_bytes = fixture["registration_path"].read_bytes()
    with pytest.raises(FileExistsError):
        RUNNER.write_new(fixture["registration_path"], {"overwrite": True})
    assert fixture["registration_path"].read_bytes() == registration_bytes
    fixture["out"].mkdir()
    sentinel = fixture["out"] / "sentinel.txt"
    sentinel.write_bytes(b"prior result remains")
    with pytest.raises(FileExistsError):
        RUNNER.run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert sentinel.read_bytes() == b"prior result remains"
    assert set(fixture["out"].iterdir()) == {sentinel}


def test_strict_production_config_still_requires_fixed_trial_scope(fixture):
    with pytest.raises(ValueError, match="Fixed human study identity"):
        fixture["strict_config"]()
    cfg = deepcopy(fixture["cfg"])
    cfg["human"].update(trials=1000, regime_boundary_after_trial=800)
    save_json(fixture["config_path"], cfg)
    with pytest.raises(ValueError, match="Fixed model scope"):
        fixture["strict_config"]()
    cfg["model"]["trials"] = 400
    save_json(fixture["config_path"], cfg)
    assert fixture["strict_config"]() == cfg
    cfg["model"]["histories"] = cfg["model"]["histories"][:-1]
    save_json(fixture["config_path"], cfg)
    with pytest.raises(ValueError, match="All prior conditions/histories"):
        fixture["strict_config"]()


def test_cli_registration_run_and_existing_output_guard_use_synthetic_inputs_only(fixture, capsys):
    cache = str(fixture["cache"])
    registration = str(fixture["registration_path"])
    out = str(fixture["out"])
    assert RUNNER.main(["register", "--human-cache", cache, "--out", registration]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "registered"
    assert RUNNER.main(["run", "--human-cache", cache, "--registration", registration, "--out", out]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    original = (fixture["out"] / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="Output already exists"):
        RUNNER.main(["run", "--human-cache", cache, "--registration", registration, "--out", out])
    assert (fixture["out"] / "metadata.json").read_bytes() == original


def test_cli_run_requires_registration_before_creating_output(fixture):
    with pytest.raises(ValueError, match="Run requires registration"):
        RUNNER.main(["run", "--human-cache", str(fixture["cache"]), "--out", str(fixture["out"])])
    assert not fixture["out"].exists()
