"""Independent structure replay using invented records only, never human source values."""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import subprocess

import numpy as np
import pytest
from scipy.io import savemat

from test_human_task_structure import audit


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_human_task_structure.py"
SPEC = importlib.util.spec_from_file_location("independent_task_structure", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def invented_arrays(irregular=False):
    return {"A": np.array([[1., 2., 3., np.nan, 2., 1.]]),
            "C": np.array([[0., 0., np.nan, np.nan, 1., 0.]] if irregular else
                          [[1., 0., np.nan, np.nan, 1., 0.]]),
            "S": np.array([[2], [1], [1], [1], [2], [2]] if irregular else
                          [[1], [1], [1], [2], [2], [2]]),
            "R": np.array([[1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 1]])}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def complete_fixture(tmp_path, monkeypatch, alteration=None):
    """Invent a 25-file parent audit and two schedules with six trial positions."""
    root, cache = tmp_path / "synthetic_repo", tmp_path / "external_cache"
    root.mkdir(); cache.mkdir()
    for name in VERIFIER.REQUIRED_FROZEN:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen source fixture\n", encoding="utf8")
    config = {"study_id": "human_task_structure_v1",
              "scope": "exploratory_source_task_structure_not_learning_performance", "trials_per_row": 6,
              "origin_manifest": "configs/human_osf_origin_manifest.json",
              "origin_result": "results/human_osf_origin_v1/comparison.json",
              "source_manifest": "data/human/task_structure_evidence_20261015/source-manifest.json",
              "frozen_files": sorted(VERIFIER.REQUIRED_FROZEN)}
    evidence = root / "data/human/task_structure_evidence_20261015/invented.txt"
    evidence.write_bytes(b"synthetic task semantics only\n")
    write_json(root / config["source_manifest"], {"files": [{"path": evidence.name,
        "bytes": evidence.stat().st_size, "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]})
    entries = []
    for slot in range(25):
        arrays = invented_arrays(slot >= 15)
        if alteration == "wrong_C" and slot == 0:
            arrays["C"][0, 0] = 0
        if alteration == "schedule_drift" and slot == 0:
            arrays["R"][0, 0] = -1
        if alteration == "invalid_schema" and slot == 0:
            arrays["A"][0, 0] = 4
        path = cache / ("invented_" + str(slot) + ".mat")
        savemat(path, arrays)
        entries.append({"role": "osf_raw", "path": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "derived_group": "main",
            "derived_row_index": slot if slot < 15 else slot + 12, "osf_file_id": "invented_" + str(slot)})
    write_json(root / config["origin_manifest"], {"source_files": entries,
                                                "groups": {"main": {"regular_rows": 27}}})
    write_json(root / config["origin_result"], {"status": "pass", "gate": {"passed": True},
        "total_values_compared": 75000, "total_mismatches": 0,
        "fixture_note": "Invented parent audit metadata; no human values or estimates."})
    config_path = root / "configs/human_task_structure_v1.json"
    write_json(config_path, config)
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
                    "commit", "--allow-empty", "-qm", "synthetic fixture"], cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "CONFIG", config_path)
    monkeypatch.setattr(VERIFIER, "ROOT", root)
    monkeypatch.setattr(VERIFIER, "CONFIG", config_path)
    registration_path, result_path = tmp_path / "registration.json", tmp_path / "result.json"
    registration = audit.register(cache)
    write_json(registration_path, registration)
    result = audit.run(cache, registration_path)
    write_json(result_path, result)
    return {"root": root, "cache": cache, "config_path": config_path,
            "registration_path": registration_path, "result_path": result_path,
            "registration": registration, "result": result}


def verify_fixture(fixture):
    return VERIFIER.verify(fixture["result_path"], fixture["registration_path"], fixture["cache"])


def test_known_boundaries_signature_and_unchanged_inputs():
    arrays = invented_arrays()
    before = {key: value.copy() for key, value in arrays.items()}
    result = VERIFIER.raw_structure(arrays, 6)
    assert result["structural_passed"] is True
    assert result["state_segment_starts_1based"] == [1, 4]
    assert result["state_segment_lengths"] == [3, 3]
    integers = [1, 1, 1, 2, 2, 2, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]
    assert result["schedule_sha256_little_endian_int64_S_then_R"] == hashlib.sha256(struct.pack("<18q", *integers)).hexdigest()
    assert all(np.array_equal(value, before[key], equal_nan=True) for key, value in arrays.items())


@pytest.mark.parametrize("trial,value", [(0, 0), (0, np.nan), (2, 0), (3, 0)])
def test_binary_query_and_missing_correctness_are_checked_separately(trial, value):
    arrays = invented_arrays()
    arrays["C"][0, trial] = value
    assert VERIFIER.raw_structure(arrays, 6)["structural_passed"] is False


def test_full_registration_producer_and_independent_all25_replay(tmp_path, monkeypatch):
    fixture = complete_fixture(tmp_path, monkeypatch)
    result = verify_fixture(fixture)
    assert result["status"] == "pass" and result["source_coded_descriptive_gate_passed"] is True
    assert result["raw_files_replayed"] == 25 and result["schedule_groups_verified"] == 2
    assert fixture["result"]["schedule_groups"]["regular"]["rows"] == 15
    assert fixture["result"]["schedule_groups"]["irregular"]["rows"] == 10


@pytest.mark.parametrize("mutation", ["shift_start", "length", "signature", "drop_row", "schedule_count", "similarity_claim"])
def test_rewritten_report_does_not_override_raw_replay(tmp_path, monkeypatch, mutation):
    fixture = complete_fixture(tmp_path, monkeypatch)
    result = fixture["result"]
    if mutation == "shift_start":
        result["rows"][0]["state_segment_starts_1based"] = [1, 3]
    elif mutation == "length":
        result["rows"][0]["state_segment_lengths"] = [2, 4]
    elif mutation == "signature":
        result["rows"][0]["schedule_sha256_little_endian_int64_S_then_R"] = "0" * 64
    elif mutation == "drop_row":
        result["rows"].pop()
    elif mutation == "schedule_count":
        result["schedule_groups"]["regular"]["unique_schedules"] = 2
    else:
        result["gate"]["direct_human_model_similarity_allowed"] = True
    write_json(fixture["result_path"], result)
    with pytest.raises(VERIFIER.VerificationError, match="Independent raw structure"):
        verify_fixture(fixture)


@pytest.mark.parametrize("alteration", ["wrong_C", "schedule_drift", "invalid_schema"])
def test_honest_failed_gate_is_preserved_with_passed_independent_verification(tmp_path, monkeypatch, alteration):
    fixture = complete_fixture(tmp_path, monkeypatch, alteration)
    assert fixture["result"]["status"] == "failed"
    verification = verify_fixture(fixture)
    assert verification["status"] == "pass" and verification["source_coded_descriptive_gate_passed"] is False


def test_raw_byte_tamper_is_rejected_before_replay(tmp_path, monkeypatch):
    fixture = complete_fixture(tmp_path, monkeypatch)
    with (fixture["cache"] / "invented_0.mat").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(VERIFIER.VerificationError, match="Raw source bytes"):
        verify_fixture(fixture)


def test_registration_rehash_does_not_hide_changed_input_identity(tmp_path, monkeypatch):
    fixture = complete_fixture(tmp_path, monkeypatch)
    registration = fixture["registration"]
    registration["inputs"][0]["sha256"] = "0" * 64
    write_json(fixture["registration_path"], registration)
    result = fixture["result"]
    result["registration_sha256"] = VERIFIER.sha(fixture["registration_path"])
    result["inputs"] = registration["inputs"]
    write_json(fixture["result_path"], result)
    with pytest.raises(VERIFIER.VerificationError, match="Registered source or raw input"):
        verify_fixture(fixture)


def test_cli_exclusive_verification_keeps_primary_artifacts_unchanged(tmp_path, monkeypatch, capsys):
    fixture = complete_fixture(tmp_path, monkeypatch)
    before = {name: fixture[name].read_bytes() for name in ("result_path", "registration_path", "config_path")}
    out = tmp_path / "verification.json"
    args = ["--result", str(fixture["result_path"]), "--registration", str(fixture["registration_path"]),
            "--cache", str(fixture["cache"]), "--out", str(out)]
    assert VERIFIER.main(args) == 0
    assert len(capsys.readouterr().out) < 300
    assert json.loads(out.read_text(encoding="utf8"))["status"] == "pass"
    assert all(fixture[name].read_bytes() == raw for name, raw in before.items())
    saved = out.read_bytes()
    with pytest.raises(VERIFIER.VerificationError, match="already exists"):
        VERIFIER.main(args)
    assert out.read_bytes() == saved
