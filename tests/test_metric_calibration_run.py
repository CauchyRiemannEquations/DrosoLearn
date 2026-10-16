"""Registered runs use short synthetic streams only, never scientific seeds/data."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from drosolearn import metric_calibration as calibration


REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def read_csv(path):
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def synthetic_config():
    cfg = read_json(REPO / "configs/metric_calibration_v1.json")
    cfg.update(scope="integration_test", seeds=[3, 4], trials=8)
    cfg["report_segments"] = [
        {"id": "phase1", "start_trial": 1, "end_trial": 4},
        {"id": "phase2", "start_trial": 5, "end_trial": 8},
    ]
    cfg["schedules"] = [
        {"id": "fixed", "state0_end_trial": 8},
        {"id": "reversal", "state0_end_trial": 4},
    ]
    cfg["rng"]["array_length"] = 8
    return cfg


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_repo"
    root.mkdir()
    cfg = synthetic_config()
    for rel in cfg["frozen_files"]:
        if rel == "configs/metric_calibration_v1.json":
            continue
        source = REPO / rel
        assert source.is_file(), f"Frozen implementation must exist before test execution: {rel}"
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    # Parent contents are deliberately synthetic and contain no human/model
    # behavioral rows. Registration must not scan previous raw output folders.
    for index, rel in enumerate(cfg["parent_files"]):
        save_json(root / rel, {"synthetic_parent": index, "status": "passed"})
    config_path = root / "configs/metric_calibration_v1.json"
    save_json(config_path, cfg)

    def synthetic_git(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert Path(kwargs["cwd"]) == root
        return "b" * 40 + "\n"

    monkeypatch.setattr(calibration.subprocess, "check_output", synthetic_git)
    original_simulate = calibration.simulate

    def guarded_simulate(cfg, seed, *args, **kwargs):
        assert cfg["scope"] == "integration_test"
        assert seed in (3, 4) and seed not in range(9001, 9011)
        assert cfg["trials"] == 8
        return original_simulate(cfg, seed, *args, **kwargs)

    monkeypatch.setattr(calibration, "simulate", guarded_simulate)
    return {"root": root, "cfg": cfg, "config_path": config_path,
            "registration": root / "registration.json", "out": root / "fresh_run"}


def register(fixture):
    return calibration.register(fixture["root"], fixture["config_path"], fixture["registration"])


def run(fixture):
    return calibration.run(fixture["root"], fixture["config_path"],
                           fixture["registration"], fixture["out"])


def test_registration_freezes_sources_and_parent_bytes_without_generating_trials(fixture, monkeypatch):
    monkeypatch.setattr(calibration, "simulate", lambda *a, **k: pytest.fail("Registration generated trials"))
    registration = register(fixture)
    assert registration == read_json(fixture["registration"])
    assert registration["source_git_commit"] == "b" * 40
    assert datetime.fromisoformat(registration["registered_at_utc"]).utcoffset() is not None
    assert not fixture["out"].exists()
    serialized = json.dumps(registration)
    for rel in fixture["cfg"]["frozen_files"] + fixture["cfg"]["parent_files"]:
        assert rel in serialized
        assert digest(fixture["root"] / rel) in serialized


def test_run_keeps_every_synthetic_stream_and_complete_original_adjacency_ledger(fixture):
    register(fixture)
    result = run(fixture)
    assert result["status"] == "completed"
    assert result["streams"] == 48
    assert result["trials"] == 384
    assert result["registration_sha256"] == digest(fixture["registration"])
    assert datetime.fromisoformat(result["started_at_utc"]) <= datetime.fromisoformat(result["completed_at_utc"])
    assert result["parameter_updates"] == 0
    assert result["human_or_species_comparison"] is False
    metrics = read_json(fixture["out"] / "metrics.json")
    assert len(metrics["streams"]) == 48
    assert len({(entry["seed"], entry["schedule"], entry["policy"], entry["environment"])
                for entry in metrics["streams"]}) == 48
    assert all(set(entry["strata"]) == {"all", "phase1", "phase2"} for entry in metrics["streams"])
    ledger = read_csv(fixture["out"] / "ledger.csv")
    assert len(ledger) == 48 * 7
    assert sum(int(entry["eligible"]) for entry in ledger) == 48 * 6
    assert all(int(entry["next_trial"]) == int(entry["trial"]) + 1 for entry in ledger)
    assert sum(entry["reasons"] == "phase_boundary" for entry in ledger) == 24
    assert sum(entry["reasons"] == "phase_boundary|state_boundary" for entry in ledger) == 24
    rows = read_csv(fixture["out"] / "trials.csv")
    assert len(rows) == 384
    assert set(entry["seed"] for entry in rows) == {"3", "4"}
    assert all(int(entry["trial"]) in range(1, 9) for entry in rows)
    hashes = read_json(fixture["out"] / "artifact_hashes.json")
    assert all(digest(fixture["out"] / rel) == sha for rel, sha in hashes.items())


@pytest.mark.parametrize("target", ["config", "source", "parent"])
def test_byte_tampering_after_registration_blocks_run_before_output(fixture, target):
    register(fixture)
    path = {
        "config": fixture["config_path"],
        "source": fixture["root"] / "src/drosolearn/metric_calibration.py",
        "parent": fixture["root"] / fixture["cfg"]["parent_files"][0],
    }[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


@pytest.mark.parametrize("target", ["study_id", "config_sha256", "source_git_commit"])
def test_registration_identity_tampering_is_rejected(fixture, target):
    register(fixture)
    value = read_json(fixture["registration"])
    value[target] = "tampered"
    save_json(fixture["registration"], value)
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_future_registration_timestamp_is_rejected_before_output(fixture):
    register(fixture)
    value = read_json(fixture["registration"])
    value["registered_at_utc"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    save_json(fixture["registration"], value)
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_registration_cannot_overwrite_previous_registration(fixture):
    register(fixture)
    original = fixture["registration"].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        register(fixture)
    assert fixture["registration"].read_bytes() == original


def test_existing_output_is_preserved_with_no_partial_extra_artifacts(fixture):
    register(fixture)
    fixture["out"].mkdir()
    sentinel = fixture["out"] / "previous_result.bin"
    sentinel.write_bytes(b"preserve previous scientific result")
    with pytest.raises((ValueError, FileExistsError)):
        run(fixture)
    assert sentinel.read_bytes() == b"preserve previous scientific result"
    assert list(fixture["out"].iterdir()) == [sentinel]


def test_repeating_run_with_same_registration_is_deterministic_except_execution_metadata(fixture):
    register(fixture)
    run(fixture)
    original_out = fixture["out"]
    fixture["out"] = fixture["root"] / "second_fresh_run"
    run(fixture)
    for name in ("trials.csv", "ledger.csv", "metrics.json"):
        assert (fixture["out"] / name).read_bytes() == (original_out / name).read_bytes()


def test_registered_inputs_are_unchanged_after_execution(fixture):
    expected = {rel: digest(fixture["root"] / rel)
                for rel in fixture["cfg"]["frozen_files"] + fixture["cfg"]["parent_files"]}
    register(fixture)
    registration_hash = digest(fixture["registration"])
    run(fixture)
    assert expected == {rel: digest(fixture["root"] / rel) for rel in expected}
    assert digest(fixture["registration"]) == registration_hash


def test_summary_keeps_ratio_of_counts_separate_from_mean_defined_seed_ratios(fixture):
    register(fixture)
    run(fixture)
    streams = read_json(fixture["out"] / "metrics.json")["streams"]
    summary = read_json(fixture["out"] / "summary.json")
    assert summary["parameter_updates"] == 0
    assert summary["monte_carlo_exact_match_required"] is False
    assert all(value["criterion_met"] for value in summary["analytic_claims"].values())
    different_quantities_found = False
    for case in summary["cases"]:
        matched = [stream["strata"][case["stratum"]] for stream in streams
                   if all(stream[key] == case[key] for key in ("schedule", "policy", "environment"))]
        for feedback in ("0", "1"):
            cells = [entry["empirical"]["by_feedback"][feedback] for entry in matched]
            denominator = sum(entry["denominator"] for entry in cells)
            numerator = sum(entry["stay"] for entry in cells)
            defined = [entry["stay_rate"] for entry in cells if entry["stay_rate"] is not None]
            actual = case["by_feedback"][feedback]
            assert actual["summed_denominator"] == denominator
            assert actual["summed_stay"] == numerator
            assert actual["defined_seeds"] == len(defined)
            expected_pooled = numerator / denominator if denominator else None
            expected_mean = sum(defined) / len(defined) if defined else None
            assert actual["ratio_of_summed_empirical_counts"] == pytest.approx(expected_pooled)
            assert actual["mean_defined_seed_ratio"] == pytest.approx(expected_mean)
            if expected_mean is not None and expected_pooled != expected_mean:
                different_quantities_found = True
    assert different_quantities_found, "Synthetic example must exercise distinct ratio aggregations"


def test_midrun_error_preserves_failed_metadata_and_partial_artifacts(fixture, monkeypatch):
    register(fixture)
    original = calibration.simulate
    calls = 0

    def fail_second_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic deliberate failure, never remove it")
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration, "simulate", fail_second_stream)
    result = run(fixture)
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert result["error_message"] == "synthetic deliberate failure, never remove it"
    assert len(read_csv(fixture["out"] / "trials.csv")) == 8
    assert read_json(fixture["out"] / "metadata.json") == result
    hashes = read_json(fixture["out"] / "artifact_hashes.json")
    assert all(digest(fixture["out"] / rel) == sha for rel, sha in hashes.items())


def test_cli_registration_execution_and_existing_output_guard(fixture, capsys):
    options = ["--root", str(fixture["root"]), "--config", str(fixture["config_path"])]
    assert calibration.main(["register", *options, "--out", str(fixture["registration"])]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "registered"
    assert calibration.main(["run", *options, "--registration", str(fixture["registration"]),
                             "--out", str(fixture["out"])]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    before = (fixture["out"] / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="Output already exists"):
        calibration.main(["run", *options, "--registration", str(fixture["registration"]),
                          "--out", str(fixture["out"])])
    assert (fixture["out"] / "metadata.json").read_bytes() == before


def test_cli_requires_registration_before_creating_run_directory(fixture):
    with pytest.raises(ValueError, match="Run requires registration"):
        calibration.main(["run", "--root", str(fixture["root"]),
                          "--config", str(fixture["config_path"]), "--out", str(fixture["out"])])
    assert not fixture["out"].exists()
