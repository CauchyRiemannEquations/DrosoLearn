"""Copy one completed tiny parent; static tests perform no new learning."""
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import shutil
import stat
import zipfile

import numpy as np
import pytest

from drosolearn import state_decomposition as study
from test_state_decomposition import synthetic_config


REPO = Path(__file__).resolve().parents[1]
FIXTURE_ARCHIVE = REPO / "data/engineering_fixtures/learning_ablation_tiny_v1.zip"
FIXTURE_MANIFEST = REPO / "data/engineering_fixtures/learning_ablation_tiny_v1_manifest.json"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_csv(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_existing_fixture(destination):
    """Safely copy the frozen142 original files; never regenerate a learner."""
    manifest = read_json(FIXTURE_MANIFEST)
    assert manifest["fixture_id"] == "learning_ablation_tiny_v1"
    assert manifest["archive_path"] == "data/engineering_fixtures/learning_ablation_tiny_v1.zip"
    assert FIXTURE_ARCHIVE.stat().st_size == manifest["archive_bytes"]
    assert study.digest(FIXTURE_ARCHIVE) == manifest["archive_sha256"]
    assert manifest["file_count"] == len(manifest["files"]) == 142
    assert manifest["original_primary_files"] == 138
    assert all(manifest[key] == 0 for key in ("new_learning_updates", "new_learner_constructions", "new_dataset_generations", "new_rng_draws"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    assert len(expected) == 142
    destination.mkdir()
    with zipfile.ZipFile(FIXTURE_ARCHIVE) as archive:
        assert len(archive.infolist()) == len(expected) and set(archive.namelist()) == set(expected)
        for member in archive.infolist():
            assert not member.is_dir() and member.date_time == (2000, 1, 1, 0, 0, 0)
            assert stat.S_IFMT(member.external_attr >> 16) != stat.S_IFLNK
            target = study.safe(destination, member.filename)
            entry = expected[member.filename]
            assert member.file_size == entry["bytes"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(archive.read(member))
            assert target.stat().st_size == entry["bytes"] and study.digest(target) == entry["sha256"]
    return destination


def make_fixture(tmp_path, monkeypatch):
    # This is an already completed engineering run. Missing it is an explicit
    # reproducibility limitation; the test must never silently generate training.
    existing_parent = extract_existing_fixture(tmp_path / "existing_completed_parent")
    assert (existing_parent / "fresh_run/verification.json").is_file()
    root = tmp_path / "static_repo"
    root.mkdir()
    cfg = synthetic_config()
    for name in cfg["frozen_files"]:
        if name == "configs/state_decomposition_v1.json":
            continue
        source = REPO / name
        assert source.is_file(), f"Wait for all new static source files before integration: {name}"
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    parent = root / cfg["parent_root"]
    shutil.copytree(existing_parent / "fresh_run", parent)
    shutil.copyfile(existing_parent / "registration.json", root / "results/registration_learning_ablation_v1.json")
    shutil.copyfile(existing_parent / "configs/learning_ablation_v1.json", root / cfg["parent_config"])
    checked, metadata = read_json(parent / "verification.json"), read_json(parent / "metadata.json")
    assert checked["status"] == "passed"
    assert metadata["config"]["seeds"] == [17, 18] and metadata["config"]["trials_per_phase"] == 8
    # Four new engineering-only documentation/provenance fixtures link the
    # unchanged completed tiny parent. They are not scientific validation claims.
    corrected = root / "docs/learning-ablation-findings-v2.md"
    corrected.write_text("Synthetic documentation link for an existing tiny engineering run.\n", encoding="utf8")
    correction = {"status": "documentation_corrected", "corrected_document": "docs/learning-ablation-findings-v2.md",
        "corrected_sha256": study.digest(corrected), "summary_sha256": checked["summary_sha256"],
        "verification_sha256": study.digest(parent / "verification.json"), "scientific_values_or_criteria_changed": False,
        "additional_learning_updates": 0, "synthetic_engineering_evidence": True}
    save_json(root / "results/learning_ablation_documentation_correction_v1.json", correction)
    save_json(root / "results/learning_ablation_final_validation_v1.json", {"status": "passed", "study_id": "learning_ablation_v1",
        "all_sealed_scientific_and_original_document_bytes_unchanged": True, "documentation_correction": correction,
        "synthetic_engineering_evidence": True})
    save_json(root / "results/learning_ablation_input_freeze_v1.json", {"study_id": "learning_ablation_v1",
        "source_commit": metadata["source_git_commit"], "registration_sha256": checked["registration_sha256"],
        "config_sha256": checked["config_sha256"], "learning_updates": 0, "learner_constructions": 0,
        "synthetic_engineering_evidence": True})
    cfg["parent_gate_identity"] = {"published_commit": "c" * 40,
        "file_sha256": {name: study.digest(root / name) for name in cfg["parent_files"]}}
    config = root / "configs/state_decomposition_v1.json"
    save_json(config, cfg)
    original_git = study.subprocess.check_output
    def synthetic_git(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "e" * 40 + "\n"
        return original_git(command, **kwargs)
    monkeypatch.setattr(study.subprocess, "check_output", synthetic_git)
    from drosolearn.reward_linear import LinearLearner, NormalizedLearner
    from drosolearn.model import Learner
    from drosolearn import data, learning_ablation
    def forbidden(*args, **kwargs):
        pytest.fail("Static analysis invoked a learner, training, data generation or RNG")
    for cls in (LinearLearner, NormalizedLearner, Learner):
        monkeypatch.setattr(cls, "__init__", forbidden)
        monkeypatch.setattr(cls, "update", forbidden)
    monkeypatch.setattr(data, "make_dataset", forbidden)
    monkeypatch.setattr(learning_ablation, "selected_update", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    return {"root": root, "cfg": cfg, "config": config, "registration": root / "registration.json", "out": root / "static_result",
            "copied_parent_source": str(FIXTURE_ARCHIVE), "copied_parent_archive_sha256": study.digest(FIXTURE_ARCHIVE)}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return make_fixture(tmp_path, monkeypatch)


def register(fixture):
    return study.register(fixture["root"], fixture["config"], fixture["registration"])


def run(fixture):
    return study.run(fixture["root"], fixture["config"], fixture["registration"], fixture["out"])


@pytest.fixture
def completed(fixture):
    register(fixture)
    assert run(fixture)["status"] == "completed"
    return fixture


def test_bundled_fixture_hash_rejects_modified_archive_before_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "modified.zip"
    archive.write_bytes(FIXTURE_ARCHIVE.read_bytes() + b"unregistered bytes")
    monkeypatch.setitem(globals(), "FIXTURE_ARCHIVE", archive)
    with pytest.raises(AssertionError):
        extract_existing_fixture(tmp_path / "no_restore")
    assert not (tmp_path / "no_restore").exists()


@pytest.mark.parametrize("mode", ["traversal", "symlink"])
def test_safe_restore_rejects_unsafe_member_even_when_manifest_and_archive_are_rehashed(tmp_path, monkeypatch, mode):
    manifest = read_json(FIXTURE_MANIFEST)
    archive, manifest_path = tmp_path / "unsafe.zip", tmp_path / "unsafe_manifest.json"
    with zipfile.ZipFile(FIXTURE_ARCHIVE) as original, zipfile.ZipFile(archive, "x") as rewritten:
        for index, member in enumerate(original.infolist()):
            original_name = member.filename
            if index == 0 and mode == "traversal":
                member.filename = "../escape.json"
                next(row for row in manifest["files"] if row["path"] == original_name)["path"] = member.filename
            elif index == 0:
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
            rewritten.writestr(member, original.read(original_name))
    manifest.update(archive_bytes=archive.stat().st_size, archive_sha256=study.digest(archive))
    save_json(manifest_path, manifest)
    monkeypatch.setitem(globals(), "FIXTURE_ARCHIVE", archive)
    monkeypatch.setitem(globals(), "FIXTURE_MANIFEST", manifest_path)
    with pytest.raises((ValueError, AssertionError)):
        extract_existing_fixture(tmp_path / "safe_root")
    assert not (tmp_path / "escape.json").exists()


def test_registration_hashes_all_existing_inputs_without_decoding_or_learning(fixture, monkeypatch):
    monkeypatch.setattr(study, "load_npz", lambda *a, **k: pytest.fail("Registration decoded original states"))
    record = register(fixture)
    assert record["source_git_commit"] == "e" * 40
    assert len(record["input_files"]) == 106
    assert len({row["path"] for row in record["input_files"]}) == 106
    for row in record["input_files"]:
        assert (fixture["root"] / row["path"]).stat().st_size == row["bytes"]
        assert study.digest(fixture["root"] / row["path"]) == row["sha256"]
    assert not fixture["out"].exists()


@pytest.mark.parametrize("target", ["source", "config", "selected_npz", "nonselected_parent", "corrected_document"])
def test_all_input_parent_and_source_byte_changes_block_execution_before_output(fixture, target):
    record = register(fixture)
    paths = {"source": fixture["root"] / "src/drosolearn/state_decomposition.py", "config": fixture["config"],
             "selected_npz": fixture["root"] / record["input_files"][0]["path"],
             "nonselected_parent": fixture["root"] / "results/learning_ablation_v1/seed_17/trials.csv",
             "corrected_document": fixture["root"] / "docs/learning-ablation-findings-v2.md"}
    path = paths[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_selected_input_manifest_cannot_drop_or_change_registered_original_files(fixture):
    record = register(fixture)
    record["input_files"].pop()
    save_json(fixture["registration"], record)
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_parent_passing_verification_cannot_be_reused_for_rehashed_replacement_summary(fixture):
    base = fixture["root"] / "results/learning_ablation_v1"
    summary = read_json(base / "summary.json")
    summary["unverified_replacement"] = True
    save_json(base / "summary.json", summary)
    artifacts = read_json(base / "artifact_hashes.json")
    artifacts["summary.json"] = study.digest(base / "summary.json")
    save_json(base / "artifact_hashes.json", artifacts)
    fixture["cfg"]["parent_gate_identity"]["file_sha256"] = {name: study.digest(fixture["root"] / name) for name in fixture["cfg"]["parent_files"]}
    save_json(fixture["config"], fixture["cfg"])
    with pytest.raises(ValueError):
        register(fixture)


def test_existing_registration_and_result_directory_are_preserved(fixture):
    register(fixture)
    original = fixture["registration"].read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        register(fixture)
    assert fixture["registration"].read_bytes() == original
    fixture["out"].mkdir()
    keep = fixture["out"] / "keep.bin"
    keep.write_bytes(b"old static result")
    with pytest.raises((ValueError, FileExistsError)):
        run(fixture)
    assert keep.read_bytes() == b"old static result"
    assert list(fixture["out"].iterdir()) == [keep]


@pytest.mark.parametrize("mode", ["future", "naive", "git"])
def test_invalid_registration_chronology_or_source_version_is_rejected(fixture, mode):
    record = register(fixture)
    if mode == "git":
        record["source_git_commit"] = "missing"
    else:
        stamp = datetime.now(timezone.utc) + timedelta(days=1) if mode == "future" else datetime(2000, 1, 1)
        record["registered_at_utc"] = stamp.isoformat()
    save_json(fixture["registration"], record)
    with pytest.raises(ValueError):
        run(fixture)
    assert not fixture["out"].exists()


def test_static_run_reuses_completed_states_with_zero_all_learning_resources(completed):
    meta = read_json(completed["out"] / "metadata.json")
    assert meta["cases"] == 1536 and meta["crossed_policy_values"] == 6144
    assert all(meta[key] == 0 for key in ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws", "behavior_hypotheses_tested"))
    assert 0 < meta["maximum_case_csv_bytes"] < 100 * 1024 * 1024
    assert meta["human_or_species_comparison"] is False
    assert len(meta["input_files"]) == 106


def test_every_case_has_four_policies_same_smaller_target_and_target_aligned_identity(completed):
    rows = read_csv(completed["out"] / "seed_17/cases.csv")
    assert len(rows) == 768
    assert set(rows[0]) == set(study.CASE_FIELDS)
    for row in rows:
        target = 0 if int(row["left_count"]) < int(row["right_count"]) else 1
        sign = 2 * target - 1
        assert int(row["target_action"]) == target and int(row["target_sign"]) == sign
        for cross in study.CROSSES:
            p = float(row["p_right_" + cross])
            assert float(row["p_rule_" + cross]) == (p if target else 1 - p)
            assert float(row["aligned_raw_gap_" + cross]) == sign * float(row["raw_gap_" + cross])
            assert int(row["clipped_" + cross]) == int(abs(float(row["raw_gap_" + cross])) > 40)
        assert abs(float(row["bilinear_residual"])) <= 1e-10
        assert abs(float(row["raw_interaction_residual"])) <= 1e-10
        assert float(row["aligned_total_raw_gap_change"]) == pytest.approx(sum(float(row[field]) for field in
               ("aligned_head_term", "aligned_internal_term", "aligned_cross_term")), abs=1e-10)


def test_masked_frozen_initial_and_linear_terms_are_exactly_zero(completed):
    for row in read_csv(completed["out"] / "seed_17/cases.csv"):
        zeros = set()
        if row["model"] == "linear" or row["branch"] == "head_only":
            zeros.update(("internal_term", "cross_term"))
        if row["branch"] == "internal_only":
            zeros.update(("head_term", "cross_term"))
        if row["branch"] == "frozen" or row["trial"] == "0":
            zeros.update(("head_term", "internal_term", "cross_term", "total_raw_gap_change"))
        assert all(float(row[field]) == 0 for field in zeros)
    controls = read_json(completed["out"] / "controls.json")
    assert controls["state_pairs"] == 96
    assert controls["initial_state_checks"] == 32
    assert controls["frozen_state_checks"] == 24
    assert controls["saved_policy_entries_checked"] == 3072


def test_source_parent_states_remain_unchanged_and_every_output_hash_is_recorded(completed):
    reg, output = read_json(completed["registration"]), completed["out"]
    assert all(study.digest(completed["root"] / row["path"]) == row["sha256"] for row in reg["input_files"])
    assert all(study.digest(completed["root"] / name) == sha for name, sha in reg["parent_files"].items())
    artifacts = read_json(output / "artifact_hashes.json")
    assert set(artifacts) == {"registration.json", "metadata.json", "summary.json", "controls.json", "seed_17/cases.csv", "seed_18/cases.csv"}
    assert all(study.digest(output / name) == sha for name, sha in artifacts.items())
    summary = read_json(output / "summary.json")
    assert summary["cases"] == 1536 and summary["arithmetic_identity_passed"] is True
    assert summary["behavior_hypotheses_tested"] == 0
    assert len(summary["seed_summaries"]) == 192 and len(summary["cells"]) == 96


def test_midrun_failure_keeps_completed_seed_csv_and_partial_static_summaries(fixture, monkeypatch):
    register(fixture)
    original = study.load_npz
    def fail_second(path, *args, **kwargs):
        if "seed_18" in str(path):
            raise RuntimeError("deliberate second-seed static failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(study, "load_npz", fail_second)
    with pytest.raises(RuntimeError, match="deliberate second-seed static failure"):
        run(fixture)
    output = fixture["out"]
    metadata = read_json(output / "metadata.json")
    assert metadata["status"] == "failed" and metadata["cases"] == 768
    assert len(read_csv(output / "seed_17/cases.csv")) == 768
    assert len(read_json(output / "partial_seed_summaries.json")) == 96
    assert all(study.digest(output / name) == sha for name, sha in read_json(output / "artifact_hashes.json").items())
