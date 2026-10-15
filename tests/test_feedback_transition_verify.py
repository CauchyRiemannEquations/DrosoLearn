"""Synthetic transition-denominator and independent-tamper verification tests."""
from copy import deepcopy
import importlib.util
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_feedback_transition.py"
SPEC = importlib.util.spec_from_file_location("independent_feedback_transition_verifier", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def record(trial, choice=0, feedback=1, **changes):
    value = {"trial": trial, "source_position": trial + 1, "choice": choice, "feedback": feedback,
             "task": "cards", "phase": "task", "state": "1", "actual": True, "training": True}
    value.update(changes)
    return value


def measure(records):
    return VERIFY.measure_stream("synthetic/record", "human", "stable_card_index", records)


def test_independent_metric_uses_actual_current_feedback_and_counts_exact_adjacency():
    rows = [record(index + 1, choice, feedback)
            for index, (choice, feedback) in enumerate(zip([0, 0, 1, 1, 0], [1, 0, 1, 0, 1], strict=True))]
    summary, ledger = measure(rows)
    assert summary["trials"] == 5
    assert summary["candidates"] == summary["eligible"] == 4
    assert summary["excluded"] == 0
    assert summary["by_feedback"]["1"] == {"denominator": 2, "stay": 2, "switch": 0,
                                               "stay_rate": 1., "switch_rate": 0.}
    assert summary["by_feedback"]["0"] == {"denominator": 2, "stay": 0, "switch": 2,
                                               "stay_rate": 0., "switch_rate": 1.}
    assert [item["trial"] for item in ledger] == ["1", "2", "3", "4"]
    assert [item["stay"] for item in ledger] == ["1", "0", "1", "0"]


def test_query_is_not_removed_to_join_its_neighbors():
    summary, ledger = measure([record(1), record(2, "query", "information"), record(3, 1, 0)])
    assert summary["candidates"] == summary["excluded"] == 2
    assert summary["eligible"] == 0
    assert ledger[0]["reasons"] == "next_query"
    assert ledger[1]["reasons"] == "current_feedback_nonbinary|current_query"
    assert all(item["stay"] == "" for item in ledger)


def test_missing_choice_is_not_removed_to_join_its_neighbors():
    summary, ledger = measure([record(1), record(2, None, None), record(3, 1, 0)])
    assert summary["candidates"] == 2
    assert summary["eligible"] == 0
    assert ledger[0]["reasons"] == "next_choice_missing"
    assert ledger[1]["reasons"] == "current_choice_missing|current_feedback_missing"


def test_next_feedback_missing_does_not_remove_current_valid_pair():
    summary, ledger = measure([record(1, 0, 1), record(2, 1, None)])
    assert summary["eligible"] == 1
    assert summary["by_feedback"]["1"]["switch"] == 1
    assert ledger[0]["eligible"] == "1"


def test_current_feedback_missing_excludes_pair_but_is_not_zero_reward():
    summary, ledger = measure([record(1, 0, None), record(2, 1, 0)])
    assert summary["eligible"] == 0
    assert ledger[0]["reasons"] == "current_feedback_missing"
    assert ledger[0]["feedback"] == ""
    assert summary["by_feedback"]["0"]["denominator"] == 0


def test_zero_denominator_is_null_not_zero_or_perfect_rate():
    summary, _ = measure([record(1, 0, 0), record(2, 0, 1)])
    assert summary["by_feedback"]["0"]["stay_rate"] == 1.
    assert summary["by_feedback"]["1"]["stay_rate"] is None
    assert summary["by_feedback"]["1"]["switch_rate"] is None


def test_single_trial_stream_is_preserved_without_cross_record_transition():
    summary, ledger = measure([record(1)])
    assert summary["trials"] == 1
    assert summary["candidates"] == summary["eligible"] == summary["excluded"] == 0
    assert ledger == []


@pytest.mark.parametrize("field,value,reason", [
    ("task", "another_task", "task_boundary"),
    ("phase", "another_phase", "phase_boundary"),
    ("state", "2", "state_boundary"),
    ("trial", 3, "trial_gap"),
    ("actual", False, "not_actual"),
    ("training", False, "not_training"),
])
def test_boundary_gap_or_nonactual_pair_has_explicit_exclusion(field, value, reason):
    following = record(2)
    following[field] = value
    summary, ledger = measure([record(1), following])
    assert summary["excluded"] == 1
    assert summary["reason_counts"] == {reason: 1}
    assert ledger[0]["reasons"] == reason


def test_overlapping_exclusions_count_one_pair_and_retain_every_reason():
    rows = [record(1, "query", "information", actual=False),
            record(3, None, None, task="other", phase="new", state="2", training=False)]
    summary, ledger = measure(rows)
    reasons = {"current_query", "current_feedback_nonbinary", "next_choice_missing", "trial_gap",
               "task_boundary", "phase_boundary", "state_boundary", "not_actual", "not_training"}
    assert summary["excluded"] == 1
    assert set(summary["reason_counts"]) == reasons
    assert sum(summary["reason_counts"].values()) == 9
    assert ledger[0]["reasons"] == "|".join(sorted(reasons))
    assert ledger[0]["eligible"] == "0"


def test_stay_depends_on_canonical_option_identity_not_source_row_proximity():
    summary, ledger = measure([record(1, 0, 1, source_position=9), record(2, 0, 0, source_position=111)])
    assert summary["eligible"] == 1
    assert ledger[0]["stay"] == "1"
    assert ledger[0]["source_position"] == "9"
    assert ledger[0]["next_source_position"] == "111"


@pytest.mark.parametrize("field,value", [
    ("choice", True), ("choice", 1.), ("choice", "1"), ("choice", 2), ("choice", 3),
    ("feedback", True), ("feedback", 1.), ("feedback", 2),
    ("actual", 1), ("training", "True"),
    ("trial", True), ("trial", 1.), ("source_position", True),
    ("state", ""), ("phase", None), ("task", ""),
])
def test_invalid_canonical_types_are_rejected_not_coerced(field, value):
    rows = [record(1), record(2)]
    rows[0][field] = value
    with pytest.raises(VERIFY.VerificationError):
        measure(rows)


@pytest.mark.parametrize("field", ["trial", "source_position"])
def test_repeated_or_reversed_source_positions_are_not_sorted_or_skipped(field):
    rows = [record(1), record(2)]
    rows[1][field] = rows[0][field]
    with pytest.raises(VERIFY.VerificationError):
        measure(rows)


def test_independent_metric_leaves_inputs_unchanged():
    rows = [record(1), record(2, 1, 0)]
    original = deepcopy(rows)
    measure(rows)
    assert rows == original


def test_exact_json_comparison_distinguishes_boolean_count_from_integer_count():
    assert VERIFY.exact({"count": True}, {"count": 1}) is False
    assert VERIFY.exact({"rate": 1}, {"rate": 1.}) is False


def test_duplicate_or_nonfinite_json_is_rejected(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.read_json(path)
    path.write_text('{"rate": NaN}', encoding="utf-8")
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.read_json(path)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    from test_feedback_transition_run import make_fixture, register_fixture
    fixture = make_fixture(tmp_path, monkeypatch)
    root, cfg = fixture["root"], fixture["cfg"]
    cfg.update(exclude_boundaries=["state", "phase", "task"], zero_denominator="null", pool_group_or_species=False)
    cfg["human"].update(coordinate="source_coded_card_identity", historical_execution_version_verified=False)
    cfg["model"]["coordinate"] = "smaller_larger_quantity_category"
    cfg["frozen_files"] = sorted(VERIFY.FROZEN_PATHS)
    save_json(fixture["config_path"], cfg)
    # All data below came from make_fixture's invented arrays and records.
    origin = read_json(fixture["origin_path"])
    origin["trials_per_row"] = cfg["human"]["trials"]
    for entry in origin["source_files"]:
        entry["derived_group"] = "main"
    save_json(fixture["origin_path"], origin)
    prior_path = root / cfg["model"]["parent_config"]
    prior = read_json(prior_path)
    prior.update(seeds=cfg["model"]["seeds"], trials_per_phase=cfg["model"]["trials"], pair=[2, 4])
    save_json(prior_path, prior)
    task_path = root / cfg["human"]["structure_result"]
    task = read_json(task_path)
    task["gate"].update(exact_historical_task_execution_version_verified=False, direct_human_model_similarity_allowed=False)
    task["rows"] = [{"row_index": entry["derived_row_index"], "osf_file_id": entry["osf_file_id"],
                     "structural_passed": True} for entry in origin["source_files"]]
    task["inputs"] = [{key: entry[key] for key in
                       ("path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")}
                      for entry in origin["source_files"]]
    save_json(task_path, task)
    save_json(root / cfg["human"]["structure_verification"], {
        "status": "pass", "result_sha256": VERIFY.digest(task_path), "source_coded_descriptive_gate_passed": True})
    repository = SCRIPT.parents[1]
    for relative in cfg["frozen_files"]:
        target = root / relative
        if target == fixture["config_path"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if (repository / relative).is_file():
            shutil.copy2(repository / relative, target)
        else:
            target.write_text("Synthetic frozen fixture document.\n", encoding="utf-8")
    register_fixture(fixture)
    metadata = fixture["runner"].run(fixture["cache"], fixture["registration_path"], fixture["out"])
    assert metadata["status"] == "completed"
    return fixture


def verify_fixture(fixture):
    return VERIFY.verify(fixture["registration_path"], fixture["out"], fixture["cache"], root=fixture["root"])


def rehash_output(fixture):
    output = fixture["out"]
    save_json(output / "artifact_hashes.json", {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("metadata.json", "ledger.csv", "metrics.json")})


def test_completed_synthetic305_streams_replay_without_importing_primary_calculation(completed):
    result = verify_fixture(completed)
    assert result["status"] == "passed"
    assert result["human_records_replayed"] == 25
    assert result["model_streams_replayed"] == 280
    assert result["streams_verified"] == 305
    assert result["candidate_pairs_verified"] == 610
    assert result["primary_artifacts_verified"] == 3
    assert result["group_or_species_contrast_computed"] is False
    assert result["level4_or_level5_supported"] is False
    assert result["errors"] == []


@pytest.mark.parametrize("field,value", [
    ("denominator", 2), ("stay", 0), ("switch", 1), ("stay_rate", 0.), ("switch_rate", 1.),
])
def test_rehashed_wrong_metric_numerator_denominator_or_rate_is_detected(completed, field, value):
    path = completed["out"] / "metrics.json"
    report = read_json(path)
    report["streams"][0]["by_feedback"]["0"][field] = value
    save_json(path, report)
    rehash_output(completed)
    with pytest.raises(VERIFY.VerificationError, match="denominator, numerator, rate or scope"):
        verify_fixture(completed)


def test_rehashed_zero_denominator_rate_is_not_allowed_to_become_zero(completed):
    path = completed["out"] / "metrics.json"
    report = read_json(path)
    assert report["streams"][0]["by_feedback"]["1"]["denominator"] == 0
    report["streams"][0]["by_feedback"]["1"]["stay_rate"] = 0.
    save_json(path, report)
    rehash_output(completed)
    with pytest.raises(VERIFY.VerificationError):
        verify_fixture(completed)


def test_rehashed_count_as_boolean_is_not_equal_to_integer_one(completed):
    path = completed["out"] / "metrics.json"
    report = read_json(path)
    report["streams"][0]["by_feedback"]["0"]["denominator"] = True
    save_json(path, report)
    rehash_output(completed)
    with pytest.raises(VERIFY.VerificationError):
        verify_fixture(completed)


@pytest.mark.parametrize("field,value", [
    ("eligible", "0"), ("reasons", "phase_boundary"), ("feedback", "1"),
    ("stay", "0"), ("source_position", "99"), ("next_trial", "3"),
])
def test_rehashed_ledger_adjacency_exclusion_or_choice_is_detected(completed, field, value):
    path = completed["out"] / "ledger.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=VERIFY.LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    rehash_output(completed)
    with pytest.raises(VERIFY.VerificationError, match="adjacency, exclusions or choice transition"):
        verify_fixture(completed)


def test_rehashed_ledger_cannot_silently_drop_a_candidate_pair(completed):
    path = completed["out"] / "ledger.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8")
    rehash_output(completed)
    with pytest.raises(VERIFY.VerificationError):
        verify_fixture(completed)


@pytest.mark.parametrize("field", ["group_or_species_contrast_computed", "new_model_learning_executed", "level4_or_level5_supported"])
def test_rehashed_metadata_cannot_expand_scientific_scope(completed, field):
    path = completed["out"] / "metadata.json"
    metadata = read_json(path)
    metadata[field] = True
    save_json(path, metadata)
    rehash_output(completed)
    with pytest.raises(VERIFY.VerificationError, match="interpretation scope"):
        verify_fixture(completed)


def test_frozen_protocol_edit_is_detected(completed):
    path = completed["root"] / "docs/feedback-transition-protocol.md"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(VERIFY.VerificationError, match="Frozen code"):
        verify_fixture(completed)


@pytest.mark.parametrize("role", ["human_raw", "model_actual"])
def test_raw_source_change_is_detected_before_metric_replay(completed, role):
    registration = read_json(completed["registration_path"])
    entry = next(entry for entry in registration["inputs"] if entry["role"] == role)
    path = (completed["cache"] if role == "human_raw" else completed["root"]) / entry["path"]
    path.write_bytes(path.read_bytes() + b"synthetic alteration")
    with pytest.raises(VERIFY.VerificationError):
        verify_fixture(completed)


def test_parent_result_binding_change_is_detected(completed):
    path = completed["root"] / completed["cfg"]["human"]["structure_result"]
    value = read_json(path)
    value["synthetic_extra"] = True
    save_json(path, value)
    with pytest.raises(VERIFY.VerificationError, match="Human structure independent gate"):
        verify_fixture(completed)


@pytest.mark.parametrize("field,value", [
    ("path", "raw/different.mat"), ("bytes", 11), ("sha256", "1" * 64),
    ("derived_group", "pilot"), ("derived_row_index", 9), ("osf_file_id", "different_id"),
])
def test_parent_gate_binding_compares_bytes_path_row_group_and_file_id(field, value):
    old = {"path": "raw/a.mat", "bytes": 10, "sha256": "0" * 64, "derived_group": "main",
           "derived_row_index": 0, "osf_file_id": "synthetic_id"}
    newer = {"path": old["path"], "bytes": old["bytes"], "sha256": old["sha256"],
             "row_index": 0, "osf_file_id": "synthetic_id"}
    assert VERIFY.check_task_source_binding({"inputs": [old]}, [newer]) is None
    changed = dict(old)
    changed[field] = value
    with pytest.raises(VERIFY.VerificationError, match="source byte/row/group binding"):
        VERIFY.check_task_source_binding({"inputs": [changed]}, [newer])


@pytest.mark.parametrize("alteration", ["reverse", "drop", "duplicate"])
def test_parent_raw_identity_order_coverage_and_uniqueness_are_preserved(alteration):
    old = [{"path": f"raw/{index}.mat", "bytes": 10, "sha256": str(index) * 64,
            "derived_group": "main", "derived_row_index": index, "osf_file_id": f"synthetic{index}"}
           for index in range(2)]
    newer = [{"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"],
              "row_index": item["derived_row_index"], "osf_file_id": item["osf_file_id"]} for item in old]
    assert VERIFY.check_task_source_binding({"inputs": old}, newer) is None
    altered = list(reversed(old)) if alteration == "reverse" else (old[:-1] if alteration == "drop" else old + [old[0]])
    with pytest.raises(VERIFY.VerificationError, match="source byte/row/group binding"):
        VERIFY.check_task_source_binding({"inputs": altered}, newer)


def test_rehashed_new_raw_cannot_reuse_old_human_structure_gate(completed):
    from scipy.io import loadmat, savemat
    origin = read_json(completed["origin_path"])
    entry = origin["source_files"][0]
    path = completed["cache"] / entry["path"]
    arrays = {key: value for key, value in loadmat(path).items() if not key.startswith("__")}
    arrays["R"][0, 0] = 1 if arrays["R"][0, 0] == -1 else -1
    savemat(path, arrays)
    entry.update(bytes=path.stat().st_size, sha256=VERIFY.digest(path))
    save_json(completed["origin_path"], origin)
    registration = read_json(completed["registration_path"])
    for item in registration["inputs"]:
        if item["role"] == "human_raw" and item["osf_file_id"] == entry["osf_file_id"]:
            item.update(bytes=entry["bytes"], sha256=entry["sha256"])
        elif item["role"] == "parent" and item["path"] == completed["cfg"]["human"]["origin_manifest"]:
            item.update(bytes=completed["origin_path"].stat().st_size, sha256=VERIFY.digest(completed["origin_path"]))
    save_json(completed["registration_path"], registration)
    metadata_path = completed["out"] / "metadata.json"
    metadata = read_json(metadata_path)
    metadata.update(inputs=registration["inputs"], registration_sha256=VERIFY.digest(completed["registration_path"]))
    save_json(metadata_path, metadata)
    rehash_output(completed)
    with pytest.raises(VERIFY.VerificationError, match="source byte/row/group binding"):
        verify_fixture(completed)


def test_cli_pass_preserves_outputs_and_refuses_existing_verification(completed, monkeypatch):
    monkeypatch.setattr(VERIFY, "ROOT", completed["root"])
    destination = completed["out"] / "independent.json"
    args = ["--registration", str(completed["registration_path"]), "--result-dir", str(completed["out"]),
            "--human-cache", str(completed["cache"]), "--out", str(destination)]
    assert VERIFY.main(args) == 0
    original = destination.read_bytes()
    assert read_json(destination)["status"] == "passed"
    with pytest.raises(VERIFY.VerificationError, match="already exists"):
        VERIFY.main(args)
    assert destination.read_bytes() == original


def test_cli_failure_is_retained_without_changing_primary_artifacts(completed, monkeypatch):
    monkeypatch.setattr(VERIFY, "ROOT", completed["root"])
    ledger_path = completed["out"] / "ledger.csv"
    ledger_path.write_bytes(ledger_path.read_bytes() + b"bad row\n")
    destination = completed["out"] / "failed_verification.json"
    args = ["--registration", str(completed["registration_path"]), "--result-dir", str(completed["out"]),
            "--human-cache", str(completed["cache"]), "--out", str(destination)]
    assert VERIFY.main(args) == 1
    failure = read_json(destination)
    assert failure["status"] == "failed"
    assert failure["errors"]
    assert failure["group_or_species_contrast_computed"] is False
