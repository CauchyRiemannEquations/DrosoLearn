"""Synthetic-only tests: event provenance, task context and missingness.

No original human file is opened and no behavioral performance is calculated.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from drosolearn.human_trial_data import normalize_marker, parse_events, read_events, register, run


def events(*markers, rows=None):
    result = [
        {"onset": str(index), "duration": "0.1", "sample": str(index * 1000), "value": marker}
        for index, marker in enumerate(markers)
    ]
    if rows is not None:
        for event, row in zip(result, rows, strict=True):
            event["row"] = row
    return result


def write_events(tmp_path, rows, fields=("onset", "duration", "sample", "value")):
    path = tmp_path / "synthetic_events.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("raw", ["S 51", "S_51", "S51"])
def test_specified_marker_variants_share_canonical_code(raw):
    assert normalize_marker(raw) == "S51"


@pytest.mark.parametrize("raw", ["", "unknown", "51", "S51junk", "S-51"])
def test_unrecognized_markers_are_not_silently_coerced(raw):
    assert normalize_marker(raw) is None


def test_unknown_numeric_code_stays_distinct_from_known_code():
    assert normalize_marker("S999") == "S999"
    report = parse_events(events("S100", "S999", "S101"))
    assert report["issues"]
    assert report["choices"] == []


def test_tsv_reader_retains_literal_event_values_and_original_data_lines(tmp_path):
    rows = events("S100", "S 51", "S_53", "S61", "S101")
    loaded = read_events(write_events(tmp_path, rows))
    assert len(loaded) == len(rows)
    for expected, actual, row_number in zip(rows, loaded, range(2, 7), strict=True):
        assert {key: actual[key] for key in expected} == expected
        assert actual["row"] == row_number


@pytest.mark.parametrize("fields", [
    ("onset", "duration", "value"),
    ("onset", "duration", "sample", "value", "accuracy"),
])
def test_reader_rejects_schema_drift_instead_of_guessing(tmp_path, fields):
    path = tmp_path / "bad_schema.tsv"
    path.write_text("\t".join(fields) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_events(path)


def test_empty_event_stream_is_rejected_instead_of_inventing_trials():
    with pytest.raises(ValueError):
        parse_events([])


@pytest.mark.parametrize("selection,expectation,feedback,side,level,positive", [
    ("S51", "S53", "S60", "left", 0, 0),
    ("S52", "S56", "S61", "right", 3, 1),
    ("S71", "S73", "S80", "left", 0, 0),
    ("S72", "S76", "S81", "right", 3, 1),
])
def test_complete_choice_uses_documented_code_meaning_and_provenance(selection, expectation, feedback, side, level, positive):
    start, end = ("S100", "S101") if selection in {"S51", "S52"} else ("S200", "S201")
    report = parse_events(events(start, selection, expectation, feedback, end))
    assert report["issues"] == []
    assert len(report["choices"]) == 1
    choice = report["choices"][0]
    assert choice["choice_side"] == side
    assert choice["expectation_level"] == level
    assert choice["feedback_code"] == feedback
    assert choice["positive_feedback_flag"] == positive
    assert choice["complete"] is True
    assert choice["observed_choice_index"] == 1
    assert choice["choice_event_row"] == 3
    assert choice["expectation_event_row"] == 4
    assert choice["feedback_event_row"] == 5
    assert float(choice["choice_onset"]) == 1
    assert float(choice["feedback_onset"]) == 3


@pytest.mark.parametrize("code,level", [("S53", 0), ("S54", 1), ("S55", 2), ("S56", 3)])
def test_expectation_is_ordinal_and_is_not_converted_to_probability(code, level):
    choice = parse_events(events("S100", "S51", code, "S61", "S101"))["choices"][0]
    assert choice["expectation_level"] == level
    assert not {"choice_probability", "reward_probability", "accuracy", "correct", "reversal"} & set(choice)


def test_explicit_source_line_numbers_survive_gaps():
    report = parse_events(events("S100", "S51", "S54", "S61", "S101", rows=[2, 7, 9, 15, 18]))
    choice = report["choices"][0]
    assert (choice["choice_event_row"], choice["expectation_event_row"], choice["feedback_event_row"]) == (7, 9, 15)


def test_new_selection_retains_previous_incomplete_choice():
    report = parse_events(events("S100", "S51", "S52", "S54", "S61", "S101"))
    assert report["issues"]
    assert len(report["choices"]) == 2
    first, second = report["choices"]
    assert first["complete"] is False
    assert first["feedback_code"] is None
    assert second["complete"] is True
    assert [item["observed_choice_index"] for item in report["choices"]] == [1, 2]


def test_missing_expectation_is_not_filled_by_feedback():
    report = parse_events(events("S100", "S51", "S61", "S101"))
    assert report["issues"]
    choice = report["choices"][0]
    assert choice["expectation_level"] is None
    assert choice["complete"] is False


def test_missing_feedback_stays_missing_rather_than_becoming_zero():
    report = parse_events(events("S100", "S51", "S53", "S101"))
    assert report["issues"]
    choice = report["choices"][0]
    assert choice["complete"] is False
    assert choice["feedback_code"] is None
    assert choice["positive_feedback_flag"] is None


def test_pending_selection_at_end_is_preserved():
    report = parse_events(events("S100", "S52", "S56"))
    assert report["issues"]
    assert len(report["choices"]) == 1
    assert report["choices"][0]["complete"] is False


def test_feedback_without_selection_does_not_create_response():
    report = parse_events(events("S100", "S53", "S61", "S101"))
    assert report["issues"]
    assert report["choices"] == []


def test_duplicate_expectation_marks_choice_incomplete():
    report = parse_events(events("S100", "S51", "S53", "S54", "S61", "S101"))
    assert report["issues"]
    assert len(report["choices"]) == 1
    assert report["choices"][0]["complete"] is False


def test_other_task_expectation_cannot_complete_current_choice():
    report = parse_events(events("S100", "S51", "S73", "S61", "S101"))
    assert report["issues"]
    assert report["choices"][0]["complete"] is False


def test_other_task_feedback_cannot_complete_current_choice():
    report = parse_events(events("S100", "S51", "S53", "S81", "S101"))
    assert report["issues"]
    assert report["choices"][0]["complete"] is False


def test_task_end_prevents_late_feedback_linkage():
    report = parse_events(events("S100", "S51", "S53", "S101", "S61"))
    assert report["issues"]
    choice = report["choices"][0]
    assert choice["complete"] is False
    assert choice["feedback_code"] is None


def test_selection_outside_task_remains_visible_but_incomplete():
    report = parse_events(events("S51", "S53", "S61"))
    assert report["issues"]
    assert len(report["choices"]) == 1
    assert report["choices"][0]["complete"] is False


def test_pause_markers_do_not_create_trials_or_responses():
    report = parse_events(events("S100", "S51", "S53", "S61", "S150", "S151", "S52", "S56", "S60", "S101"))
    assert report["issues"] == []
    assert len(report["choices"]) == 2
    assert all(choice["complete"] for choice in report["choices"])
    assert sum(report["marker_counts"].values()) == 10


def test_unmatched_pause_is_recorded():
    report = parse_events(events("S100", "S151", "S101"))
    assert report["issues"]
    assert report["choices"] == []


def test_decreasing_onset_is_flagged_and_not_sorted():
    raw = events("S100", "S51", "S53", "S61", "S101")
    raw[2]["onset"] = "0.5"
    report = parse_events(raw)
    assert report["issues"]
    assert report["events"] == len(raw)
    assert report["choices"][0]["choice_event_row"] == 3
    assert report["choices"][0]["expectation_event_row"] == 4
    assert report["choices"][0]["complete"] is False


def test_raw_event_and_input_list_are_not_rewritten():
    raw = events("S100", "S 51", "S_53", "S61", "S101")
    original = [dict(row) for row in raw]
    parse_events(raw)
    assert raw == original


@pytest.mark.parametrize("field,value", [
    ("onset", "not-a-number"), ("onset", "nan"), ("onset", "inf"),
    ("duration", "not-a-number"), ("sample", "not-a-number"),
])
def test_malformed_numeric_fields_are_rejected(field, value):
    raw = events("S100")
    raw[0][field] = value
    with pytest.raises(ValueError):
        parse_events(raw)


def test_missing_required_raw_field_is_rejected():
    raw = events("S100")
    del raw[0]["sample"]
    with pytest.raises(ValueError):
        parse_events(raw)


@pytest.fixture
def audit_fixture(tmp_path):
    """New repository with invented metadata and invented event rows only."""
    root = tmp_path / "synthetic_repository"
    root.mkdir()
    source = root / "data/human/stolz_2022_openneuro"
    event_dir = source / "sub-synthetic/eeg"
    event_dir.mkdir(parents=True)
    (source / "README").write_text("Synthetic software fixture; no human data.\n", encoding="utf-8")
    (source / "dataset_description.json").write_text(json.dumps({
        "License": "CC0", "DatasetDOI": "doi:10.18112/openneuro.ds004295.v1.0.0",
        "Name": "Synthetic test identity only",
    }), encoding="utf-8")
    marker_codes = ["S51", "S52", "S53", "S54", "S55", "S56", "S60", "S61",
                    "S71", "S72", "S73", "S74", "S75", "S76", "S80", "S81",
                    "S100", "S101", "S150", "S151", "S200", "S201"]
    (source / "task-task_events.json").write_text(json.dumps({
        "value": {"Levels": {code: "synthetic fixture code" for code in marker_codes}}
    }), encoding="utf-8")
    event_path = event_dir / "sub-synthetic_task-task_events.tsv"
    raw = events("S100", "S51", "S53", "S61", "S101",
                 "S200", "S72", "S76", "S80", "S201")
    with event_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("onset", "duration", "sample", "value"), delimiter="\t")
        writer.writeheader()
        writer.writerows(raw)
    manifest = {"fixture_notice": "No real participants or observations", "files": [
        {"path": path.relative_to(source).as_posix(), "size_bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(source.rglob("*")) if path.is_file()
    ]}
    (source / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    for relative in ("src/drosolearn/human_trial_data.py", "tests/test_human_trial_data.py",
                     "scripts/verify_human_trial_data.py", "tests/test_human_trial_data_verify.py",
                     "docs/human-trial-data-protocol.md", "requirements-lock.txt", "pyproject.toml"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
    config = {"audit_id": "human_trial_data_readiness_v1", "scope": "source_and_structure_only",
              "adapter": "openneuro_ds004295_events_v1", "performance_analysis": False,
              "source_root": "data/human/stolz_2022_openneuro",
              "source_manifest": "data/human/stolz_2022_openneuro/source-manifest.json"}
    config_path = root / "configs/synthetic.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Synthetic Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "--allow-empty", "-m", "synthetic fixture"],
                   check=True, capture_output=True)
    return root, config_path, root / "registration.json", root / "new_results"


def test_registration_freezes_protocol_source_data_and_test_bytes(audit_fixture):
    root, config, registration, _ = audit_fixture
    record = register(root, config, registration)
    frozen = record["source_hashes"]
    assert "docs/human-trial-data-protocol.md" in frozen
    assert "tests/test_human_trial_data.py" in frozen
    assert "src/drosolearn/human_trial_data.py" in frozen
    assert "data/human/stolz_2022_openneuro/source-manifest.json" in frozen
    assert any(name.endswith("_events.tsv") for name in frozen)
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == value for name, value in frozen.items())
    assert record["config"]["performance_analysis"] is False
    assert "exploratory" in record["notice"]


def test_registration_does_not_overwrite_prior_record(audit_fixture):
    root, config, registration, _ = audit_fixture
    register(root, config, registration)
    original = registration.read_bytes()
    with pytest.raises(FileExistsError):
        register(root, config, registration)
    assert registration.read_bytes() == original


def test_completed_audit_preserves_source_and_records_only_structural_observations(audit_fixture):
    root, config, registration, output = audit_fixture
    register(root, config, registration)
    source = root / "data/human/stolz_2022_openneuro"
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*") if path.is_file()}
    report = run(root, config, registration, output)
    assert report["source_integrity_passed"] is True
    assert report["event_files"] == 1
    assert report["event_rows"] == 10
    assert report["observed_choices"] == report["complete_triplets"] == 2
    assert report["issues_count"] == 0
    assert report["readiness"]["human_model_similarity"] == "not_tested"
    assert not {"accuracy", "mean_accuracy", "learning_curve", "effect_size", "correlation"} & set(report)
    with (output / "observed_choices.csv").open(encoding="utf-8", newline="") as stream:
        choices = list(csv.DictReader(stream))
    assert len(choices) == 2
    assert {row["task"] for row in choices} == {"REW", "PUN"}
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
    artifacts = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert all(hashlib.sha256((output / name).read_bytes()).hexdigest() == value for name, value in artifacts.items())
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_run_refuses_existing_output_without_overwriting(audit_fixture):
    root, config, registration, output = audit_fixture
    register(root, config, registration)
    run(root, config, registration, output)
    original = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        run(root, config, registration, output)
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("relative", [
    "src/drosolearn/human_trial_data.py", "tests/test_human_trial_data.py",
    "docs/human-trial-data-protocol.md", "requirements-lock.txt", "pyproject.toml",
    "data/human/stolz_2022_openneuro/source-manifest.json",
    "data/human/stolz_2022_openneuro/sub-synthetic/eeg/sub-synthetic_task-task_events.tsv",
])
def test_frozen_source_drift_is_rejected_before_result_creation(audit_fixture, relative):
    root, config, registration, output = audit_fixture
    register(root, config, registration)
    target = root / relative
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after audit freeze"):
        run(root, config, registration, output)
    assert not output.exists()


def test_even_whitespace_only_config_change_breaks_freeze(audit_fixture):
    root, config, registration, output = audit_fixture
    register(root, config, registration)
    config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after audit freeze"):
        run(root, config, registration, output)
    assert not output.exists()


def test_performance_scope_cannot_be_enabled_in_config(audit_fixture):
    root, config, registration, _ = audit_fixture
    value = json.loads(config.read_text(encoding="utf-8"))
    value["performance_analysis"] = True
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="scope"):
        register(root, config, registration)
    assert not registration.exists()


def test_failed_source_identity_audit_leaves_failure_record(audit_fixture):
    root, config, registration, output = audit_fixture
    source = root / "data/human/stolz_2022_openneuro"
    description_path = source / "dataset_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    description["License"] = "synthetic incompatible license"
    description_path.write_text(json.dumps(description), encoding="utf-8")
    manifest_path = source / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = next(item for item in manifest["files"] if item["path"] == "dataset_description.json")
    item.update(size_bytes=description_path.stat().st_size, sha256=hashlib.sha256(description_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    register(root, config, registration)
    with pytest.raises(ValueError, match="license or identity"):
        run(root, config, registration, output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "license or identity" in metadata["failure"]
    assert not (output / "structure-audit.json").exists()


def test_manifest_path_cannot_escape_selected_dataset(audit_fixture):
    root, config, registration, _ = audit_fixture
    source = root / "data/human/stolz_2022_openneuro"
    outside = source.parent / "outside.txt"
    outside.write_text("synthetic file outside selected dataset", encoding="utf-8")
    manifest_path = source / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append({"path": "../outside.txt", "size_bytes": outside.stat().st_size,
                              "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the selected dataset"):
        register(root, config, registration)
    assert not registration.exists()


def test_execution_metadata_source_root_remains_relative_and_portable(audit_fixture):
    root, config, registration, output = audit_fixture
    register(root, config, registration)
    run(root, config, registration, output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    relative_root = Path(metadata["source_root"])
    assert not relative_root.is_absolute()
    assert (output / relative_root).resolve() == root.resolve()
