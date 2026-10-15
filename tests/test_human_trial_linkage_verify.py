"""Synthetic-only independent linkage replay and rehashed semantic corruption."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from drosolearn.human_trial_linkage import register, run
from test_human_trial_linkage_run import audit_fixture


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_human_trial_linkage.py"
SPEC = importlib.util.spec_from_file_location("independent_human_linkage_verifier", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(tmp_path):
    root, cfg, reg, out = audit_fixture(tmp_path)
    run(root, cfg, reg, out)
    return root, out


def rehash(out):
    path = out / "artifact_hashes.json"
    write(path, {name: digest(out / name) for name in read(path)})


def change_csv(out, name, transform):
    path = out / name
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    transform(fields, rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rehash(out)


def refresh_source_manifest(root):
    path = root / "data/human/stolz_2022_openneuro/source-manifest.json"
    manifest = read(path)
    for item in manifest["files"]:
        source = path.parent / item["path"]
        item.update(size_bytes=source.stat().st_size, sha256=digest(source))
    write(path, manifest)


def test_independent_verifier_replays_all_seven_primary_artifacts(tmp_path):
    _, out = prepare(tmp_path)
    original = {path: path.read_bytes() for path in out.iterdir()}
    result = VERIFIER.verify(out)
    assert result["status"] == "passed"
    assert result["primary_artifacts_verified"] == 7
    assert result["matrix_pairs_verified"] == 6
    assert result["degree_one_pairs_verified"] == 2
    assert result["feedback_blocks_verified"] == 24
    assert result["numeric_diagnostic_rows_verified"] == 16
    assert result["tail_event_rows_preserved"] == 3
    assert all(path.read_bytes() == data for path, data in original.items())


def test_cli_writes_only_exclusive_verification_file(tmp_path):
    _, out = prepare(tmp_path)
    before = {path: path.read_bytes() for path in out.iterdir()}
    command = [sys.executable, str(SCRIPT), "--output", str(out)]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr + process.stdout
    target = out / "verification.json"
    assert read(target)["status"] == "passed"
    original = target.read_bytes()
    assert set(out.iterdir()) == set(before) | {target}
    assert all(path.read_bytes() == data for path, data in before.items())
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert target.read_bytes() == original


@pytest.mark.parametrize("filename,field,value", [
    ("alignment-matrix.csv", "REW_equal", "False"),
    ("alignment-matrix.csv", "both_equal", "False"),
    ("record-links.csv", "mat_row", "mat-02"),
    ("corroboration.csv", "strict_comparable", "False"),
    ("corroboration.csv", "feedback_event_row", "999"),
    ("corroboration.csv", "task", "PUN"),
    ("corroboration.csv", "event_choice_side_code", "2"),
    ("corroboration.csv", "event_expectation_plus3", "6"),
    ("corroboration.csv", "mat_action_code", "2"),
    ("corroboration.csv", "mat_expect_code", "6"),
    ("corroboration.csv", "action_numeric_agreement", "False"),
    ("corroboration.csv", "expectation_numeric_agreement", "False"),
])
def test_rehashed_csv_corruption_is_rejected(tmp_path, filename, field, value):
    _, out = prepare(tmp_path)
    change_csv(out, filename, lambda fields, rows: rows[0].update({field: value}))
    with pytest.raises(VERIFIER.VerificationError, match="CSV semantic replay"):
        VERIFIER.verify(out)


def test_rehashed_missing_matrix_pair_is_rejected(tmp_path):
    _, out = prepare(tmp_path)
    change_csv(out, "alignment-matrix.csv", lambda fields, rows: rows.pop())
    with pytest.raises(VERIFIER.VerificationError, match="CSV row count"):
        VERIFIER.verify(out)


@pytest.mark.parametrize("change", ["global_previous", "tail", "strict", "segment", "issue"])
def test_rehashed_global_partition_or_flag_change_is_rejected(tmp_path, change):
    _, out = prepare(tmp_path)
    path = out / "feedback-blocks.json"
    data = read(path)
    entry = data["sub-s1"]
    if change == "global_previous":
        entry["blocks"][4]["previous_feedback_event_row"] = None
    elif change == "tail":
        entry["tail_event_rows"] = []
    elif change == "strict":
        entry["blocks"][0]["strict_comparable"] = False
    elif change == "segment":
        entry["blocks"][0]["segment_event_rows"].pop()
    else:
        entry["blocks"][0]["issues"] = ["invented_issue"]
    write(path, data)
    rehash(out)
    with pytest.raises(VERIFIER.VerificationError, match="Global feedback block/tail"):
        VERIFIER.verify(out)


@pytest.mark.parametrize("field,value", [("matrix_pairs", 5), ("exact_feedback_candidate_pairs", 3),
                                        ("mat_rows", 2.0), ("event_files", True)])
def test_rehashed_summary_count_or_json_type_change_is_rejected(tmp_path, field, value):
    _, out = prepare(tmp_path)
    path = out / "linkage-report.json"
    data = read(path)
    data[field] = value
    write(path, data)
    rehash(out)
    with pytest.raises(VERIFIER.VerificationError, match="Linkage report semantic replay"):
        VERIFIER.verify(out)


def test_shape_identity_cannot_be_claimed_from_numeric_agreement(tmp_path):
    _, out = prepare(tmp_path)
    path = out / "linkage-report.json"
    data = read(path)
    data["readiness"]["action_identity"] = "verified_shape_identity"
    write(path, data)
    rehash(out)
    with pytest.raises(VERIFIER.VerificationError, match="Linkage report semantic replay"):
        VERIFIER.verify(out)


def test_source_rehash_cannot_hide_semantic_drift(tmp_path):
    root, out = prepare(tmp_path)
    event = root / "data/human/stolz_2022_openneuro/sub-s1/eeg/sub-s1_task-task_events.tsv"
    event.write_text(event.read_text().replace("S61", "S60", 1), encoding="utf-8")
    refresh_source_manifest(root)
    # Even after deliberately re-authoring all local hash links, stale semantic
    # outputs must fail. This does not claim resistance to rewriting every file.
    reg_path = out / "registration.json"
    reg = read(reg_path)
    for name in reg["source_hashes"]:
        reg["source_hashes"][name] = digest(root / name)
    write(reg_path, reg)
    metadata_path = out / "metadata.json"
    metadata = read(metadata_path)
    metadata["registration_sha256"] = digest(reg_path)
    write(metadata_path, metadata)
    rehash(out)
    with pytest.raises(VERIFIER.VerificationError, match="Linkage report semantic replay"):
        VERIFIER.verify(out)


def test_source_change_without_rehash_fails_integrity(tmp_path):
    root, out = prepare(tmp_path)
    event = root / "data/human/stolz_2022_openneuro/sub-s1/eeg/sub-s1_task-task_events.tsv"
    event.write_bytes(event.read_bytes() + b"\n")
    with pytest.raises(VERIFIER.VerificationError, match="Manifest/source bytes"):
        VERIFIER.verify(out)


def test_both_sided_degree_rule_retains_ambiguous_records_without_greedy_link(tmp_path):
    root, cfg, _, out = audit_fixture(tmp_path)
    source = root / "data/human/stolz_2022_openneuro"
    first = source / "sub-s1/eeg/sub-s1_task-task_events.tsv"
    duplicate = source / "sub-s4/eeg/sub-s4_task-task_events.tsv"
    duplicate.parent.mkdir(parents=True)
    shutil.copyfile(first, duplicate)
    path = source / "source-manifest.json"
    manifest = read(path)
    manifest["files"].append({"path": duplicate.relative_to(source).as_posix(),
                              "size_bytes": duplicate.stat().st_size, "sha256": digest(duplicate)})
    write(path, manifest)
    reg = root / "results/ambiguous-registration.json"
    register(root, cfg, reg)
    run(root, cfg, reg, out)
    result = VERIFIER.verify(out)
    report = read(out / "linkage-report.json")
    assert result["degree_one_pairs_verified"] == 1
    assert report["ambiguous_mat_rows"] == ["mat-01"]
    assert report["event_candidates"]["sub-s1"] == report["event_candidates"]["sub-s4"] == ["mat-01"]
    assert result["event_files_verified"] == 4


def test_incomplete_feedback_segment_keeps_numeric_columns_without_comparing_them(tmp_path):
    root, cfg, _, out = audit_fixture(tmp_path)
    event = root / "data/human/stolz_2022_openneuro/sub-s1/eeg/sub-s1_task-task_events.tsv"
    lines = event.read_text().splitlines()
    event.write_text("\n".join(line for line in lines if not line.endswith("\tS53")) + "\n", encoding="utf-8")
    refresh_source_manifest(root)
    reg = root / "results/incomplete-registration.json"
    register(root, cfg, reg)
    run(root, cfg, reg, out)
    result = VERIFIER.verify(out)
    report = read(out / "linkage-report.json")
    assert result["degree_one_pairs_verified"] == 2
    assert report["corroboration_counts"]["not_strict_comparable"] == 1
    with (out / "corroboration.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    first = rows[0]
    assert first["strict_comparable"] == "False"
    assert first["mat_expect_code"] == "3"
    assert first["event_expectation_plus3"] == first["expectation_numeric_agreement"] == ""
