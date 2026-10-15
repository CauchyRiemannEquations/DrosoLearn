"""Synthetic-only independent replay and rehashed-corruption tests."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from drosolearn.human_trial_data import register, run
from test_human_trial_data import audit_fixture as base_audit_fixture  # noqa: F401


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_human_trial_data.py"
SPEC = importlib.util.spec_from_file_location("human_trial_data_independent_verifier", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture
def audit_fixture(base_audit_fixture):
    """Extend the shared invented-source fixture with the independent verifier."""
    root = base_audit_fixture[0]
    for relative in ("scripts/verify_human_trial_data.py", "tests/test_human_trial_data_verify.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPT.parents[1] / relative, target)
    return base_audit_fixture


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def rehash(output):
    path = output / "artifact_hashes.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    dump(path, {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in old})


def prepare(audit_fixture):
    root, config, registration, output = audit_fixture
    register(root, config, registration)
    run(root, config, registration, output)
    return root, output


def alter_choices(output, transform):
    path = output / "observed_choices.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    transform(fields, rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rehash(output)


def test_independent_replay_passes_complete_source_without_modifying_artifacts(audit_fixture):
    root, output = prepare(audit_fixture)
    before = {path: path.read_bytes() for path in output.iterdir()}
    result = VERIFIER.verify(output)
    assert result["status"] == "passed"
    assert result["event_rows_verified"] == 10
    assert result["observed_choice_rows_verified"] == result["complete_triplets_verified"] == 2
    assert result["incomplete_choices_preserved"] == result["issues_preserved"] == 0
    assert all(path.read_bytes() == data for path, data in before.items())


def test_command_creates_only_new_verification_and_refuses_overwrite(audit_fixture):
    _, output = prepare(audit_fixture)
    before = {path: path.read_bytes() for path in output.iterdir()}
    command = [sys.executable, str(SCRIPT), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    verification = output / "verification.json"
    assert json.loads(verification.read_text(encoding="utf-8"))["status"] == "passed"
    previous = verification.read_bytes()
    assert set(output.iterdir()) == set(before) | {verification}
    assert all(path.read_bytes() == data for path, data in before.items())
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert verification.read_bytes() == previous


@pytest.mark.parametrize("field,value", [
    ("choice_side", "right"), ("choice_event_row", "4"),
    ("choice_onset", "999"), ("task", "PUN"),
    ("expectation_level", "0.0"), ("expectation_event_row", "5"),
    ("feedback_code", "S60"), ("positive_feedback_flag", "0"),
    ("feedback_event_row", "9"), ("feedback_onset", "999"),
    ("complete", "False"), ("observed_choice_index", "2"),
])
def test_rehashed_choice_corruption_still_fails_raw_event_replay(audit_fixture, field, value):
    _, output = prepare(audit_fixture)
    alter_choices(output, lambda fields, rows: rows[0].update({field: value}))
    with pytest.raises(VERIFIER.VerificationError, match="Choice row"):
        VERIFIER.verify(output)


def test_rehashed_lost_choice_is_detected(audit_fixture):
    _, output = prepare(audit_fixture)
    alter_choices(output, lambda fields, rows: rows.pop())
    with pytest.raises(VERIFIER.VerificationError, match="Lost or fabricated"):
        VERIFIER.verify(output)


def test_rehashed_fabricated_target_or_rt_column_is_detected(audit_fixture):
    _, output = prepare(audit_fixture)
    def add_column(fields, rows):
        fields.append("reaction_time")
        for row in rows:
            row["reaction_time"] = "1.0"
    alter_choices(output, add_column)
    with pytest.raises(VERIFIER.VerificationError, match="Unexpected choice columns"):
        VERIFIER.verify(output)


@pytest.mark.parametrize("field,value", [
    ("event_rows", 9), ("event_files", 2), ("observed_choices", 3),
    ("complete_triplets", 3), ("issues_count", 1), ("source_integrity_passed", False),
    ("complete_triplets", 2.0), ("source_integrity_passed", 1),
])
def test_rehashed_summary_count_corruption_is_detected(audit_fixture, field, value):
    _, output = prepare(audit_fixture)
    path = output / "structure-audit.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    dump(path, report)
    rehash(output)
    with pytest.raises(VERIFIER.VerificationError, match="Summary count/scope"):
        VERIFIER.verify(output)


def test_rehashed_context_conflation_claim_is_rejected(audit_fixture):
    _, output = prepare(audit_fixture)
    path = output / "structure-audit.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["readiness"]["reward_vs_punishment"] = "identical_monetary_reward"
    dump(path, report)
    rehash(output)
    with pytest.raises(VERIFIER.VerificationError, match="Reward/punishment"):
        VERIFIER.verify(output)


def test_source_byte_drift_is_detected_even_if_artifacts_are_unchanged(audit_fixture):
    root, output = prepare(audit_fixture)
    source = next((root / "data/human/stolz_2022_openneuro").rglob("*_events.tsv"))
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(VERIFIER.VerificationError, match="Frozen source differs"):
        VERIFIER.verify(output)


def test_unrehashed_artifact_change_is_detected(audit_fixture):
    _, output = prepare(audit_fixture)
    path = output / "observed_choices.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(VERIFIER.VerificationError, match="Primary artifact hash"):
        VERIFIER.verify(output)


def test_missing_expectation_retains_incomplete_choice_and_issue(audit_fixture):
    root, config, registration, output = audit_fixture
    source_root = root / "data/human/stolz_2022_openneuro"
    path = next(source_root.rglob("*_events.tsv"))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line for line in lines if not line.endswith("\tS53")) + "\n", encoding="utf-8")
    manifest_path = source_root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = next(item for item in manifest["files"] if item["path"].endswith("_events.tsv"))
    item.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    dump(manifest_path, manifest)
    register(root, config, registration)
    run(root, config, registration, output)
    result = VERIFIER.verify(output)
    assert result["observed_choice_rows_verified"] == 2
    assert result["complete_triplets_verified"] == result["incomplete_choices_preserved"] == 1
    assert result["issues_preserved"] == 1
