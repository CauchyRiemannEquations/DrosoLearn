"""Independent replay and rehashed semantic corruption of reward histories."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

from drosolearn import bandit_reversal as producer
from drosolearn.bandit_reversal_audit import verify
from test_bandit_reversal import prepare_fixture


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audited_fixture(tmp_path_factory):
    root, config, registration = prepare_fixture(tmp_path_factory.mktemp("bandit-reversal-audit"))
    output = root / "fixture-output"
    producer.run(root, config, registration, output)
    report = verify(output)
    assert report["passed"], report["errors"]
    return output


def clone(fixture, path):
    root = path / "tampered-study"
    shutil.copytree(fixture.parent, root)
    return root / fixture.name


def rehash(output, relative):
    path = output / "artifact_hashes.json"
    hashes = json.loads(path.read_text(encoding="utf-8"))
    hashes[relative] = hashlib.sha256((output / relative).read_bytes()).hexdigest()
    path.write_text(json.dumps(hashes), encoding="utf-8")


def save_json(output, relative, data):
    (output / relative).write_text(json.dumps(data), encoding="utf-8")
    rehash(output, relative)


def arrays(output, relative):
    with np.load(output / relative, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def save_arrays(output, relative, data):
    np.savez_compressed(output / relative, **data)
    rehash(output, relative)


def rows(output, relative):
    with (output / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_rows(output, relative, data):
    with (output / relative).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)
    rehash(output, relative)


def test_audit_imports_only_previous_independent_auditors_and_standard_libraries():
    tree = ast.parse((ROOT / "src/drosolearn/bandit_reversal_audit.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert {node.module for node in imports} == {"datetime", "pathlib", "encoding_audit", "linearity_audit"}
    assert {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)} == {"json", "csv", "numpy"}
    projection = [node for node in imports if node.module == "linearity_audit"][0]
    assert [alias.name for alias in projection.names] == ["_source_projection"]


def test_complete_independent_replay_preserves_all_primary_bytes(audited_fixture):
    manifest = json.loads((audited_fixture / "artifact_hashes.json").read_text())
    report = verify(audited_fixture)
    assert report["passed"], report["errors"]
    assert manifest == {name: hashlib.sha256((audited_fixture / name).read_bytes()).hexdigest() for name in manifest}
    assert report["counts"]["actual_training_trials"] == 7 * (8 + 3 * 16)
    assert report["counts"]["expected_reference_updates"] == 8 + 3 * 16


@pytest.mark.parametrize("condition", ["linear_normalized", "fixed", "plastic", "expected_linear"])
@pytest.mark.parametrize("field", ["readout_weights", "choice_probability", "decision_activity", "feature_scale", "iid_p_right", "head_updates"])
def test_every_final_state_is_recomputed_after_artifact_rehash(audited_fixture, tmp_path, condition, field):
    output = clone(audited_fixture, tmp_path)
    relative = f"seed_91/{condition}/high_to_low/phase2_0008.npz"
    data = arrays(output, relative)
    data[field].flat[0] += 1 if field == "head_updates" else .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("condition", ["linear_normalized", "fixed", "plastic", "expected_linear"])
@pytest.mark.parametrize("boundary", ["before", "after"])
@pytest.mark.parametrize("field", ["readout_weights", "feature_scale", "head_updates"])
def test_boundary_raw_evidence_is_replayed_after_rehash(audited_fixture, tmp_path, condition, boundary, field):
    output = clone(audited_fixture, tmp_path)
    relative = f"seed_91/{condition}/high_to_low_reset_head/boundary_{boundary}.npz"
    data = arrays(output, relative)
    data[field].flat[0] += 1 if field == "head_updates" else .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("condition", ["linear_normalized", "plastic"])
@pytest.mark.parametrize("field", ["action", "reward", "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2"])
def test_reward_only_choices_and_updates_are_replayed(audited_fixture, tmp_path, condition, field):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_91/trials.csv"
    data = rows(output, relative)
    row = next(row for row in data if row["condition"] == condition and row["history"] == "high_to_low" and
               row["phase"] == "phase2" and row["trial"] == "2")
    row[field] = 1 - int(row[field]) if field in {"action", "reward"} else float(row[field]) + .013
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["target", "expected_factor", "p_right", "readout_step_l2"])
def test_privileged_expected_action_reference_is_independently_enumerated(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_91/expected_updates.csv"
    data = rows(output, relative)
    row = next(row for row in data if row["history"] == "high_to_low" and row["phase"] == "phase2" and row["trial"] == "2")
    row[field] = 1 - int(row[field]) if field == "target" else float(row[field]) + .013
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["action_phase1_uniforms", "action_phase2_uniforms", "train_phase1_nuisance", "train_phase2_nuisance"])
def test_shared_streams_and_initial_normalization_inputs_are_regenerated(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_91/stimuli.npz"
    data = arrays(output, relative)
    data[field].flat[0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


def change(value):
    if value is None:
        return True
    if type(value) is bool:
        return not value
    if isinstance(value, str):
        return value + " changed"
    return value + .013


def test_audit_never_calls_any_producer_learning_data_or_summary(audited_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Producer called by independent audit")
    for name in ("make_splits", "summarize", "state_pack", "expected_factor"):
        monkeypatch.setattr(producer, name, forbidden)
    for learner in (producer.LinearLearner, producer.NormalizedLearner):
        for name in ("features", "update", "probabilities", "snapshot"):
            monkeypatch.setattr(learner, name, forbidden)
    monkeypatch.setattr(producer.ExpectedLinear, "update_expected", forbidden)
    report = verify(audited_fixture)
    assert report["passed"], report["errors"]


@pytest.mark.parametrize("section,field", [
    ("outcomes", "phase1_before"), ("outcomes", "phase1_after"), ("outcomes", "phase1_acquired"),
    ("outcomes", "phase2_before"), ("outcomes", "phase2_after"), ("outcomes", "phase2_gain"),
    ("outcomes", "phase2_auc"), ("outcomes", "phase2_baseline_adjusted_auc"),
    ("outcomes", "phase2_target_passed"), ("outcomes", "qualified_adaptation"),
    ("outcomes", "fresh_acquired"), ("outcomes", "first_observed_75_checkpoint"),
    ("outcomes", "phase2_ood_after"), ("outcomes", "phase2_greedy_after"),
    ("outcomes", "head_updates"), ("outcomes", "internal_updates"), ("outcomes", "algorithm"),
    ("paired_contrasts", "endpoint_cost"), ("paired_contrasts", "raw_auc_cost"),
    ("paired_contrasts", "baseline_adjusted_auc_cost"), ("paired_contrasts", "gain_cost"),
    ("paired_contrasts", "reset_minus_reverse_endpoint"), ("paired_contrasts", "reset_minus_fresh_endpoint"),
    ("paired_contrasts", "repeated_minus_reverse_endpoint"), ("paired_contrasts", "old_rule_acquired"),
    ("paired_contrasts", "fresh_acquired"),
    ("family_verdicts", "n_seeds"), ("family_verdicts", "mean_endpoint_cost"),
    ("family_verdicts", "sd_endpoint_cost"), ("family_verdicts", "positive_seed_count"),
    ("family_verdicts", "high_prefix_acquisition_count"), ("family_verdicts", "fresh_acquisition_count"),
    ("family_verdicts", "criterion_met"),
    ("reference_comparisons", "stochastic_p_target"), ("reference_comparisons", "reference_p_target"),
    ("reference_comparisons", "difference"),
])
def test_every_outcome_contrast_gate_and_reference_comparison_is_recomputed(audited_fixture, tmp_path, section, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    data[section][0][field] = change(data[section][0][field])
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["name", "condition", "mean_endpoint_cost", "positive_seed_count",
                                  "high_prefix_acquisition_count", "fresh_acquisition_count", "criterion_met"])
def test_primary_hypothesis_cannot_be_redefined_in_rehashed_result(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    data["primary_hypothesis"][field] = change(data["primary_hypothesis"][field])
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["phase1_acquired", "fresh_acquired", "phase2_target_passed",
                                  "qualified_adaptation", "first_observed_75_checkpoint"])
def test_expected_reference_cannot_be_counted_as_reward_learner_success(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    row = next(row for row in data["outcomes"] if row["condition"] == "expected_linear")
    assert row[field] is None
    row[field] = True
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["head_change_l2", "internal_change_l2", "feature_scale_before", "feature_scale_after",
                                  "head_updates_before", "head_updates_after", "internal_updates_before", "internal_updates_after",
                                  "reset_head"])
def test_boundary_event_summary_matches_raw_pre_and_post_state(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "boundaries.json").read_text())
    data[0][field] = change(data[0][field])
    save_json(output, "boundaries.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["p_target", "greedy_accuracy", "mean_policy_sensitivity", "internal_change_l2",
                                  "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2",
                                  "probe_decision_change_l2", "lower_bound_fraction", "upper_bound_fraction", "larger"])
def test_curve_metrics_are_calculated_from_replayed_state(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    relative = "learning_curves.csv"
    data = rows(output, relative)
    data[0][field] = 1 - int(data[0][field]) if field == "larger" else float(data[0][field]) + .013
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field,value", [("seeds", [7001]), ("pair", [1, 3]),
                                        ("conditions", ["linear_normalized"]),
                                        ("feature_normalization", "refit_phase2"),
                                        ("expected_reference", "reward_only"), ("head_initialization", "random")])
def test_rehashed_fixture_scope_cannot_alias_planned_seeds_or_changed_design(audited_fixture, tmp_path, field, value):
    output = clone(audited_fixture, tmp_path)
    metadata = json.loads((output / "metadata.json").read_text())
    registration = json.loads((output / "registration.json").read_text())
    config = dict(metadata["config"])
    config[field] = value
    path = output.parent / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    registration["config"] = metadata["config"] = config
    registration["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(output, "registration.json", registration)
    metadata["registration_sha256"] = hashlib.sha256((output / "registration.json").read_bytes()).hexdigest()
    save_json(output, "metadata.json", metadata)
    report = verify(output)
    assert not report["passed"] and report["errors"]


def test_source_drift_is_detected_without_modifying_old_registrations(audited_fixture, tmp_path):
    output = clone(audited_fixture, tmp_path)
    source = output.parent / "src/drosolearn/bandit_reversal.py"
    source.write_bytes(source.read_bytes() + b"\n")
    report = verify(output)
    assert not report["passed"] and report["errors"]
