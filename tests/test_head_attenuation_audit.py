"""Independent reconstruction and rehashed semantic corruption tests."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

from drosolearn import head_attenuation as producer
from drosolearn.head_attenuation_audit import _raw_decisions, _probability_decisions, verify
from test_head_attenuation import prepare_fixture


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audited_fixture(tmp_path_factory):
    root, config, registration = prepare_fixture(tmp_path_factory.mktemp("head-attenuation-audit"))
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
        writer.writeheader()
        writer.writerows(data)
    rehash(output, relative)


def change(value):
    if value is None:
        return True
    if type(value) is bool:
        return not value
    if isinstance(value, str):
        return value + " changed"
    return value + .013


def test_audit_imports_only_previous_independent_auditors_and_standard_libraries():
    tree = ast.parse((ROOT / "src/drosolearn/head_attenuation_audit.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert {node.module for node in imports} == {
        "datetime", "pathlib", "encoding_audit", "linearity_audit", "bandit_reversal_audit"}
    assert {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)} == {"json", "numpy"}


def test_independent_rank_audit_distinguishes_exact_raw_ties_from_probability_tolerance():
    margins = np.array([-1e-15, 0., 1e-15, -1., 1.])
    probability = 1 / (1 + np.exp(-margins))
    assert np.array_equal(_raw_decisions(margins), np.array([0., .5, 1., 0., 1.]))
    assert np.array_equal(_probability_decisions(probability), np.array([.5, .5, .5, 0., 1.]))


def test_complete_independent_replay_preserves_primary_bytes_and_counts_one_physical_prefix(audited_fixture):
    manifest = json.loads((audited_fixture / "artifact_hashes.json").read_text())
    report = verify(audited_fixture)
    assert report["passed"], report["errors"]
    assert manifest == {name: hashlib.sha256((audited_fixture / name).read_bytes()).hexdigest() for name in manifest}
    assert report["counts"]["actual_training_trials"] == 7 * 6 * 8
    assert report["counts"]["expected_reference_updates"] == 6 * 8
    assert report["counts"]["actual_shared_prefixes"] == 7
    assert report["counts"]["checkpoints"] == 8 * 6 * 3
    assert report["counts"]["boundary_state_files"] == 8 * 5 * 2
    assert len(report["boundary_diagnostics"]) == 8 * 5 * 2


def test_second_engineering_seed_also_has_complete_independent_replay(tmp_path):
    root, config, registration = prepare_fixture(tmp_path, seeds=[102])
    output = root / "fixture-output"
    producer.run(root, config, registration, output)
    report = verify(output)
    assert report["passed"], report["errors"]
    assert report["counts"]["actual_training_trials"] == 7 * 6 * 8
    assert report["counts"]["expected_reference_updates"] == 6 * 8


@pytest.mark.parametrize("condition", [*producer.FAMILIES, "expected_linear"])
def test_every_family_and_privileged_reference_final_state_is_independently_replayed(audited_fixture, tmp_path, condition):
    output = clone(audited_fixture, tmp_path)
    relative = f"seed_101/{condition}/gamma_025/phase2_0008.npz"
    data = arrays(output, relative)
    data["readout_weights"].flat[0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["readout_weights", "choice_probability", "decision_activity", "feature_scale",
                                  "iid_p_right", "ood_raw_margin", "head_updates", "internal_updates",
                                  "initial_internal_weights", "internal_upper_bounds", "incoming_normalizer"])
def test_every_reconstructed_and_frozen_state_field_rejects_rehashed_corruption(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_101/plastic/gamma_025/phase2_0008.npz"
    data = arrays(output, relative)
    data[field].flat[0] += 1 if "updates" in field else .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("boundary", ["before", "after"])
@pytest.mark.parametrize("field", ["readout_weights", "internal_weights", "feature_scale", "head_updates",
                                  "initial_internal_weights", "internal_upper_bounds", "incoming_normalizer",
                                  "iid_p_right", "ood_raw_margin"])
def test_boundary_head_only_clone_and_raw_case_evidence_are_reconstructed(audited_fixture, tmp_path, boundary, field):
    output = clone(audited_fixture, tmp_path)
    relative = f"seed_101/fixed/gamma_025/boundary_{boundary}.npz"
    data = arrays(output, relative)
    data[field].flat[0] += 1 if field == "head_updates" else .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("condition", ["linear_normalized", "plastic", "expected_linear"])
def test_all_branches_must_reference_the_replayed_single_shared_prefix(audited_fixture, tmp_path, condition):
    output = clone(audited_fixture, tmp_path)
    relative = f"seed_101/{condition}/shared_prefix/phase1_0008.npz"
    data = arrays(output, relative)
    data["readout_weights"].flat[0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("condition", ["linear_normalized", "plastic"])
@pytest.mark.parametrize("field", ["action", "reward", "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2"])
def test_reward_only_choices_old_head_internal_gradient_and_all_updates_are_replayed(audited_fixture, tmp_path, condition, field):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_101/trials.csv"
    data = rows(output, relative)
    row = next(row for row in data if row["condition"] == condition and row["history"] == "gamma_025" and row["trial"] == "2")
    row[field] = 1 - int(row[field]) if field in {"action", "reward"} else float(row[field]) + .013
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["target", "expected_factor", "p_right", "readout_step_l2"])
def test_reference_uses_independent_two_action_enumeration(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_101/expected_updates.csv"
    data = rows(output, relative)
    row = next(row for row in data if row["history"] == "gamma_025" and row["trial"] == "2")
    row[field] = 1 - int(row[field]) if field == "target" else float(row[field]) + .013
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["action_phase1_uniforms", "action_phase2_uniforms", "train_phase1_nuisance", "train_phase2_nuisance"])
def test_shared_streams_and_train_only_scale_inputs_are_independently_regenerated(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_101/stimuli.npz"
    data = arrays(output, relative)
    data[field].flat[0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("kind", ["real", "rewired", "ann"])
def test_topology_is_independently_reconstructed_from_source_and_registered_rng(audited_fixture, tmp_path, kind):
    output = clone(audited_fixture, tmp_path)
    relative = f"seed_101/{kind}_topology.npz"
    data = arrays(output, relative)
    data["counts"][0] += 1
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("mutation", ["duplicate_prefix", "missing_prefix_update", "missing_branch_update"])
def test_physical_coverage_cannot_duplicate_or_omit_prefix_or_branch_updates(audited_fixture, tmp_path, mutation):
    output = clone(audited_fixture, tmp_path)
    relative = "seed_101/trials.csv"
    data = rows(output, relative)
    history = "shared_prefix" if mutation != "missing_branch_update" else "gamma_025"
    index = next(index for index, row in enumerate(data) if row["history"] == history)
    if mutation == "duplicate_prefix":
        data.append(dict(data[index]))
    else:
        data.pop(index)
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


def test_audit_calls_no_producer_learning_data_clone_metric_or_summary(audited_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Producer called by independent audit")
    for name in ("make_splits", "summarize", "state_pack", "expected_factor", "clone_scaled", "raw_margin", "probe_metrics", "normalized_auc"):
        monkeypatch.setattr(producer, name, forbidden)
    for learner in (producer.LinearLearner, producer.NormalizedLearner):
        for name in ("features", "update", "probabilities", "snapshot"):
            monkeypatch.setattr(learner, name, forbidden)
    monkeypatch.setattr(producer.ExpectedLinear, "update_expected", forbidden)
    report = verify(audited_fixture)
    assert report["passed"], report["errors"]


@pytest.mark.parametrize("section,field", [
    ("prefix_outcomes", "before"), ("prefix_outcomes", "after"), ("prefix_outcomes", "gain"),
    ("prefix_outcomes", "acquired"), ("prefix_outcomes", "physical_training_updates"),
    ("outcomes", "gamma"), ("outcomes", "phase1_before"), ("outcomes", "phase1_after"),
    ("outcomes", "phase1_acquired"), ("outcomes", "phase2_before"), ("outcomes", "phase2_after"),
    ("outcomes", "phase2_gain"), ("outcomes", "phase2_auc"), ("outcomes", "phase2_baseline_adjusted_auc"),
    ("outcomes", "frozen_no_learning_endpoint"), ("outcomes", "frozen_no_learning_auc"),
    ("outcomes", "phase2_target_passed"), ("outcomes", "qualified_adaptation"), ("outcomes", "fresh_acquired"),
    ("outcomes", "first_observed_75_checkpoint"), ("outcomes", "phase2_ood_after"),
    ("outcomes", "phase2_greedy_after"), ("outcomes", "head_updates"), ("outcomes", "internal_updates"),
    ("outcomes", "physical_phase2_updates"), ("outcomes", "algorithm"),
    ("paired_contrasts", "endpoint_benefit"), ("paired_contrasts", "initial_probability_difference"),
    ("paired_contrasts", "gain_difference"), ("paired_contrasts", "raw_auc_difference"),
    ("paired_contrasts", "baseline_adjusted_auc_difference"), ("paired_contrasts", "reset_minus_full_endpoint"),
    ("paired_contrasts", "reset_minus_fresh_endpoint"), ("paired_contrasts", "half_minus_full_endpoint"),
    ("paired_contrasts", "old_rule_acquired"), ("paired_contrasts", "fresh_acquired"),
    ("family_verdicts", "n_seeds"), ("family_verdicts", "mean_endpoint_benefit"),
    ("family_verdicts", "sd_endpoint_benefit"), ("family_verdicts", "positive_seed_count"),
    ("family_verdicts", "high_prefix_acquisition_count"), ("family_verdicts", "fresh_acquisition_count"),
    ("family_verdicts", "criterion_met"),
    ("reference_comparisons", "stochastic_p_target"), ("reference_comparisons", "reference_p_target"),
    ("reference_comparisons", "difference"),
])
def test_every_summary_outcome_contrast_gate_and_reference_comparison_is_recomputed(audited_fixture, tmp_path, section, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    data[section][0][field] = change(data[section][0][field])
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["name", "condition", "n_seeds", "mean_endpoint_benefit", "positive_seed_count",
                                  "high_prefix_acquisition_count", "fresh_acquisition_count", "criterion_met"])
def test_primary_hypothesis_cannot_be_redefined_in_rehashed_results(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    data["primary_hypothesis"][field] = change(data["primary_hypothesis"][field])
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["phase1_acquired", "fresh_acquired", "phase2_target_passed", "qualified_adaptation", "first_observed_75_checkpoint"])
def test_privileged_reference_has_no_hypothesis_success_flags(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    row = next(row for row in data["outcomes"] if row["condition"] == "expected_linear" and row["history"] == "gamma_025")
    assert row[field] is None
    row[field] = True
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["gamma", "shared_prefix_path", "shared_prefix_sha256", "head_change_l2",
                                  "internal_change_l2", "feature_scale_before", "feature_scale_after",
                                  "head_updates_before", "head_updates_after", "internal_updates_before", "internal_updates_after"])
def test_boundary_event_summary_matches_raw_state_and_shared_prefix_hash(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    data = json.loads((output / "boundaries.json").read_text())
    row = next(row for row in data if row["history"] == "gamma_025")
    row[field] = change(row[field])
    save_json(output, "boundaries.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["p_target", "greedy_accuracy", "mean_policy_sensitivity", "internal_change_l2",
                                  "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2",
                                  "probe_decision_change_l2", "lower_bound_fraction", "upper_bound_fraction", "larger"])
def test_all_curve_metrics_are_derived_from_replayed_state(audited_fixture, tmp_path, field):
    output = clone(audited_fixture, tmp_path)
    relative = "learning_curves.csv"
    data = rows(output, relative)
    data[0][field] = 1 - int(data[0][field]) if field == "larger" else float(data[0][field]) + .013
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field,value", [("seeds", [8001]), ("pair", [1, 3]),
                                        ("conditions", ["linear_normalized"]), ("gammas", [0, .2, .5, 1]),
                                        ("feature_normalization", "refit_phase2"), ("primary_gamma", .5),
                                        ("attenuation_mode", "reset_internal"), ("expected_reference", "reward_only")])
def test_rehashed_fixture_config_cannot_alias_scientific_seeds_or_change_design(audited_fixture, tmp_path, field, value):
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


def test_source_drift_is_detected_without_changing_any_previous_registration(audited_fixture, tmp_path):
    output = clone(audited_fixture, tmp_path)
    source = output.parent / "src/drosolearn/head_attenuation.py"
    source.write_bytes(source.read_bytes() + b"\n")
    report = verify(output)
    assert not report["passed"] and report["errors"]
