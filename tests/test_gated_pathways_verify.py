"""Independent common-time arithmetic on handwritten saved tensors only."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gated_pathways_independent_tests", ROOT / "scripts/verify_gated_pathways.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
FIXTURE_SPEC = importlib.util.spec_from_file_location("gated_pathways_handwritten_fixture", ROOT / "tests/test_gated_pathways.py")
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


@pytest.fixture(scope="module")
def independent_fixture():
    cfg, parent = fixtures.fixture_data()
    arrays = audit.expected_artifacts(cfg, parent)
    rows = audit.expected_rows(cfg, arrays, parent)
    return cfg, parent, arrays, rows


def test_complete_current_producer_matches_independent_handwritten_fixture(independent_fixture):
    from drosolearn.gated_pathways import analyze, summarize_arrays
    cfg, parent, wanted, wanted_rows = independent_fixture
    arrays, rows = analyze(cfg, parent)
    audit.check_artifacts(arrays, wanted, rows, wanted_rows)
    assert len(arrays) == 12 and len(rows) == 180 and len(rows[0]) == 24
    summary = summarize_arrays(cfg, arrays, rows, parent)
    receipt = audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert receipt["parent_trial_records_checked"] == 240
    assert receipt["destination_effects_checked"] == 960
    assert len(summary["families"]) == 30 and len(summary["contrasts"]) == 24


def test_one_reference_time_split_is_used_even_when_other_method_knows_more_or_less(independent_fixture):
    _, _, arrays, rows = independent_fixture
    assert arrays["first_full_trials"][0, 0, :, 0].tolist() == [9, 8, 6, 4, 6]
    assert arrays["first_gated_balanced_trials"][0, 0, 0] == 7
    assert arrays["gated_phase_before"][0, 0, 0].tolist() == [False] * 6 + [True] * 2
    assert arrays["own_full_before"][0, 0, 3, 0].tolist() == [False] * 4 + [True] * 4
    assert not arrays["own_full_before"][0, 0, 1, 0].any()
    selected = [row for row in rows if row["topology"] == "real" and row["representation"] == "raw"]
    assert all(row["trial_count"] == (6 if row["phase"] == "pre" else 2) for row in selected)
    assert any(row["method"] == "isolated" and row["phase"] == "post" and row["category"] == "unknown_other"
               and row["exposures"] > 0 for row in selected)


def test_last_discovery_stays_pre_and_source_never_enters_unknown_other(independent_fixture):
    _, _, arrays, _ = independent_fixture
    assert not arrays["gated_phase_before"][0, 0, 0, 5]
    assert arrays["gated_phase_before"][0, 0, 0, 6]
    for si, order in enumerate(arrays["schedules"]):
        for t, source in enumerate(order): assert np.all(arrays["effect_categories"][..., si, t, source] == 0)


def test_cost_is_counted_once_per_phase_and_repeated_only_for_category_display(independent_fixture):
    cfg, parent, arrays, rows = independent_fixture
    summary = audit.expected_summary(cfg, arrays, rows, parent)
    for family in summary["families"]:
        for record in family["seeds"]:
            assert sum(cost["trial_count"] for cost in record["phase_costs"]) == cfg["training_trials"]
            assert sum(cost["endpoint_contribution"] for cost in record["phase_costs"]) == pytest.approx(record["reconstructed_accuracy_gain"])
    selected = [row for row in rows if (row["topology"], row["representation"], row["method"], row["phase"]) == ("real", "raw", "gated_balanced", "pre")]
    assert len({row["head_path_l2"] for row in selected}) == 1
    assert sum(row["trial_count"] for row in selected) == 3 * selected[0]["trial_count"]


def test_zero_cost_and_never_opening_gate_preserve_empty_post_records_and_null_ratios():
    from drosolearn.gated_pathways import analyze, summarize_arrays
    cfg, parent = fixtures.fixture_data(gate_never=True)
    arrays, rows = analyze(cfg, parent)
    summary = summarize_arrays(cfg, arrays, rows, parent)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert not arrays["gated_phase_before"].any()
    assert np.all(arrays["first_gated_balanced_trials"] == 9)
    post = [row for row in rows if row["phase"] == "post"]
    assert len(post) == 90
    assert all(row["trial_count"] == row["exposures"] == row["head_path_l2"] == row["go_trials"] == 0 for row in post)
    assert all(row["gain_per_Go"] is None and row["gain_per_path"] is None for row in post)
    assert all(family["mean_phase_costs"][1]["gain_per_Go"] is None for family in summary["families"])


def test_null_csv_values_are_empty_and_defined_ratios_are_not_zero_filled(independent_fixture):
    _, _, arrays, rows = independent_fixture
    serialized = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    audit.check_artifacts(arrays, arrays, serialized, rows)
    position = next(i for i, row in enumerate(rows) if row["gain_per_Go"] is None)
    serialized[position]["gain_per_Go"] = "0"
    with pytest.raises(ValueError): audit.check_artifacts(arrays, arrays, serialized, rows)


@pytest.mark.parametrize("field", audit.FIELDS)
def test_every_derived_array_is_bound_to_independent_saved_path_arithmetic(independent_fixture, field):
    _, _, original, rows = independent_fixture
    arrays = copy.deepcopy(original)
    flat = arrays[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + (1 if flat.dtype.kind in "iu" else .03)
    with pytest.raises(ValueError): audit.check_artifacts(arrays, original, rows, rows)


@pytest.mark.parametrize("field", ["trial_update_factors", "trial_directions", "trial_head_steps", "trial_head_step_squared_l2",
    "trial_cumulative_head_path_l2", "trial_cumulative_head_squared_l2", "checkpoint_heads", "checkpoint_probabilities",
    "trial_correct_probability_delta", "trial_aligned_margin_delta", "first_balanced_mode_trials", "model_seeds", "targets", "schedules"])
def test_parent_effect_cost_or_reference_binding_corruption_is_rejected(independent_fixture, field):
    cfg, original, _, _ = independent_fixture
    parent = copy.deepcopy(original)
    flat = parent[field].reshape(-1)
    flat[-1] += 1 if flat.dtype.kind in "iu" else .03
    with pytest.raises(ValueError): audit.expected_artifacts(cfg, parent)


@pytest.mark.parametrize("field", ["trial_count", "exposures", "positive_probability_effects", "negative_probability_effects",
    "probability_positive_mass", "probability_negative_mass", "endpoint_contribution", "go_trials", "nonzero_head_steps", "head_path_l2", "head_squared_l2"])
def test_measurement_counts_masses_and_costs_cannot_be_changed(independent_fixture, field):
    _, _, arrays, original = independent_fixture
    rows = copy.deepcopy(original)
    rows[0][field] += 1
    with pytest.raises(ValueError): audit.check_artifacts(arrays, arrays, rows, original)


@pytest.mark.parametrize("key,value", [("phase_definition", "each_methods_own_gate"), ("reference_method", "balanced"),
    ("category_definition", "reference_method_observations_only"), ("numeric_tolerance", .001),
    ("contrast_controls", ["balanced"]), ("new_confirmatory_hypothesis", True)])
def test_fixed_common_phase_scientific_choices_cannot_be_relaxed(key, value):
    cfg, _ = fixtures.fixture_data()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_fixture_cannot_enter_actual_registered_analysis():
    cfg, _ = fixtures.fixture_data()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17601]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_full_known_discovery_difference_is_not_a_count_of_full_state_trials(independent_fixture):
    cfg, parent, arrays, rows = independent_fixture
    summary = audit.expected_summary(cfg, arrays, rows, parent)
    selected = {record["control"]: record for record in summary["contrasts"] if record["topology"] == "real" and record["representation"] == "raw"}
    assert selected["balanced"]["seeds"][0]["own_full_trial_difference"] == 2
    assert selected["isolated"]["seeds"][0]["own_full_trial_difference"] == -2


def test_tiny_help_and_harm_are_separated_without_erasing_signs_by_numeric_tolerance(independent_fixture):
    from drosolearn.gated_pathways import analyze, summarize_arrays
    cfg, original, _, _ = independent_fixture
    parent = copy.deepcopy(original)
    for ti in range(3):
        for ri in range(2):
            at = (ti, ri, 3, 0)
            parent["trial_directions"][at + (0,)] = np.array([1., 0., 1., 0.]) / np.sqrt(2.)
            parent["trial_update_factors"][at + (0,)] = 1e-10
            steps = parent["trial_update_factors"][at][:, None] * parent["trial_directions"][at]
            heads = np.concatenate((np.zeros((1, 4)), np.cumsum(steps, axis=0)))
            lengths = np.linalg.norm(steps, axis=-1)
            parent["trial_head_steps"][at] = lengths
            parent["trial_head_step_squared_l2"][at] = lengths ** 2
            parent["trial_cumulative_head_path_l2"][at] = np.cumsum(lengths)
            parent["trial_cumulative_head_squared_l2"][at] = np.cumsum(lengths ** 2)
            parent["trial_correct_probability_delta"][at] = audit.SIGNS * (audit.sigmoid(heads[1:]) - audit.sigmoid(heads[:-1]))
            parent["trial_aligned_margin_delta"][at] = audit.SIGNS * (heads[1:] - heads[:-1])
            parent["checkpoint_heads"][at] = heads[parent["checkpoints"]]
            parent["checkpoint_probabilities"][at] = audit.sigmoid(parent["checkpoint_heads"][at])
    arrays, rows = analyze(cfg, parent)
    summary = summarize_arrays(cfg, arrays, rows, parent)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    first = arrays["probability_changes"][0, 0, 3, 0, 0]
    assert 0. < first[0] < cfg["numeric_tolerance"]
    assert -cfg["numeric_tolerance"] < first[2] < 0.
    selected = next(row for row in rows if (row["topology"], row["representation"], row["method"], row["phase"], row["category"])
                    == ("real", "raw", "balanced", "pre", "unknown_other"))
    assert selected["negative_probability_effects"] > 0 and selected["probability_negative_mass"] > 0.


@pytest.mark.parametrize("target", ["parent_verdict", "phase_mean_cost", "contrast_gain", "new_training"])
def test_summary_cannot_hide_parent_failures_or_change_accounting(independent_fixture, target):
    cfg, parent, arrays, rows = independent_fixture
    summary = audit.expected_summary(cfg, arrays, rows, parent)
    if target == "parent_verdict": summary["interpretation"]["parent_H_gain"] = True
    elif target == "phase_mean_cost": summary["families"][0]["mean_phase_costs"][0]["head_path_l2"] += .1
    elif target == "contrast_gain": summary["contrasts"][0]["means"]["endpoint_difference"] += .1
    else: summary["counts"]["new_training_trials"] = 1
    with pytest.raises(ValueError): audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)


def test_no_fresh_rng_actor_graph_solver_or_counterfactual_calculation(independent_fixture, monkeypatch):
    from drosolearn.gated_pathways import analyze, summarize_arrays
    def forbidden(*args, **kwargs): raise AssertionError("Saved-path arithmetic must not draw, construct or optimize")
    for name in ("default_rng", "SeedSequence", "random", "rand"): monkeypatch.setattr(np.random, name, forbidden)
    for name in ("svd", "pinv", "lstsq"): monkeypatch.setattr(np.linalg, name, forbidden)
    cfg, parent, _, _ = independent_fixture
    arrays, rows = analyze(cfg, parent)
    summary = summarize_arrays(cfg, arrays, rows, parent)
    audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert summary["interpretation"]["parent_H_protection"]
    assert not summary["interpretation"]["parent_H_gain"] and not summary["interpretation"]["parent_H_acquisition"]
