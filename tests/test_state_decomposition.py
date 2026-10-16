"""Handcrafted static algebra; never initialize or train a learner."""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from drosolearn import state_decomposition as study


REPO = Path(__file__).resolve().parents[1]


def synthetic_config(seeds=(17, 18)):
    cfg = json.loads((REPO / "configs/state_decomposition_v1.json").read_text(encoding="utf8"))
    cfg.update(scope="integration_test", seeds=list(seeds), baseline_trial=8, checkpoints=[0, 4, 8], eval_trials=8)
    cases = len(seeds) * 4 * 4 * 3 * 2 * 8
    cfg.update(cases=cases, crossed_policy_values=4 * cases, selected_input_files=len(seeds) * 53)
    return cfg


def invented_snapshot(linear=False):
    h = np.ones(7 if linear else 2)
    weights = np.empty(0) if linear else np.ones(2)
    encoder = np.empty((0, 7)) if linear else np.array([[1., 0, 0, 0, 0, 0, 0], [2., 0, 0, 0, 0, 0, 0]])
    hidden = 0 if linear else 2
    return {"internal_weights": weights, "readout_weights": h, "encoder": encoder, "bias": np.zeros(hidden),
        "pre": np.arange(hidden, dtype=int), "post": np.arange(hidden, dtype=int),
        "kc_activity": np.zeros((2, 2, hidden)), "mbon_activity": np.zeros((2, 2, hidden)),
        "decision_activity": np.zeros((2, 2, len(h))), "choice_probability": np.full(2, .5),
        "feature_scale": np.asarray(1.), "initial_feature_gap_rms": np.asarray(1.),
        "head_updates": np.asarray(8), "internal_updates": np.asarray(0 if linear else 8),
        "initial_internal_weights": np.full(hidden, 3.), "internal_upper_bounds": np.full(hidden, 6.),
        "incoming_normalizer": np.empty(0) if linear else np.ones(len(h)),
        "iid_p_right": np.full(2, .5), "ood_p_right": np.full(2, .5),
        "phase_head_updates": np.asarray(0), "phase_internal_updates": np.asarray(0)}


def invented_inputs():
    counts = np.array([[2, 4], [4, 2]], dtype=np.int64)
    inputs = np.concatenate((counts[..., None] / 8., np.zeros((2, 2, 6))), axis=-1)
    return inputs, counts


def test_production_config_validates_without_loading_states_or_constructing_anything(monkeypatch):
    cfg = json.loads((REPO / "configs/state_decomposition_v1.json").read_text(encoding="utf8"))
    before = deepcopy(cfg)
    monkeypatch.setattr(study, "load_npz", lambda *a, **k: pytest.fail("Validation read scientific states"))
    study.validate_config(cfg)
    study.validate_config(synthetic_config())
    assert cfg == before


@pytest.mark.parametrize("field,value", [
    ("scope", "retuned"), ("seeds", [12001]), ("seeds", [17, 17]), ("seeds", [True]),
    ("seeds", [19]), ("eval_trials", 128), ("baseline_trial", 400), ("checkpoints", [0, 8]),
    ("models", ["real"]), ("branches", ["full"]), ("pair", [1, 4]),
    ("raw_residual_tolerance", 1e-6), ("policy_match_tolerance", 1e-3),
    ("learning_updates", 1), ("learner_constructions", 1), ("dataset_generations", 1), ("rng_draws", 1),
    ("cases", 1), ("crossed_policy_values", 1), ("selected_input_files", 1),
])
def test_fixed_static_scope_and_full_original_coverage_are_not_silently_changed(field, value):
    cfg = synthetic_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_full_four_state_identity_has_hand_calculated_nonzero_bilinear_terms():
    inputs, counts = invented_inputs()
    baseline, current = invented_snapshot(), invented_snapshot()
    current["readout_weights"][:] = [2., 3.]
    current["internal_weights"][:] = [4., 5.]
    out = study.decompose(inputs, counts, baseline, current)
    assert out["raw_gap_00"][0] == pytest.approx(.75)
    assert out["raw_gap_10"][0] == pytest.approx(2.)
    assert out["raw_gap_01"][0] == pytest.approx(3.5)
    assert out["raw_gap_11"][0] == pytest.approx(9.5)
    assert out["head_term"][0] == pytest.approx(1.25)
    assert out["internal_term"][0] == pytest.approx(2.75)
    assert out["cross_term"][0] == pytest.approx(4.75)
    assert out["total_raw_gap_change"][0] == pytest.approx(8.75)
    assert np.max(np.abs(out["bilinear_residual"])) == 0.
    np.testing.assert_array_equal(out["raw_interaction"], out["cross_term"])


def test_smaller_target_alignment_prevents_left_right_mean_cancellation():
    inputs, counts = invented_inputs()
    baseline, current = invented_snapshot(), invented_snapshot()
    current["readout_weights"][:] = [2., 3.]
    out = study.decompose(inputs, counts, baseline, current)
    np.testing.assert_array_equal(out["target_sign"], [-1, 1])
    assert float(np.mean(out["head_term"])) == 0.
    assert float(np.mean(out["aligned_head_term"])) == pytest.approx(-1.25)
    for cross in study.CROSSES:
        assert out["p_rule_" + cross][0] == 1 - out["p_right_" + cross][0]
        assert out["p_rule_" + cross][1] == out["p_right_" + cross][1]


def test_policy_interaction_can_be_nonzero_when_raw_bilinear_cross_term_is_zero():
    inputs, counts = invented_inputs()
    baseline, current = invented_snapshot(), invented_snapshot()
    current["readout_weights"][:] = [2., 1.]
    current["internal_weights"][:] = [1., 2.]
    out = study.decompose(inputs, counts, baseline, current)
    np.testing.assert_array_equal(out["cross_term"], [0., 0.])
    np.testing.assert_array_equal(out["raw_interaction"], [0., 0.])
    assert np.max(np.abs(out["policy_interaction_right"])) > 1e-4
    assert np.max(np.abs(out["policy_interaction_rule"])) > 1e-4


def test_clipping_boundary_40_is_unclipped_but_41_is_clipped_and_probability_can_stay_constant():
    inputs, counts = invented_inputs()
    baseline, current = invented_snapshot(True), invented_snapshot(True)
    baseline["readout_weights"][:] = 0
    current["readout_weights"][:] = 0
    baseline["readout_weights"][0], current["readout_weights"][0] = 160., 164.
    out = study.decompose(inputs, counts, baseline, current)
    np.testing.assert_array_equal(out["clipped_00"], [0, 0])
    np.testing.assert_array_equal(out["clipped_11"], [1, 1])
    np.testing.assert_array_equal(out["policy_rule_change"], [0., 0.])
    assert np.max(np.abs(out["total_raw_gap_change"])) == 1.


@pytest.mark.parametrize("fixed_field", study.FIXED_FIELDS)
def test_changing_encoder_topology_bounds_or_feature_scale_invalidates_crossed_state_algebra(fixed_field):
    inputs, counts = invented_inputs()
    baseline, current = invented_snapshot(), invented_snapshot()
    current[fixed_field] = current[fixed_field].copy()
    current[fixed_field].flat[0] += 1
    with pytest.raises(ValueError):
        study.decompose(inputs, counts, baseline, current)


@pytest.mark.parametrize("model_kind", ["linear", "network"])
def test_frozen_and_initial_states_have_exactly_zero_changes_and_preserve_inputs(model_kind):
    inputs, counts = invented_inputs()
    baseline = invented_snapshot(model_kind == "linear")
    before = deepcopy(baseline)
    original_inputs = inputs.copy()
    out = study.decompose(inputs, counts, baseline, deepcopy(baseline))
    for field in ("head_term", "internal_term", "cross_term", "total_raw_gap_change", "policy_rule_change"):
        np.testing.assert_array_equal(out[field], [0., 0.])
    np.testing.assert_array_equal(inputs, original_inputs)
    for field in baseline:
        np.testing.assert_array_equal(baseline[field], before[field])


@pytest.mark.parametrize("branch", ["head_only", "internal_only"])
def test_inactive_parameter_component_has_zero_term_and_zero_cross_term(branch):
    inputs, counts = invented_inputs()
    baseline, current = invented_snapshot(), invented_snapshot()
    current["readout_weights" if branch == "head_only" else "internal_weights"][:] += .5
    out = study.decompose(inputs, counts, baseline, current)
    np.testing.assert_array_equal(out["internal_term" if branch == "head_only" else "head_term"], [0., 0.])
    np.testing.assert_array_equal(out["cross_term"], [0., 0.])


def test_linear_internal_and_cross_terms_are_zero_for_every_readout_change():
    inputs, counts = invented_inputs()
    baseline, current = invented_snapshot(True), invented_snapshot(True)
    current["readout_weights"] += np.arange(7)
    out = study.decompose(inputs, counts, baseline, current)
    np.testing.assert_array_equal(out["internal_term"], [0., 0.])
    np.testing.assert_array_equal(out["cross_term"], [0., 0.])
    np.testing.assert_array_equal(out["raw_gap_00"], out["raw_gap_01"])
    np.testing.assert_array_equal(out["raw_gap_10"], out["raw_gap_11"])


def test_runtime_contains_no_learner_dataset_generator_or_rng_calls():
    tree = ast.parse(inspect.getsource(study))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    assert not set(names) & {"LinearLearner", "NormalizedLearner", "Learner", "make_dataset", "default_rng", "selected_update", "random"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update"
                   and isinstance(node.func.value, ast.Name) and node.func.value.id in {"model", "learner"} for node in ast.walk(tree))


def test_duplicate_and_nonfinite_json_are_rejected(tmp_path):
    path = tmp_path / "invented.json"
    for text in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(text, encoding="utf8")
        with pytest.raises(ValueError):
            study.read_json(path)


def test_summary_requires_all_original_cells_and_keeps_seed_sd_separate_from_case_counts():
    cfg, cells = synthetic_config(), []
    inputs, counts = invented_inputs()
    # Eight invented rows match the tiny evaluation budget; no dataset generator.
    inputs, counts = np.tile(inputs, (4, 1, 1)), np.tile(counts, (4, 1))
    values = study.decompose(inputs, counts, invented_snapshot(), invented_snapshot())
    for seed in cfg["seeds"]:
        for model in study.MODELS:
            for branch in study.BRANCHES:
                for trial in cfg["checkpoints"]:
                    for split in ("iid", "ood"):
                        cells.append(study.cell_summary(seed, model, branch, trial, split, values))
    summary = study.summarize(cfg, cells)
    assert summary["cases"] == 1536 and summary["crossed_policy_values"] == 6144
    assert summary["behavior_hypotheses_tested"] == summary["learning_updates"] == 0
    assert summary["arithmetic_identity_passed"] is True
    assert all(row["cases"] == 16 and row["n_seeds"] == 2 for row in summary["cells"])
    assert all(row["statistics"]["aligned_cross_term"]["sample_sd_across_seed_means"] == 0. for row in summary["cells"])
    with pytest.raises(ValueError):
        study.summarize(cfg, cells[:-1])
