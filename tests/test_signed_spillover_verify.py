"""Invented saved trajectories only; no stored scientific outcomes are read."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "signed_spillover_independent_tests",
    Path(__file__).resolve().parents[1] / "scripts/verify_signed_spillover.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def saved_fixture():
    """Make short hand-set observed actions; no random draw or model import."""
    cfg = {"study_id": "signed_spillover_v1", "scope": "integration_test",
           "module": "drosolearn.signed_spillover", "protocol": "docs/signed-spillover-protocol.md",
           "source_result": "results/spillover_tradeoff_v1", "topologies": list(audit.TOPOLOGIES),
           "representations": list(audit.REPRESENTATIONS), "alphas": list(audit.ALPHAS),
           "model_seeds": [90091, 90092], "training_trials": 8, "checkpoints": [0, 4, 8],
           "head_lr": .1, "numeric_tolerance": 1e-8, "neutral_tolerance": 1e-8,
           "labels": [1, 1, -1, -1], "categories": list(audit.CATEGORY_NAMES), "external_manifest": None}
    shape = (3, 2, 5, 2, 8)
    arrays = {key: np.zeros(shape, int if key in ("trial_actions", "trial_rewards") else float)
              for key in ("trial_actions", "trial_rewards", "trial_probabilities", "trial_update_factors",
                          "trial_own_score_steps", "trial_other_score_steps", "trial_head_steps",
                          "trial_internal_update_calls", "trial_value_update_calls", "trial_internal_steps", "trial_value_steps")}
    arrays.update({"model_seeds": np.asarray(cfg["model_seeds"]),
                   "checkpoints": np.asarray(cfg["checkpoints"]), "alphas": np.asarray(audit.ALPHAS),
                   "targets": np.array([1, 1, 0, 0]),
                   "schedules": np.tile(np.asarray([0, 1, 2, 3, 3, 2, 1, 0]), (2, 1))})
    cshape = (3, 2, 5, 2, 3, 4)
    for key in ("checkpoint_heads", "checkpoint_logits", "checkpoint_probabilities"):
        arrays[key] = np.zeros(cshape)
    rows = []
    for ti, topology in enumerate(audit.TOPOLOGIES):
        for ri, representation in enumerate(audit.REPRESENTATIONS):
            d = np.eye(4)
            asymmetric = np.asarray([[1., 1., 1., 1.], [1., 2., -1., 1.],
                                     [2., 1., 1., 1.], [1., 2., 1., 3.]])
            asymmetric /= np.linalg.norm(asymmetric, axis=-1, keepdims=True)
            w = np.asarray([(1-a) * asymmetric + a * np.eye(4) for a in audit.ALPHAS])
            w /= np.linalg.norm(w, axis=-1, keepdims=True)
            prefix = "t_" + topology + "_" + representation + "_"
            arrays[prefix + "base_budget_base_geometry_unit_rows"] = np.stack([d, d])
            arrays[prefix + "directions"] = np.stack([w, w])
            arrays[prefix + "influences"] = np.stack([w @ d.T, w @ d.T])
            for si, seed in enumerate(cfg["model_seeds"]):
                for ai, alpha in enumerate(audit.ALPHAS):
                    head, path, square = np.zeros(4), 0., 0.
                    def checkpoint(k):
                        at = (ti, ri, ai, si, k)
                        arrays["checkpoint_heads"][at] = head
                        arrays["checkpoint_logits"][at] = d @ head
                        arrays["checkpoint_probabilities"][at] = audit.sigmoid(d @ head)
                    checkpoint(0)
                    for trial in range(8):
                        source = int(arrays["schedules"][si, trial])
                        at = (ti, ri, ai, si, trial)
                        p = float(audit.sigmoid((d @ head)[source]))
                        action = int((trial + si) % 3 != 1)
                        reward = action * int(audit.SIGNS[source])
                        g = .1 * (action - p) * reward
                        arrays["trial_actions"][at], arrays["trial_rewards"][at] = action, reward
                        arrays["trial_probabilities"][at], arrays["trial_update_factors"][at] = p, g
                        new = head + g * w[ai, source]
                        diff = new - head
                        step = float(np.linalg.norm(diff))
                        arrays["trial_head_steps"][at] = step
                        arrays["trial_own_score_steps"][at] = float((d @ diff)[source])
                        arrays["trial_other_score_steps"][at] = float(np.linalg.norm(np.delete(d @ diff, source)))
                        path, square = path + step, square + step * step
                        rows.append({"topology": topology, "representation": representation, "alpha": alpha,
                                     "seed": seed, "trial": trial + 1, "example_index": source,
                                     "target": int(audit.SIGNS[source] > 0), "probability_go": p,
                                     "action": action, "reward": reward, "update_factor": g,
                                     "head_step_l2": step, "own_score_step": float((d @ diff)[source]),
                                     "other_score_step_l2": float(np.linalg.norm(np.delete(d @ diff, source))),
                                     "internal_step_l2": 0., "value_step_l2": 0., "head_update_calls": trial + 1,
                                     "internal_update_calls": 0, "value_update_calls": 0,
                                     "cumulative_head_path_l2": path, "cumulative_head_squared_l2": square,
                                     "head_norm": float(np.linalg.norm(new))})
                        head = new
                        if trial + 1 in cfg["checkpoints"]:
                            checkpoint(cfg["checkpoints"].index(trial + 1))
    return cfg, arrays, rows


@pytest.fixture
def invented_saved():
    return saved_fixture()


def test_masks_partition_every_source_destination_pair():
    masks = audit.category_masks()
    assert np.array_equal(masks[:3].sum(axis=0), np.ones((4, 4)))
    assert masks.sum(axis=(1, 2)).tolist() == [4, 4, 8, 12, 16]
    assert masks[1, 0, 1] and masks[1, 2, 3]
    assert masks[2, 0, 2] and masks[2, 3, 1]


def test_saved_state_audit_requires_no_action_generation(invented_saved):
    cfg, arrays, rows = invented_saved
    result = audit.saved_parent_state(cfg, arrays, rows)
    assert result["influences"].shape == (3, 2, 5, 2, 4, 4)
    assert result["trial_margin_delta"].shape == (3, 2, 5, 2, 8, 4)
    assert result["parent_increment_residual"] < 1e-14
    final_margin = audit.SIGNS * arrays["checkpoint_logits"][..., -1, :]
    assert np.allclose(result["trial_margin_delta"].sum(axis=-2), final_margin)
    final_correct = audit.sigmoid(final_margin)
    assert np.allclose(result["trial_probability_delta"].sum(axis=-2), final_correct - .5)


def test_actual_signed_changes_distinguish_helpful_and_harmful(invented_saved):
    cfg, arrays, rows = invented_saved
    result = audit.saved_parent_state(cfg, arrays, rows)
    c = result["trial_margin_delta"][0, 0, 0, 0, 0]
    assert c[0] > 0 and c[1] > 0 and c[2] < 0 and c[3] < 0
    isolated = result["trial_margin_delta"][0, 0, -1, 0, 0]
    assert isolated[0] > 0 and np.array_equal(isolated[1:], [0., 0., 0.])


def test_nogo_makes_all_destination_effects_exact_zero(invented_saved):
    cfg, arrays, rows = invented_saved
    result = audit.saved_parent_state(cfg, arrays, rows)
    assert np.array_equal(result["trial_margin_delta"][arrays["trial_actions"] == 0],
                          np.zeros((np.count_nonzero(arrays["trial_actions"] == 0), 4)))
    assert np.array_equal(result["trial_probability_delta"][arrays["trial_actions"] == 0],
                          np.zeros((np.count_nonzero(arrays["trial_actions"] == 0), 4)))


@pytest.mark.parametrize("field", ["trial_probabilities", "trial_update_factors", "checkpoint_heads",
                                    "checkpoint_logits", "checkpoint_probabilities", "t_real_raw_influences",
                                    "t_real_raw_base_budget_base_geometry_unit_rows", "t_ann_identity_directions"])
def test_saved_numeric_tampering_rejected(invented_saved, field):
    cfg, arrays, rows = invented_saved
    arrays[field].reshape(-1)[-1] += .02
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


@pytest.mark.parametrize("field", ["trial_actions", "trial_rewards", "model_seeds", "checkpoints", "schedules", "targets"])
def test_saved_identity_or_action_tampering_rejected(invented_saved, field):
    cfg, arrays, rows = invented_saved
    arrays[field].reshape(-1)[-1] += 2
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


@pytest.mark.parametrize("field", ["probability_go", "action", "reward", "update_factor", "head_norm",
                                    "internal_step_l2", "value_step_l2", "cumulative_head_path_l2"])
def test_saved_csv_tampering_rejected(invented_saved, field):
    cfg, arrays, rows = invented_saved
    rows[-1][field] += 1
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


def test_duplicate_or_missing_saved_trial_rejected(invented_saved):
    cfg, arrays, rows = invented_saved
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows[:-1])
    rows[-1] = rows[0].copy()
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


def test_nonfinite_or_wrong_shape_saved_arrays_rejected(invented_saved):
    cfg, arrays, rows = invented_saved
    arrays["trial_probabilities"][..., -1] = np.nan
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)
    cfg, arrays, rows = saved_fixture()
    arrays["trial_probabilities"] = arrays["trial_probabilities"][..., :-1]
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


def test_summary_exact_field_closure_and_meaning_flags():
    expected = {"counts": {"trials": 8}, "trained": False, "value": .2}
    audit.compare_tree(copy.deepcopy(expected), expected)
    for key, value in (("trained", True), ("value", .3), ("extra", 1)):
        tampered = copy.deepcopy(expected)
        tampered[key] = value
        with pytest.raises(ValueError):
            audit.compare_tree(tampered, expected)


@pytest.fixture
def invented_analysis(invented_saved):
    cfg, parent, parent_rows = invented_saved
    arrays, rows = audit.expected_artifacts(cfg, parent, parent_rows)
    return cfg, parent, parent_rows, arrays, rows


def test_complete_eight_array_contract_and_five_category_accounting(invented_analysis):
    cfg, parent, parent_rows, arrays, rows = invented_analysis
    summary = audit.expected_summary(cfg, arrays, rows)
    receipt = audit.verify_science(cfg, parent, parent_rows, arrays, rows, summary, allow_fixture=True)
    assert len(arrays) == 8 and len(rows) == 300
    assert arrays["reconstruction_residuals"].shape == (3, 2, 5, 2, 11)
    assert summary["validation"]["all_mass_balances_passed"]
    assert summary["validation"]["all_classification_counts_complete"]
    assert summary["validation"]["all_reconstruction_and_telescoping_checks_passed"]
    assert summary["counts"]["analyzed_parent_trial_records"] == 480
    assert summary["counts"]["new_training_trials"] == 0
    assert summary["counts"]["new_random_draws"] == 0
    assert receipt["arrays"] == 8 and receipt["no_rng_or_learner_calls"]
    assert summary["interpretation"]["pathwise_accounting_not_isolated_causal_effect"]
    assert len(summary["families"]) == 30 and len(summary["checkpoint_expectations"]) == 90


def test_source_destination_orientation_is_not_transposed(invented_analysis):
    _, parent, _, arrays, _ = invented_analysis
    m = parent["t_real_raw_influences"][0, 0]
    b = arrays["static_target_aligned"][0, 0, 0, 0]
    assert not np.allclose(m, m.T)
    assert b[1, 2] == pytest.approx(-m[1, 2])
    assert b[2, 1] == pytest.approx(-m[2, 1])
    assert b[1, 2] > 0 and b[2, 1] < 0


def test_uniform_source_action_expectation_matches_explicit_go_branch(invented_analysis):
    cfg, parent, _, arrays, _ = invented_analysis
    ti, ri, ai, si, ki = (0, 0, 1, 0, 1)
    logits = parent["checkpoint_logits"][ti, ri, ai, si, ki]
    p = audit.sigmoid(logits)
    m = parent["t_real_raw_influences"][si, ai]
    expected_margin = np.zeros((4, 4))
    expected_probability = np.zeros((4, 4))
    for source in range(4):
        for destination in range(4):
            dz_go = cfg["head_lr"] * (1-p[source]) * audit.SIGNS[source] * m[source, destination]
            dm_go = audit.SIGNS[destination] * dz_go
            expected_margin[source, destination] = .25 * p[source] * dm_go
            before = audit.sigmoid(audit.SIGNS[destination] * logits[destination])
            after = audit.sigmoid(audit.SIGNS[destination] * (logits[destination] + dz_go))
            expected_probability[source, destination] = .25 * p[source] * (after-before)
    actual_at = (ti, ri, ai, si, ki)
    assert np.allclose(arrays["checkpoint_expected_margin_by_source"][actual_at], expected_margin)
    assert np.allclose(arrays["checkpoint_expected_probability_by_source"][actual_at], expected_probability)
    wrong = audit.sigmoid(audit.SIGNS * logits + expected_margin.sum(axis=0))-audit.sigmoid(audit.SIGNS * logits)
    assert np.max(np.abs(wrong - expected_probability.sum(axis=0))) > 1e-9


def test_expected_checkpoint_groups_divide_four_destinations(invented_analysis):
    cfg, _, _, arrays, rows = invented_analysis
    summary = audit.expected_summary(cfg, arrays, rows)
    record = summary["checkpoint_expectations"][1]
    assert record["checkpoint"] == 4
    source = arrays["checkpoint_expected_probability_by_source"][0, 0, 0, :, 1]
    for ci, category in enumerate(audit.CATEGORY_NAMES):
        expected = (source * audit.category_masks()[ci]).sum(axis=(-1, -2)) / 4
        result = record["categories"][category]
        assert result["mean_probability_drift"] == pytest.approx(expected.mean())
        assert result["sd_probability_drift"] == pytest.approx(expected.std(ddof=1))


def test_threshold_neutral_counts_do_not_remove_exact_small_masses():
    values = np.array([1e-9, -1e-9, 0., 2e-8, -3e-8])
    result = audit.signed_values(values, np.ones(5, bool), "margin_", "sum", 1e-8)
    assert result["margin_neutral_cells"] == 3
    assert result["margin_helpful_cells"] == 1 and result["margin_harmful_cells"] == 1
    assert result["margin_helpful_sum"] == pytest.approx(2.1e-8)
    assert result["margin_harmful_sum"] == pytest.approx(3.1e-8)
    assert result["margin_net_sum"] == pytest.approx(-1e-8)


def test_actual_groups_sum_to_saved_final_probability_change(invented_analysis):
    _, parent, _, arrays, rows = invented_analysis
    selected = rows[:5]
    net = {row["category"]: row["probability_net_sum"] for row in selected}
    assert net["all"] == pytest.approx(net["self"]+net["cross_same"]+net["cross_opposite"])
    assert net["cross_all"] == pytest.approx(net["cross_same"]+net["cross_opposite"])
    endpoint = arrays["final_correct_probability"][0, 0, 0, 0].mean()
    assert net["all"] / 4. == pytest.approx(endpoint-.5)
    assert selected[0]["actual_cells"] == 8
    assert selected[1]["actual_cells"] == 8
    assert selected[2]["actual_cells"] == 16
    assert selected[3]["actual_cells"] == 24
    assert selected[4]["actual_cells"] == 32


@pytest.mark.parametrize("field", ["static_target_aligned", "trial_margin_delta", "trial_probability_delta", "trial_pre_logits",
                                    "checkpoint_expected_margin_by_source", "checkpoint_expected_probability_by_source",
                                    "reconstruction_residuals", "final_correct_probability"])
def test_every_derived_array_tampering_rejected(invented_analysis, field):
    _, _, _, expected, rows = invented_analysis
    actual = {key: value.copy() for key, value in expected.items()}
    actual[field].reshape(-1)[-1] += .01
    with pytest.raises(ValueError):
        audit.check_artifacts(expected, rows, actual, rows)


@pytest.mark.parametrize("field", ["static_cells", "static_helpful_cells", "static_harmful_cells", "static_neutral_cells",
                                    "static_helpful_mass", "static_harmful_mass", "static_net_mass", "actual_cells",
                                    "margin_helpful_cells", "margin_harmful_cells", "margin_neutral_cells",
                                    "margin_helpful_sum", "margin_harmful_sum", "margin_net_sum",
                                    "probability_helpful_cells", "probability_harmful_cells", "probability_neutral_cells",
                                    "probability_helpful_sum", "probability_harmful_sum", "probability_net_sum"])
def test_every_category_metric_tampering_rejected(invented_analysis, field):
    _, _, _, arrays, expected = invented_analysis
    rows = copy.deepcopy(expected)
    rows[-1][field] += 1
    with pytest.raises(ValueError):
        audit.check_artifacts(arrays, expected, arrays, rows)


@pytest.mark.parametrize("field,value", [("head_lr", .2), ("neutral_tolerance", .01), ("numeric_tolerance", .01),
                                        ("labels", [1, -1, 1, -1]), ("alphas", [0., .5, 1.]),
                                        ("source_result", "other"), ("categories", ["all"])])
def test_fixed_analysis_design_changes_rejected(invented_saved, field, value):
    cfg, _, _ = invented_saved
    audit.validate_science(cfg, allow_fixture=True)
    cfg[field] = value
    with pytest.raises(ValueError):
        audit.validate_science(cfg, allow_fixture=True)


def test_invented_fixture_cannot_satisfy_actual_execution_gate(invented_saved):
    cfg, _, _ = invented_saved
    with pytest.raises(ValueError):
        audit.validate_science(cfg)
    cfg["model_seeds"] = [17301]
    with pytest.raises(ValueError):
        audit.validate_science(cfg, allow_fixture=True)


def test_interpretive_claim_or_summary_count_tampering_rejected(invented_analysis):
    cfg, parent, parent_rows, arrays, rows = invented_analysis
    summary = audit.expected_summary(cfg, arrays, rows)
    summary["interpretation"]["shared_neural_mechanism_claim"] = True
    with pytest.raises(ValueError):
        audit.verify_science(cfg, parent, parent_rows, arrays, rows, summary, allow_fixture=True)
    summary = audit.expected_summary(cfg, arrays, rows)
    summary["counts"]["new_training_trials"] = 1
    with pytest.raises(ValueError):
        audit.verify_science(cfg, parent, parent_rows, arrays, rows, summary, allow_fixture=True)


def test_current_producer_matches_all_independent_invented_artifacts(invented_saved):
    from drosolearn.signed_spillover import analyze_saved, summarize
    cfg, parent, parent_rows = invented_saved
    arrays, rows = analyze_saved(cfg, parent, parent_rows)
    expected, expected_rows = audit.expected_artifacts(cfg, parent, parent_rows)
    audit.check_artifacts(expected, expected_rows, arrays, rows)
    summary = summarize(cfg, arrays, rows)
    audit.verify_science(cfg, parent, parent_rows, arrays, rows, summary, allow_fixture=True)


def test_partition_failure_fails_even_when_each_signed_mass_row_balances(invented_analysis):
    cfg, _, _, arrays, original = invented_analysis
    rows = copy.deepcopy(original)
    row = next(row for row in rows if row["category"] == "all")
    row["margin_helpful_sum"] += .01
    row["margin_net_sum"] += .01
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not summary["validation"]["all_mass_balances_passed"]


def test_sd_is_computational_seed_sd_not_confidence_interval(invented_analysis):
    cfg, _, _, arrays, rows = invented_analysis
    summary = audit.expected_summary(cfg, arrays, rows)
    values = [row["margin_helpful_sum"] for row in rows if row["topology"] == "real"
              and row["representation"] == "raw" and row["alpha"] == 0. and row["category"] == "all"]
    actual = summary["families"][0]["categories"]["all"]["margin_helpful_sum"]
    assert actual["sd"] == pytest.approx(np.std(values, ddof=1))
