"""Invented saved records only; actual graph, seeds, and results are never read."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "normalization_independent_tests", Path(__file__).resolve().parents[1] / "scripts/verify_normalization_loss.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def saved_fixture():
    cfg = {"study_id": "normalization_loss_v1", "scope": "integration_test",
           "module": "drosolearn.normalization_loss", "protocol": "docs/normalization-loss-protocol.md",
           "source_result": "results/selective_spillover_v1", "topologies": list(audit.TOPOLOGIES),
           "representations": list(audit.REPRESENTATIONS), "methods": list(audit.METHODS),
           "model_seeds": [90091, 90092], "training_trials": 8, "checkpoints": [0, 4, 8],
           "head_lr": .1, "numeric_tolerance": 1e-8, "neutral_tolerance": 1e-8,
           "labels": [1, 1, -1, -1], "phases": list(audit.PHASES),
           "categories": list(audit.CATEGORIES), "external_manifest": None}
    shape = (3, 2, 4, 2, 8)
    integer = {"actions", "rewards", "head_update_calls", "internal_update_calls", "value_update_calls",
               "memory_changed", "memory_observation_calls", "memory_discoveries"}
    fields = ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
        "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls",
        "update_factors", "expected_head_steps", "head_step_squared_l2", "cumulative_head_path_l2",
        "cumulative_head_squared_l2", "head_norms", "raw_direction_norms", "normalization_scales",
        "memory_changed", "memory_observation_calls", "memory_discoveries")
    arrays = {"trial_" + field: np.zeros(shape, int if field in integer else float) for field in fields}
    arrays.update({"model_seeds": np.asarray(cfg["model_seeds"]), "checkpoints": np.asarray(cfg["checkpoints"]),
                   "targets": np.array([1, 1, 0, 0]),
                   "schedules": np.tile(np.asarray([0, 1, 2, 3, 3, 2, 1, 0]), (2, 1)),
                   "uniforms": np.asarray([[0., 0., 0., 0., .999, 0., .999, 0.],
                                            [0., .999, .999, .999, .999, .999, .999, .999]])})
    for field in ("known_before", "known_after", "routing_signs"):
        arrays["trial_" + field] = np.empty(shape + (4,), np.int8)
    arrays["trial_harm_mask"] = np.empty(shape + (4,), bool)
    for field in ("desired_influence", "actual_influence"):
        arrays["trial_" + field] = np.empty(shape + (4,))
    arrays["trial_directions"] = np.empty(shape + (4,))
    cp = shape[:-1] + (3,)
    for field in ("heads", "logits", "probabilities", "lookup_probabilities"):
        arrays["checkpoint_" + field] = np.empty(cp + (4,))
    arrays["checkpoint_known"] = np.empty(cp + (4,), np.int8)
    for field in ("head_calls", "internal_calls", "value_calls", "memory_observation_calls", "memory_discoveries"):
        arrays["checkpoint_" + field] = np.zeros(cp, int)
    arrays["first_discovery_trials"] = np.full(shape[:-1] + (4,), 9, int)
    rows = []
    # Negative same-target geometry and negative opposite-target geometry are both present.
    raw = np.asarray([[1., 0., 0., 0.], [-.3, .9, 0., 0.], [.4, .1, .8, 0.], [-.2, .3, .1, .9]])
    d = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    inverse, gram = np.linalg.pinv(d, rcond=1e-12), d @ d.T
    for ti, topology in enumerate(audit.TOPOLOGIES):
        for ri, representation in enumerate(audit.REPRESENTATIONS):
            prefix = "g_" + topology + "_" + representation + "_"
            arrays[prefix + "unit_rows"] = np.stack([d, d])
            arrays[prefix + "pseudoinverse"] = np.stack([inverse, inverse])
            arrays[prefix + "base_influences"] = np.stack([gram, gram])
            for mi, method in enumerate(audit.METHODS):
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, mi, si)
                    head, known = np.zeros(4), np.zeros(4, np.int8)
                    path = energy = 0.
                    calls = discoveries = 0
                    def checkpoint(step):
                        pos = at + (cfg["checkpoints"].index(step),)
                        for field, value in (("heads", head), ("logits", d @ head),
                            ("probabilities", audit.sigmoid(d @ head)), ("lookup_probabilities", (known + 1.) / 2.),
                            ("known", known), ("head_calls", step), ("memory_observation_calls", calls),
                            ("memory_discoveries", discoveries)):
                            arrays["checkpoint_" + field][pos] = value
                    checkpoint(0)
                    for t, source in enumerate(arrays["schedules"][si]):
                        i, pos = int(source), at + (t,)
                        p = float(audit.sigmoid((d @ head)[i]))
                        a = int(arrays["uniforms"][si, t] < p)
                        r = a * int(audit.SIGNS[i])
                        old_known = known.copy()
                        changed = int(a == 1 and known[i] == 0)
                        if a:
                            known[i] = r
                            calls += 1
                            discoveries += changed
                            if changed:
                                arrays["first_discovery_trials"][at + (i,)] = t + 1
                        routing = audit.SIGNS.astype(int) if mi == 2 else known.copy() if mi == 3 else np.zeros(4, int)
                        desired, unscaled, direction, harm, scale = audit.independent_direction(d, inverse, i, routing, method, 1e-8)
                        factor = .1 * (a - p) * r
                        newer = head + factor * direction
                        increment, influence = newer - head, d @ direction
                        effect, step = d @ increment, float(np.linalg.norm(increment))
                        path += step
                        energy += step ** 2
                        values = {"probabilities": p, "actions": a, "rewards": r, "head_steps": step,
                            "own_score_steps": float(effect[i]), "other_score_steps": float(np.linalg.norm(np.delete(effect, i))),
                            "head_update_calls": t + 1, "update_factors": factor, "expected_head_steps": abs(factor) * np.linalg.norm(direction),
                            "head_step_squared_l2": step ** 2, "cumulative_head_path_l2": path, "cumulative_head_squared_l2": energy,
                            "head_norms": float(np.linalg.norm(newer)), "raw_direction_norms": np.linalg.norm(unscaled),
                            "normalization_scales": scale, "memory_changed": changed, "memory_observation_calls": calls,
                            "memory_discoveries": discoveries}
                        for field, value in values.items():
                            arrays["trial_" + field][pos] = value
                        for field, value in (("known_before", old_known), ("known_after", known), ("routing_signs", routing),
                            ("harm_mask", harm), ("desired_influence", desired), ("actual_influence", influence), ("directions", direction)):
                            arrays["trial_" + field][pos] = value
                        row = {"topology": topology, "representation": representation, "method": method, "seed": seed,
                            "trial": t + 1, "example_index": i, "target": int(audit.SIGNS[i] > 0), "probability_go": p,
                            "action": a, "reward": r, "head_update_calls": t + 1, "internal_update_calls": 0,
                            "value_update_calls": 0, "head_step_l2": step, "own_score_step": float(effect[i]),
                            "other_score_step_l2": float(np.linalg.norm(np.delete(effect, i))), "internal_step_l2": 0.,
                            "value_step_l2": 0., "head_nonzero_step": int(step > 0), "internal_nonzero_step": 0,
                            "value_nonzero_step": 0, "update_factor": factor, "direction_l2": float(np.linalg.norm(direction)),
                            "expected_head_step_l2": abs(factor) * np.linalg.norm(direction), "head_step_squared_l2": step ** 2,
                            "cumulative_head_path_l2": path, "cumulative_head_squared_l2": energy,
                            "head_norm": float(np.linalg.norm(newer)), "raw_direction_norm": float(np.linalg.norm(unscaled)),
                            "normalization_scale": scale, "memory_changed": changed, "memory_observation_calls": calls,
                            "memory_discoveries": discoveries, "direction_residual_max": 0.}
                        for field, value in (("known_before", old_known), ("known_after", known), ("routing_sign", routing),
                            ("harm", harm), ("desired_influence", desired), ("actual_influence", influence)):
                            for j in range(4):
                                row[field + "_" + str(j)] = float(value[j])
                        rows.append(row)
                        head = newer
                        if t + 1 in cfg["checkpoints"]:
                            checkpoint(t + 1)
    return cfg, arrays, rows


@pytest.fixture(scope="module")
def cached_saved():
    return saved_fixture()


@pytest.fixture
def invented_saved(cached_saved):
    return copy.deepcopy(cached_saved)


@pytest.fixture(scope="module")
def cached_analysis(cached_saved):
    cfg, parent, parent_rows = cached_saved
    arrays, rows = audit.expected_artifacts(cfg, parent, parent_rows)
    summary = audit.expected_summary(cfg, arrays, rows)
    return cfg, parent, parent_rows, arrays, rows, summary


@pytest.fixture
def invented_analysis(cached_analysis):
    return copy.deepcopy(cached_analysis)


def test_complete_independent_contract_and_no_new_training(cached_analysis):
    cfg, parent, parent_rows, arrays, rows, summary = cached_analysis
    report = audit.verify_science(cfg, parent, parent_rows, arrays, rows, summary, allow_fixture=True)
    assert len(arrays) == 11 and len(rows) == 1296
    assert arrays["reconstruction_residuals"].shape == (3, 2, 4, 2, 12)
    assert summary["validation"]["all_stage_and_partition_balances_passed"]
    assert summary["counts"]["analyzed_parent_trial_records"] == 384
    assert summary["counts"]["new_training_trials"] == 0
    assert report["no_rng_learner_or_graph_construction_calls"]


def test_current_producer_matches_independent_saved_numeric_contract(cached_analysis):
    from drosolearn.normalization_loss import analyze_saved, summarize
    cfg, parent, parent_rows, expected, expected_rows, expected_summary = cached_analysis
    arrays, rows = analyze_saved(cfg, parent, parent_rows)
    audit.check_artifacts(expected, expected_rows, arrays, rows)
    summary = summarize(cfg, arrays, rows)
    audit.compare_tree(summary, expected_summary)
    audit.verify_science(cfg, parent, parent_rows, arrays, rows, summary, allow_fixture=True)


def test_discovery_trial_is_full_using_actual_observed_memory(cached_analysis):
    _, parent, _, arrays, _, _ = cached_analysis
    for mi in range(4):
        assert arrays["full_known_after"][0, 0, mi, 0].tolist() == [False, False, False, True, True, True, True, True]
        assert not arrays["full_known_after"][0, 0, mi, 1].any()
    # Oracle starts with a privileged routing table, but actual observation still starts unknown.
    assert np.count_nonzero(parent["trial_routing_signs"][0, 0, 2, 0, 0]) == 4
    assert np.count_nonzero(parent["trial_known_after"][0, 0, 2, 0, 0]) == 1


def test_geometric_help_and_harm_not_equivalent_to_same_target(cached_analysis):
    _, parent, _, arrays, _, _ = cached_analysis
    classes = arrays["cross_class"][0, 0, 3, 0]
    assert classes[0, 0] == 2
    assert classes[0, 1] == -1  # Same Go target, negatively correlated features.
    assert classes[0, 3] == 1   # Opposite target, negatively correlated features.
    masks = audit.category_masks(classes, parent["schedules"][0], parent["trial_known_after"][0, 0, 3, 0])
    assert not np.array_equal(masks[1], masks[5])
    assert (masks[4] & masks[1]).any() and (masks[4] & masks[2]).any()
    assert np.array_equal(masks[0] + masks[7], masks[8])


def test_three_stages_preserve_raw_self_then_shrink_it(cached_analysis):
    _, parent, _, arrays, _, _ = cached_analysis
    pos = (0, 0, 3, 0, 3)
    source = int(parent["schedules"][0, 3])
    b, p, a = [arrays[name + "_margin_delta"][pos] for name in ("baseline", "desired", "actual")]
    assert b[source] == pytest.approx(p[source])
    assert 0 < a[source] < p[source]
    assert np.allclose(p * arrays["scales"][pos], a)
    assert np.allclose(b - a, arrays["suppression_margin_delta"][pos] + arrays["normalization_margin_delta"][pos])


def test_no_go_effects_exact_zero_but_direction_statistics_still_exist(cached_analysis):
    _, parent, _, arrays, rows, _ = cached_analysis
    zero = parent["trial_actions"] == 0
    for stage in ("baseline", "desired", "actual", "suppression", "normalization"):
        assert not np.any(arrays[stage + "_margin_delta"][zero])
    assert np.array_equal(arrays["nonzero_update"], ~zero)
    selected = next(row for row in rows if row["topology"] == "real" and row["representation"] == "raw"
                    and row["method"] == "oracle_selective" and row["seed"] == 90092
                    and row["phase"] == "all" and row["category"] == "self")
    assert selected["trial_count"] == 8 and selected["updated_trials"] == 1
    assert selected["scale_mean_all"] != pytest.approx(selected["scale_mean_updated"])


def test_empty_phase_nullable_stats_and_counts(cached_analysis):
    _, _, _, _, rows, summary = cached_analysis
    empty = next(row for row in rows if row["seed"] == 90092 and row["phase"] == "full")
    for key in ("trial_count", "updated_trials", "cells", "updated_cells"):
        assert empty[key] == 0
    for key in ("scale_mean_all", "scale_mean_updated", "raw_norm_mean", "scale_min", "scale_max",
                "helpful_retention", "harmful_retention"):
        assert empty[key] is None
    stats = summary["families"][0]["phases"]["full"]["categories"]["self"]["scale_mean_all"]
    assert stats["valid_count"] == 1 and stats["total_count"] == 2 and stats["sd"] == 0.


def test_direction_stats_are_not_weighted_by_number_of_selected_destinations(cached_analysis):
    _, _, _, _, rows, _ = cached_analysis
    same_group = [row for row in rows if row["topology"] == "real" and row["representation"] == "raw"
                  and row["method"] == "feedback_selective" and row["seed"] == 90091 and row["phase"] == "all"]
    assert len(set(row["cells"] for row in same_group)) > 1
    assert len(set(row["scale_mean_all"] for row in same_group)) == 1
    assert len(set(row["trial_count"] for row in same_group)) == 1


def test_pooled_ratio_is_mass_weighted_not_seed_ratio_mean(cached_analysis):
    _, _, _, _, rows, summary = cached_analysis
    family = summary["families"][3]["phases"]["all"]
    selected = [row for row in rows if row["topology"] == "real" and row["representation"] == "raw"
                and row["method"] == "feedback_selective" and row["phase"] == "all" and row["category"] == "self"]
    pooled = sum(row["actual_helpful"] for row in selected) / sum(row["desired_helpful"] for row in selected)
    assert family["pooled_categories"]["self"]["helpful_retention"] == pytest.approx(pooled)
    assert pooled != pytest.approx(family["categories"]["self"]["helpful_retention"]["mean"])


def test_exact_tiny_mass_is_not_zeroed_by_neutral_classification():
    arrays = {name + "_margin_delta": np.array([[1e-10, -2e-10, 0., 0.]]) for name in ("baseline", "desired", "actual")}
    arrays.update({"suppression_margin_delta": np.zeros((1, 4)), "normalization_margin_delta": np.zeros((1, 4)),
                   "scales": np.ones(1), "raw_norms": np.ones(1), "nonzero_update": np.ones(1, bool)})
    result = audit.row_metrics(arrays, (), np.ones(1, bool), np.ones((1, 4), bool))
    assert result["actual_helpful"] == pytest.approx(1e-10)
    assert result["actual_harmful"] == pytest.approx(2e-10)
    assert result["actual_net"] == pytest.approx(-1e-10)
    assert result["helpful_retention"] == 1.


@pytest.mark.parametrize("field", ["trial_actions", "trial_rewards", "trial_known_before", "trial_known_after",
    "trial_routing_signs", "trial_memory_changed", "trial_memory_observation_calls", "trial_memory_discoveries",
    "first_discovery_trials", "checkpoint_known", "checkpoint_head_calls", "checkpoint_memory_discoveries",
    "checkpoint_lookup_probabilities", "model_seeds", "checkpoints", "targets", "schedules", "uniforms"])
def test_saved_observation_or_identity_corruption_rejected(invented_saved, field):
    cfg, arrays, rows = invented_saved
    arrays[field].reshape(-1)[-1] += 2
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


@pytest.mark.parametrize("field", ["trial_probabilities", "trial_update_factors", "trial_directions",
    "trial_actual_influence", "trial_desired_influence", "trial_normalization_scales", "trial_raw_direction_norms",
    "trial_own_score_steps", "trial_other_score_steps", "trial_head_steps", "trial_cumulative_head_path_l2",
    "trial_cumulative_head_squared_l2", "trial_head_norms", "checkpoint_heads", "checkpoint_logits",
    "checkpoint_probabilities", "g_real_raw_unit_rows", "g_ann_identity_base_influences", "g_rewired_raw_pseudoinverse"])
def test_saved_arithmetic_corruption_rejected(invented_saved, field):
    cfg, arrays, rows = invented_saved
    arrays[field].reshape(-1)[-1] += .02
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


def test_boolean_harm_mask_corruption_rejected(invented_saved):
    cfg, arrays, rows = invented_saved
    arrays["trial_harm_mask"].reshape(-1)[-1] ^= True
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


@pytest.mark.parametrize("field", ["head_step_l2", "own_score_step", "other_score_step_l2", "normalization_scale",
    "raw_direction_norm", "memory_changed", "known_before_0", "known_after_0", "routing_sign_0",
    "harm_0", "desired_influence_0", "actual_influence_0", "action", "reward", "probability_go"])
def test_saved_csv_arithmetic_corruption_rejected(invented_saved, field):
    cfg, arrays, rows = invented_saved
    rows[-1][field] += 2
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, arrays, rows)


def test_missing_duplicate_or_extra_parent_rows_rejected(invented_saved):
    cfg, parent, rows = invented_saved
    for bad in (rows[:-1], rows + [rows[-1]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            audit.saved_parent_state(cfg, parent, bad)


@pytest.mark.parametrize("field", ["trial_probabilities", "trial_actual_influence", "trial_directions", "g_real_raw_unit_rows"])
def test_nonfinite_saved_numeric_rejected(invented_saved, field):
    cfg, parent, rows = invented_saved
    parent[field].reshape(-1)[0] = np.nan
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, parent, rows)


@pytest.mark.parametrize("field", ["trial_internal_steps", "trial_value_steps", "trial_internal_update_calls", "trial_value_update_calls"])
def test_internal_or_value_updates_rejected(invented_saved, field):
    cfg, parent, rows = invented_saved
    parent[field].reshape(-1)[0] = 1
    with pytest.raises(ValueError):
        audit.saved_parent_state(cfg, parent, rows)


@pytest.mark.parametrize("field", ["baseline_margin_delta", "desired_margin_delta", "actual_margin_delta",
    "suppression_margin_delta", "normalization_margin_delta", "scales", "raw_norms", "cross_class",
    "reconstruction_residuals", "full_known_after", "nonzero_update"])
def test_every_analysis_array_corruption_rejected(cached_analysis, field):
    _, _, _, expected, rows, _ = cached_analysis
    arrays = copy.deepcopy(expected)
    if arrays[field].dtype.kind == "b":
        arrays[field].reshape(-1)[-1] ^= True
    else:
        arrays[field].reshape(-1)[-1] += 1
    with pytest.raises(ValueError):
        audit.check_artifacts(expected, rows, arrays, rows)


@pytest.mark.parametrize("field", list(audit.METRICS))
def test_every_csv_metric_corruption_rejected(cached_analysis, field):
    _, _, _, arrays, original, _ = cached_analysis
    rows = copy.deepcopy(original)
    row = next(row for row in rows if row[field] is not None)
    row[field] += 1
    with pytest.raises(ValueError):
        audit.check_artifacts(arrays, original, arrays, rows)


def test_undefined_ratio_must_not_be_zero_filled(cached_analysis):
    _, _, _, arrays, original, _ = cached_analysis
    rows = copy.deepcopy(original)
    next(row for row in rows if row["helpful_retention"] is None)["helpful_retention"] = 0.
    with pytest.raises(ValueError):
        audit.check_artifacts(arrays, original, arrays, rows)


def test_empty_csv_values_are_allowed_only_for_undefined_metrics(cached_analysis):
    _, _, _, arrays, expected, _ = cached_analysis
    rows = [{key: "" if value is None else str(value) for key, value in row.items()} for row in expected]
    audit.check_artifacts(arrays, expected, arrays, rows)
    rows[0]["trial_count"] = ""
    with pytest.raises(ValueError):
        audit.check_artifacts(arrays, expected, arrays, rows)


def test_array_and_csv_closure_identity_order_are_exact(cached_analysis):
    _, _, _, arrays, rows, _ = cached_analysis
    extra = dict(arrays, extra=np.zeros(1))
    with pytest.raises(ValueError):
        audit.check_artifacts(arrays, rows, extra, rows)
    for altered in (rows[:-1], list(reversed(rows)), [dict(reversed(list(row.items()))) for row in rows]):
        with pytest.raises(ValueError):
            audit.check_artifacts(arrays, rows, arrays, altered)


@pytest.mark.parametrize("section,key,value", [("interpretation", "score_loss_not_accuracy_loss", False),
    ("interpretation", "saved_scalar_reference_not_counterfactual_policy", False),
    ("interpretation", "shared_neural_mechanism_claim", True), ("counts", "new_training_trials", 1),
    ("counts", "measurement_records", 0), ("validation", "all_stage_and_partition_balances_passed", False)])
def test_summary_claim_or_count_corruption_rejected(cached_analysis, section, key, value):
    _, _, _, _, _, expected = cached_analysis
    summary = copy.deepcopy(expected)
    summary[section][key] = value
    with pytest.raises(ValueError):
        audit.compare_tree(summary, expected)


def test_stage_partition_corruption_detected_without_relying_on_individual_row_sign_balance(cached_analysis):
    cfg, _, _, arrays, original, _ = cached_analysis
    rows = copy.deepcopy(original)
    row = next(row for row in rows if row["category"] == "all")
    row["actual_helpful"] += .01
    row["actual_net"] += .01
    row["normalization_net"] -= .01
    assert not audit.expected_summary(cfg, arrays, rows)["validation"]["all_stage_and_partition_balances_passed"]


def test_seed_stats_report_valid_count_and_sample_sd():
    assert audit.stats([None, None]) == {"mean": None, "sd": None, "valid_count": 0, "total_count": 2}
    assert audit.stats([None, 2.]) == {"mean": 2., "sd": 0., "valid_count": 1, "total_count": 2}
    result = audit.stats([1., None, 3.])
    assert result["mean"] == 2. and result["sd"] == pytest.approx(np.sqrt(2.))
    assert result["valid_count"] == 2 and result["total_count"] == 3


@pytest.mark.parametrize("field,value", [("head_lr", .2), ("neutral_tolerance", 1e-3), ("numeric_tolerance", 1e-3),
    ("labels", [1, -1, 1, -1]), ("methods", ["shared"]), ("phases", ["all"]),
    ("categories", ["all"]), ("source_result", "another-result")])
def test_fixed_scientific_choices_cannot_be_changed(invented_saved, field, value):
    cfg, _, _ = invented_saved
    audit.validate_science(cfg, allow_fixture=True)
    cfg[field] = value
    with pytest.raises(ValueError):
        audit.validate_science(cfg, allow_fixture=True)


def test_actual_gate_rejects_invented_fixture_and_fixture_rejects_real_seed(invented_saved):
    cfg, _, _ = invented_saved
    with pytest.raises(ValueError):
        audit.validate_science(cfg)
    cfg["model_seeds"] = [17401]
    with pytest.raises(ValueError):
        audit.validate_science(cfg, allow_fixture=True)


def test_verifier_does_not_import_scientific_producer_or_random_generation():
    import ast
    text = (Path(__file__).resolve().parents[1] / "scripts/verify_normalization_loss.py").read_text(encoding="utf8")
    tree = ast.parse(text)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "drosolearn.normalization_loss" not in imports
    assert "drosolearn.selective_spillover" not in imports
    for name in ("default_rng", "SeedSequence", "VectorActor", "build_graphs", "load_projection"):
        assert name not in text
