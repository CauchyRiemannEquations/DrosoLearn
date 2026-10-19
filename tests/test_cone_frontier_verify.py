"""Invented finite geometry only; no actual saved experiment is read."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "independent_cone_tests", Path(__file__).resolve().parents[1] / "scripts/verify_cone_frontier.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_unconstrained_direction_and_certificate():
    c = np.asarray([3., -4., 0., 0.])
    a = np.zeros((0, 4))
    w, value = audit.independent_unit_cone(c, a, tolerance=1e-14)
    assert np.allclose(w, [.6, -.8, 0., 0.]) and value == pytest.approx(5.)
    assert audit.independent_certificate(c, a, w)["residuals"].max() < 1e-12


def test_single_binding_face_has_expected_projection():
    c, a = np.asarray([1., -2., 0., 0.]), np.asarray([[0., 1., 0., 0.]])
    w, value = audit.independent_unit_cone(c, a)
    assert np.allclose(w, [1., 0., 0., 0.]) and value == pytest.approx(1.)
    certificate = audit.independent_certificate(c, a, w)
    assert np.allclose(certificate["multipliers"], [2.])


def test_multiple_active_faces_and_inactive_positive_constraint():
    c = np.asarray([2., -3., -4., 1.])
    a = np.asarray([[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.]])
    w, value = audit.independent_unit_cone(c, a)
    assert np.allclose(w, np.asarray([2., 0., 0., 1.]) / np.sqrt(5.))
    assert value == pytest.approx(np.sqrt(5.))
    audit.independent_certificate(c, a, w)


def test_zero_optimum_can_return_zero_inside_unit_ball():
    c, a = np.asarray([-1., -2., 0., 0.]), np.eye(4)
    w, value = audit.independent_unit_cone(c, a)
    assert np.array_equal(w, np.zeros(4)) and value == 0.
    audit.independent_certificate(c, a, w)


def test_zero_objective_does_not_get_arbitrary_nonzero_gain():
    c, a = np.zeros(4), np.eye(4)
    w, value = audit.independent_unit_cone(c, a)
    assert not w.any() and value == 0.
    audit.independent_certificate(c, a, w)


def test_ambient_rotation_preserves_optimized_influence():
    c = np.asarray([1., -.7, .4, -.5])
    a = np.asarray([[1., .5, 0., 0.], [0., 1., .3, 0.], [0., 0., 1., 1.]])
    q, _ = np.linalg.qr(np.asarray([[1., 2., 3., 4.], [2., 4., 1., 3.],
                                   [4., 1., 2., 3.], [3., 1., 4., 2.]]))
    w, value = audit.independent_unit_cone(c, a)
    rotated, rotated_value = audit.independent_unit_cone(q.T @ c, a @ q)
    assert np.allclose(q @ rotated, w) and rotated_value == pytest.approx(value)
    audit.independent_certificate(c, a, w)


def test_infeasible_point_rejected_by_certificate():
    with pytest.raises(ValueError, match="KKT"):
        audit.independent_certificate(np.asarray([0., -1.]), np.asarray([[0., 1.]]), np.asarray([0., -1.]))


@pytest.mark.parametrize("w", [[2., 0.], [0., 1.], [-1., 0.]])
def test_cap_or_suboptimal_points_rejected(w):
    with pytest.raises(ValueError, match="KKT"):
        audit.independent_certificate(np.asarray([1., 0.]), np.zeros((0, 2)), np.asarray(w))


@pytest.mark.parametrize("c,a", [([np.nan, 0.], [[1., 0.]]), ([1., 0.], [[np.inf, 0.]]),
                                  ([1., 0.], [[1., 0., 0.]]), ([[1., 0.]], [[1., 0.]])])
def test_nonfinite_or_wrong_axes_rejected(c, a):
    with pytest.raises(ValueError):
        audit.independent_unit_cone(c, a)


def test_enumerated_optimum_dominates_a_dense_circle_grid():
    c, a = np.asarray([.4, -1.]), np.asarray([[1., .2], [.7, -.3]])
    w, value = audit.independent_unit_cone(c, a)
    angles = np.linspace(0., 2. * np.pi, 10001)
    grid = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    feasible = grid[np.all(grid @ a.T >= 0., axis=1)]
    assert value >= float(np.max(feasible @ c)) - 1e-10
    audit.independent_certificate(c, a, w)


def test_rectangular_solve_retains_near_parallel_constraint_rank():
    c = np.asarray([-2., -1e-6, 1., 0.])
    a = np.asarray([[1., 0., 0., 0.], [1., 1e-6, 0., 0.]])
    w, value = audit.independent_unit_cone(c, a, tolerance=1e-14)
    assert value == pytest.approx(1., abs=1e-9)
    assert np.allclose(w, [0., 0., 1., 0.], atol=1e-9)
    audit.independent_certificate(c, a, w)


def test_independent_solver_has_no_new_producer_or_solver_import():
    import ast
    source = Path(SPEC.origin).read_text(encoding="utf8")
    modules = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert "drosolearn.cone_geometry" not in modules
    assert "drosolearn.cone_frontier" not in modules
    for token in ("default_rng", "SeedSequence", "build_graphs", "VectorActor"):
        assert token not in source


def test_algebraic_dual_bound_matches_binding_face_optimum():
    result = audit.validate_dual_witness([1., -2.], [[0., 1.]], [1., 0.], [2.])
    assert np.array_equal(result["dual_vector"], [1., 0.])
    assert result["dual_upper_bound"] == result["achieved_objective"] == 1.
    assert result["dual_optimality_gap"] == 0.


@pytest.mark.parametrize("w,mu", [([1., 0.], [-1e-12]), ([1., 0.], [1.]),
                                   ([0., 1.], [2.]), ([2., 0.], [2.])])
def test_invalid_or_loose_dual_witness_rejected(w, mu):
    with pytest.raises(ValueError):
        audit.validate_dual_witness([1., -2.], [[0., 1.]], w, mu)


def test_independent_dual_face_enumeration_avoids_near_parallel_primal_inflation():
    c = np.asarray([-2., -1e-6, 1., 0.])
    a = np.asarray([[1., 0., 0., 0.], [1., 1e-6, 0., 0.]])
    upper, mu = audit.independent_support_bound(c, a)
    assert upper == pytest.approx(1., abs=1e-12)
    assert np.all(mu >= 0.) and np.allclose(mu, [1., 1.], atol=1e-9)
    assert upper == pytest.approx(np.linalg.norm(c + a.T @ mu), abs=1e-15)
    assert float(c @ np.asarray([0., 0., 1., 0.])) <= upper


def test_independent_dual_enumeration_keeps_zero_multiplier_candidate():
    c, a = np.asarray([2., 1.]), np.eye(2)
    upper, mu = audit.independent_support_bound(c, a)
    assert upper == pytest.approx(np.sqrt(5.)) and not mu.any()
    w, primal_value = audit.independent_unit_cone(c, a)
    assert primal_value == pytest.approx(upper)
    audit.validate_dual_witness(c, a, w, mu)


def test_independent_dual_enumeration_contains_a_multi_face_optimum():
    c, a = np.asarray([2., -3., -4., 1.]), np.eye(4)[:3]
    upper, mu = audit.independent_support_bound(c, a)
    assert upper == pytest.approx(np.sqrt(5.))
    assert np.allclose(mu, [0., 3., 4.])
    w, value = audit.independent_unit_cone(c, a)
    assert value == pytest.approx(upper)
    audit.validate_dual_witness(c, a, w, mu)


@pytest.fixture(scope="module")
def invented_analysis():
    fixture_spec = importlib.util.spec_from_file_location(
        "cone_core_test_fixture", Path(__file__).with_name("test_cone_frontier.py"))
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    cfg, parent = fixture_module.saved_fixture()
    from drosolearn.cone_frontier import analyze_saved, summarize
    arrays, rows = analyze_saved(cfg, parent)
    summary = summarize(cfg, arrays, rows)
    return cfg, parent, arrays, rows, summary


def test_complete_invented_saved_contract_and_producer_bridge(invented_analysis):
    cfg, parent, arrays, rows, summary = invented_analysis
    report = audit.verify_science(cfg, parent, arrays, rows, summary, allow_fixture=True)
    assert report["arrays"] == 15
    assert report["parent_trial_records_checked"] == 96
    assert report["candidate_trial_records_checked"] == 672
    assert report["measurement_records_checked"] == 252
    assert report["active_faces_reoptimized_in_original_feature_space"]
    assert report["new_training_trials"] == 0


@pytest.mark.parametrize("field", list(audit.FIELDS))
def test_every_new_array_corruption_is_rejected(invented_analysis, field):
    cfg, parent, expected, _, _ = invented_analysis
    arrays = copy.deepcopy(expected)
    if arrays[field].dtype.kind == "b":
        arrays[field].reshape(-1)[-1] ^= True
    else:
        arrays[field].reshape(-1)[-1] += .2 if arrays[field].dtype.kind == "f" else 1
    with pytest.raises(ValueError):
        audit.expected_artifacts(cfg, parent, arrays)


@pytest.mark.parametrize("field", list(audit.METRICS))
def test_every_metric_corruption_rejected(invented_analysis, field):
    _, _, _, expected, _ = invented_analysis
    rows = copy.deepcopy(expected)
    row = next(row for row in rows if row[field] is not None)
    row[field] += 1
    with pytest.raises(ValueError):
        audit.check_rows(rows, expected)


def test_no_go_zero_effects_no_optimizer_and_phase_discovery(invented_analysis):
    cfg, _, arrays, rows, _ = invented_analysis
    empty = arrays["update_factors"] == 0.
    assert np.all(arrays["case_indices"][empty] == -1)
    for key in ("candidate_influences", "candidate_margin_delta", "candidate_step_lengths"):
        assert not arrays[key][empty].any()
    # Its first presentation of input zero is NoGo; the fourth sign is not
    # obtained until its actual Go on trial five, which belongs to full phase.
    assert arrays["full_known_after"][0, 0, 0].tolist() == [False, False, False, False, True, True, True, True]
    assert not arrays["full_known_after"][0, 0, 1].any()
    full = next(row for row in rows if row["seed"] == cfg["model_seeds"][1] and row["phase"] == "full")
    assert full["own_retention"] is None and full["help_retention"] is None


def test_unknown_sign_never_enters_observed_problem():
    d = np.eye(4)
    a, c, known, useful = audit.independent_problem(d, 0, [1, 0, 0, 0], 1, 1e-8)
    assert np.array_equal(a, d[:1]) and np.array_equal(c, d[0])
    assert known.tolist() == [0] and useful.size == 0


@pytest.mark.parametrize("field", ["trial_rewards", "trial_update_factors", "trial_known_after",
    "trial_known_before", "trial_routing_signs", "trial_directions", "trial_actual_influence"])
def test_parent_feedback_slice_corruption_rejected(invented_analysis, field):
    cfg, original, _, _, _ = invented_analysis
    parent = copy.deepcopy(original)
    parent[field][0, 0, 3, 0, 0] += 1
    with pytest.raises(ValueError):
        audit.saved_feedback(cfg, parent)


def test_array_and_table_closure_and_undefined_values(invented_analysis):
    cfg, parent, arrays, rows, _ = invented_analysis
    with pytest.raises(ValueError):
        audit.expected_artifacts(cfg, parent, dict(arrays, extra=np.zeros(1)))
    with pytest.raises(ValueError):
        audit.check_rows(rows[:-1], rows)
    with pytest.raises(ValueError):
        audit.check_rows(list(reversed(rows)), rows)
    csv_rows = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    audit.check_rows(csv_rows, rows)
    changed = copy.deepcopy(rows)
    next(row for row in changed if row["own_retention"] is None)["own_retention"] = 0.
    with pytest.raises(ValueError):
        audit.check_rows(changed, rows)


def test_summary_qualification_and_claims_cannot_be_changed(invented_analysis):
    cfg, parent, arrays, rows, expected = invented_analysis
    for section, key, value in (("interpretation", "same_saved_scalar_not_autonomous_policy", False),
        ("interpretation", "score_not_accuracy", False), ("counts", "new_training_trials", 1),
        ("validation", "all_case_certificates_passed", False)):
        changed = copy.deepcopy(expected)
        changed[section][key] = value
        with pytest.raises(ValueError):
            audit.verify_science(cfg, parent, arrays, rows, changed, allow_fixture=True)


@pytest.mark.parametrize("field,value", [("head_lr", .2), ("labels", [1, -1, 1, -1]),
    ("numeric_tolerance", 1e-3), ("neutral_tolerance", 1e-3), ("svd_rcond", 1e-4),
    ("parent_method", "shared"), ("candidates", ["observed_own"]), ("phases", ["all"])])
def test_frozen_scientific_choices_cannot_change(invented_analysis, field, value):
    cfg = copy.deepcopy(invented_analysis[0])
    cfg[field] = value
    with pytest.raises(ValueError):
        audit.validate_science(cfg, allow_fixture=True)


def test_invented_config_does_not_enter_actual_gate(invented_analysis):
    with pytest.raises(ValueError):
        audit.validate_science(invented_analysis[0])
