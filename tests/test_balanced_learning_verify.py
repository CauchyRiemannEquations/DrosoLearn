"""Independent arithmetic and invented-only autonomous verification fixtures."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("balanced_independent_tests",
    Path(__file__).resolve().parents[1] / "scripts/verify_balanced_learning.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def feature_rows(kind="correlated"):
    rows = np.array([[1., 0., 0., 0.], [1., .2, 0., 0.],
                     [1., 0., .2, 0.], [1., 0., 0., .2]])
    if kind == "orthogonal":
        rows = np.eye(4)
    elif kind == "signed":
        rows[1] *= -1.
    return rows / np.linalg.norm(rows, axis=1)[:, None]


def produced_case(rows, source, known):
    from drosolearn.cone_geometry import cone_direction
    problem = audit.independent_problem(rows, source, known)
    result = cone_direction(problem["constraints"], problem["objective"])
    return result


@pytest.mark.parametrize("kind", ["correlated", "orthogonal", "signed"])
@pytest.mark.parametrize("source", range(4))
@pytest.mark.parametrize("pattern", ["source_only", "partial", "full"])
def test_independent_dual_reoptimization_certifies_frozen_helper(kind, source, pattern):
    rows = feature_rows(kind)
    known = np.array([1, 1, -1, -1])
    if pattern != "full":
        known[(np.arange(4) != source) & ((np.arange(4) % 2 != source % 2) if pattern == "partial"
                                        else np.ones(4, bool))] = 0
    case = produced_case(rows, source, known)
    record = audit.certify_balanced(rows, source, known, case["direction"],
                                   case["projected_objective"], case["multipliers"])
    assert record["certificate"][0] == pytest.approx(record["optimum"], abs=1e-8)
    assert record["certificate"][4:].max() <= 1e-8


@pytest.mark.parametrize("source", range(4))
def test_unknown_destination_has_no_constraint_or_objective_contribution(source):
    rows = feature_rows()
    known = np.zeros(4, int)
    known[source] = -1
    problem = audit.independent_problem(rows, source, known)
    assert np.array_equal(problem["objective"], rows[source])
    assert np.array_equal(problem["known_indices"], [source])
    assert not problem["helpful_indices"].size
    assert np.all(problem["constraints"][known == 0] == 0.)


def test_helpful_mean_uses_actual_sign_and_negative_gram_pair():
    rows = feature_rows("signed")
    known = np.array([1, -1, -1, 0])
    problem = audit.independent_problem(rows, 0, known)
    assert np.array_equal(problem["helpful_indices"], [1])
    assert np.allclose(problem["objective"], .5 * rows[0] - .5 * rows[1])
    assert np.all(problem["constraints"][3] == 0.)


@pytest.mark.parametrize("known", [np.zeros(3, int), np.array([1, 2, 0, 0]),
                                  np.array([1., 0., 0., 0.]), np.zeros(4, int)])
def test_bad_or_unknown_source_state_is_not_secretly_completed(known):
    with pytest.raises(ValueError):
        audit.independent_problem(feature_rows(), 0, known)


@pytest.mark.parametrize("field", ["direction", "projection", "multipliers"])
def test_corrupt_certificate_is_rejected(field):
    rows, known = feature_rows(), np.array([1, 1, -1, -1])
    case = produced_case(rows, 0, known)
    values = {"direction": case["direction"].copy(), "projection": case["projected_objective"].copy(),
              "multipliers": case["multipliers"].copy()}
    values[field][0] += .03
    with pytest.raises(ValueError):
        audit.certify_balanced(rows, 0, known, **values)


def test_unknown_destination_cannot_carry_dual_information():
    rows, known = feature_rows(), np.array([1, 0, -1, 0])
    case = produced_case(rows, 0, known)
    mu = case["multipliers"].copy()
    mu[1] = 1e-12
    with pytest.raises(ValueError, match="unknown-free"):
        audit.certify_balanced(rows, 0, known, case["direction"], case["projected_objective"], mu)


def test_direction_with_valid_norm_but_suboptimal_objective_rejected():
    rows, known = np.eye(4), np.array([1, 0, 0, 0])
    with pytest.raises(ValueError):
        audit.certify_balanced(rows, 0, known, .9 * rows[0], rows[0], np.zeros(4))


def row_fixture():
    return [{"topology": "real", "trial": 1, "probability": .5, "known": 0},
            {"topology": "real", "trial": 2, "probability": .51, "known": 1}]


def test_vectorized_csv_accepts_decimal_strings_and_exact_column_order():
    wanted = row_fixture()
    actual = [{key: str(value) for key, value in row.items()} for row in wanted]
    audit.compare_rows(actual, wanted)


def test_unknown_other_harm_uses_probability_not_logit_and_excludes_source():
    before = np.zeros(4)
    after = np.array([-1., -1., 1., -1.])
    record = audit.independent_unknown_harm(before, after, 0, np.zeros(4, int))
    assert np.array_equal(record["unknown_other_mask"], [False, True, True, True])
    assert record["unknown_other_exposure_count"] == 3
    assert record["unknown_other_negative_correct_probability_sum"] == pytest.approx(
        2. * (.5 - 1. / (1. + np.exp(1.))))
    assert record["unknown_other_negative_correct_probability_sum"] != pytest.approx(2.)


def test_current_reward_write_cannot_change_unknown_other_evaluation_mask():
    before, after = np.zeros(4, int), np.array([1, 0, 0, 0])
    args = np.zeros(4), np.ones(4), 0
    a = audit.independent_unknown_harm(*args, before)
    b = audit.independent_unknown_harm(*args, after)
    assert np.array_equal(a["unknown_other_mask"], b["unknown_other_mask"])
    assert a["unknown_other_negative_correct_probability_sum"] == b["unknown_other_negative_correct_probability_sum"]


def test_known_other_destinations_do_not_enter_unknown_harm_endpoint():
    record = audit.independent_unknown_harm(np.zeros(4), -np.ones(4), 2, np.array([1, 0, -1, 0]))
    assert np.array_equal(record["unknown_other_mask"], [False, True, False, True])
    assert record["unknown_other_exposure_count"] == 2
    assert record["unknown_other_negative_correct_probability_sum"] == pytest.approx(.5 - audit.sigmoid(-1.))


@pytest.mark.parametrize("mutation", ["drop_row", "reorder_row", "extra_column", "reorder_column",
                                       "identity", "fractional_integer", "numeric", "nan"])
def test_vectorized_csv_rejects_integrity_corruption(mutation):
    wanted = row_fixture()
    actual = copy.deepcopy(wanted)
    if mutation == "drop_row": actual.pop()
    elif mutation == "reorder_row": actual.reverse()
    elif mutation == "extra_column": actual[0]["extra"] = 0
    elif mutation == "reorder_column": actual[0] = dict(reversed(tuple(actual[0].items())))
    elif mutation == "identity": actual[0]["topology"] = "ann"
    elif mutation == "fractional_integer": actual[0]["trial"] = 1.1
    elif mutation == "numeric": actual[1]["probability"] = .52
    elif mutation == "nan": actual[0]["probability"] = np.nan
    with pytest.raises(ValueError): audit.compare_rows(actual, wanted)


def fixture_config():
    from drosolearn.balanced_learning import SCIENCE
    cfg = copy.deepcopy(SCIENCE)
    cfg.update(scope="integration_test", model_seeds=[90085, 90086], training_trials=8,
               checkpoints=[0, 4, 8], population_pass_count=2, positive_seed_count=2,
               unknown_nonworse_seed_count=2)
    return cfg


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


@pytest.fixture(scope="module")
def producer_fixture(tmp_path_factory):
    from drosolearn import balanced_learning as producer
    from drosolearn.graph import Projection
    root = tmp_path_factory.mktemp("balanced_invented_replay")
    cfg, raw = fixture_config(), tiny_graph()
    n = raw["n_kc"]
    graph = Projection(raw["pre"], raw["post"], raw["counts"], n, n,
                       [str(i) for i in range(n)], [str(i) for i in range(n)], {})
    catalog = root / audit.INPUT_CATALOG
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"study_id": "representation_update_v1",
        "representations": list(audit.REPRESENTATIONS), "catalog": {
            key: value.tolist() for key, value in audit.prior.independent_catalog_arrays().items()}}), encoding="utf8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(producer, "load_projection", lambda _: graph)
        arrays, rows = producer.produce(cfg, root)
    expected, wanted_rows = audit.expected_artifacts(cfg, raw, arrays)
    return cfg, raw, arrays, rows, expected, wanted_rows


def test_complete_current_autonomous_producer_matches_independent_fixture(producer_fixture):
    cfg, _, arrays, rows, expected, wanted_rows = producer_fixture
    audit.validate_science(cfg, allow_fixture=True)
    audit.check_artifacts(arrays, expected, rows, wanted_rows)
    assert len(rows) == 384
    assert arrays["trial_actions"].shape == (3, 2, 4, 2, 8)
    assert arrays["trial_directions"].shape == (3, 2, 4, 2, 8, 16)
    assert all(audit.information_checks(cfg, arrays).values())
    assert np.all(arrays["trial_case_indices"][:, :, :3] == -1)
    assert np.all(arrays["trial_case_indices"][:, :, 3][arrays["trial_actions"][:, :, 3] == 0] == -1)
    assert np.all(arrays["trial_directions"][:, :, 3][arrays["trial_actions"][:, :, 3] == 0] == 0.)
    assert np.max(arrays["trial_head_steps"]) <= .1 + 1e-8


def test_entire_current_producer_summary_matches_independent_actual_replay(producer_fixture):
    from drosolearn.balanced_learning import summarize
    cfg, graph, arrays, rows, _, _ = producer_fixture
    summary = summarize(cfg, arrays, rows)
    result = audit.verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=True)
    assert result["independently_replayed_training_actions"] == 384
    assert result["unique_balanced_states_independently_reoptimized"] == len(arrays["case_keys"])
    assert result["evaluator_hidden_signs_do_not_enter_observed_direction_checked"]
    assert summary["counts"]["internal_update_calls"] == summary["counts"]["value_update_calls"] == 0
    assert summary["counts"]["source_inclusive_unknown_exposures_before_feedback"] - summary["counts"][
        "source_inclusive_unknown_exposures_after_feedback"] == summary["counts"]["auxiliary_memory_discoveries"]


@pytest.mark.parametrize("field", ["inputs", "targets", "schedules", "uniforms", "trial_actions",
    "trial_rewards", "trial_probabilities", "trial_known_before", "trial_known_after", "trial_routing_signs",
    "trial_harm_mask", "trial_desired_influence", "trial_actual_influence", "trial_directions",
    "trial_direction_residuals", "trial_direction_residuals_applicable", "trial_case_indices",
    "trial_unknown_other_mask_before", "trial_unknown_other_mask_after", "trial_correct_probability_delta",
    "trial_aligned_margin_delta", "trial_unknown_other_negative_correct_probability_sum",
    "trial_unknown_other_exposure_count", "trial_memory_changed", "first_discovery_trials",
    "checkpoint_known", "checkpoint_heads", "checkpoint_lookup_probabilities", "g_real_raw_unit_rows",
    "g_real_raw_preflight_direction_count", "w_real_raw_weights", "finalfixed_ann_identity_probe_mbon"])
def test_complete_trajectory_tampering_rejected(producer_fixture, field):
    _, _, original, rows, wanted, wanted_rows = producer_fixture
    changed = copy.deepcopy(original)
    flat = changed[field].reshape(-1)
    flat[-1] = not flat[-1] if flat.dtype.kind == "b" else flat[-1] + (1 if flat.dtype.kind in "iu" else .25)
    with pytest.raises(ValueError): audit.check_artifacts(changed, wanted, rows, wanted_rows)


@pytest.mark.parametrize("field", ["case_keys", "case_directions", "case_projections", "case_multipliers", "case_certificates"])
def test_actual_case_witness_tampering_rejected_by_fresh_independent_replay(producer_fixture, field):
    cfg, raw, original, _, _, _ = producer_fixture
    changed = copy.deepcopy(original)
    changed[field].reshape(-1)[-1] += 1 if changed[field].dtype.kind in "iu" else .1
    with pytest.raises(ValueError): audit.expected_artifacts(cfg, raw, changed)


@pytest.mark.parametrize("key,value", [("methods", list(reversed(audit.METHODS))), ("head_lr", .2),
    ("rng_namespace", 99), ("svd_rcond", 1e-6), ("normalization", "other"), ("effect_threshold", .01),
    ("balanced_own_objective_weight", .6), ("balanced_help_objective_weight", .4),
    ("unknown_harm_endpoint", "negative_logits"), ("budget_convention", "matched_realized_totals")])
def test_independent_fixed_science_choices_cannot_be_relaxed(key, value):
    cfg = fixture_config()
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_fixture_escape_cannot_enter_actual_science():
    cfg = fixture_config()
    audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(cfg)
    cfg["model_seeds"] = [17501]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def isolated_endpoint_fixture(producer_fixture, monkeypatch):
    """Isolate endpoint decision formulas; no fabricated full trajectory is claimed."""
    cfg, _, original, rows, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    information = audit.information_checks(cfg, original)
    monkeypatch.setattr(audit, "information_checks", lambda *_: information)
    arrays["checkpoint_probabilities"][:] = .5
    arrays["trial_unknown_other_negative_correct_probability_sum"][:] = 0.
    return cfg, arrays, rows


def set_primary_final(arrays, method, value, seed=slice(None)):
    arrays["checkpoint_probabilities"][0, 0, method, seed, -1] = np.where(audit.TARGETS, value, 1. - np.asarray(value))


def test_practical_gain_can_pass_without_acquisition(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    set_primary_final(arrays, 3, .7)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_gain"]["supported"] and not summary["H_gain"]["requires_acquisition"]
    assert not summary["H_acquisition"]["supported"]


def test_acquisition_can_pass_without_practical_gain(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    set_primary_final(arrays, 3, .94)
    set_primary_final(arrays, 1, .90)
    set_primary_final(arrays, 2, .90)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_acquisition"]["supported"] and not summary["H_acquisition"]["requires_gain"]
    assert not summary["H_gain"]["supported"]


def test_gain_does_not_replace_unknown_harm_endpoint(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    set_primary_final(arrays, 3, .7)
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 3, :, 0] = .01
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_gain"]["supported"]
    assert not summary["H_unknown"]["supported"]
    assert summary["H_unknown"]["mean_balanced_minus_feedback_damage"] == pytest.approx(.01)


def test_unknown_harm_nonworsening_does_not_replace_gain_or_acquisition(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_unknown"]["supported"] and summary["H_unknown"]["nonworsening_seeds"] == 2
    assert not summary["H_gain"]["supported"] and not summary["H_acquisition"]["supported"]


@pytest.mark.parametrize("control", [1, 2])
def test_gain_requires_both_frozen_controls(producer_fixture, monkeypatch, control):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    set_primary_final(arrays, 3, .7)
    set_primary_final(arrays, control, .69)
    assert not audit.expected_summary(cfg, arrays, rows)["H_gain"]["supported"]


def test_large_mean_effect_cannot_replace_positive_seed_count(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    set_primary_final(arrays, 3, .8)
    set_primary_final(arrays, 1, .85, seed=1)
    summary = audit.expected_summary(cfg, arrays, rows)
    comparison = summary["H_gain"]["endpoint_comparisons"]["isolated"]
    assert comparison["mean_effect"] > .1 and comparison["positive_seeds"] == 1
    assert not comparison["practical_advantage_pass"] and not summary["H_gain"]["supported"]


def test_unknown_harm_mean_does_not_replace_nonworsening_seed_count(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 2, 0, 0] = .1
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 3, 1, 0] = .01
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_unknown"]["mean_balanced_minus_feedback_damage"] < 0.
    assert summary["H_unknown"]["nonworsening_seeds"] == 1
    assert not summary["H_unknown"]["supported"]


def test_unknown_harm_uses_totals_not_posthoc_exposure_denominator(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 2, :, 0] = .01
    arrays["trial_unknown_other_negative_correct_probability_sum"][0, 0, 3, :, 0] = .02
    arrays["trial_unknown_other_exposure_count"][0, 0, 2] = 0
    arrays["trial_unknown_other_exposure_count"][0, 0, 2, :, 0] = 1
    arrays["trial_unknown_other_exposure_count"][0, 0, 3] = 3
    summary = audit.expected_summary(cfg, arrays, rows)
    assert not summary["H_unknown"]["supported"]
    assert summary["H_unknown"]["mean_balanced_minus_feedback_damage"] == pytest.approx(.01)


def test_zero_exposure_descriptive_ratio_is_null(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    arrays["trial_unknown_other_exposure_count"][0, 0, 3] = 0
    summary = audit.expected_summary(cfg, arrays, rows)
    primary = next(x for x in summary["families"] if x["role"] == "primary")
    assert all(x["unknown_other_negative_correct_probability_per_exposure"] is None for x in primary["seeds"])


def test_weak_class_blocks_acquisition_despite_high_average(producer_fixture, monkeypatch):
    cfg, arrays, rows = isolated_endpoint_fixture(producer_fixture, monkeypatch)
    set_primary_final(arrays, 3, [.99, .99, .65, .65])
    summary = audit.expected_summary(cfg, arrays, rows)
    assert summary["H_gain"]["supported"] and not summary["H_acquisition"]["supported"]


@pytest.mark.parametrize("field,value", [("trial_head_steps", .101), ("trial_cumulative_head_path_l2", .81),
    ("trial_cumulative_head_squared_l2", .081), ("trial_expected_head_steps", .2),
    ("trial_direction_residuals", .01)])
def test_nonprimary_numeric_failure_is_fatal_not_scientific_miss(producer_fixture, field, value):
    cfg, _, original, rows, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    arrays[field][2, 1, 2, 0, -1] = value
    with pytest.raises(ValueError): audit.expected_summary(cfg, arrays, rows)


def test_nonshared_known_damage_is_fatal(producer_fixture):
    cfg, _, original, rows, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    positions = np.argwhere(arrays["trial_known_after"][:, :, 3] != 0)
    ti, ri, si, trial, dest = map(int, positions[0])
    arrays["trial_aligned_margin_delta"][ti, ri, 3, si, trial, dest] = -.01
    with pytest.raises(ValueError): audit.expected_summary(cfg, arrays, rows)


def test_direct_memory_choice_policy_fails_even_with_matched_action_uniforms(producer_fixture):
    cfg, _, original, _, _, _ = producer_fixture
    arrays = copy.deepcopy(original)
    arrays["trial_probabilities"][0, 0, 3] = .999
    arrays["trial_actions"][0, 0, 3] = arrays["uniforms"] < .999
    assert not audit.information_checks(cfg, arrays)["head_policy_does_not_directly_lookup_memory"]
