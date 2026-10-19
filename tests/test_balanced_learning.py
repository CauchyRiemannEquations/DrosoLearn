"""Autonomous balanced updates on invented geometry, never the science cohort."""
from copy import deepcopy

import numpy as np
import pytest

from drosolearn.selective_geometry import selective_geometry


@pytest.fixture
def geometry():
    return selective_geometry(np.array([[1., 0., 0., 0., .4, .2],
        [0., 1., 0., 0., .4, .2], [0., 0., 1., 0., .4, .2],
        [0., 0., 0., 1., .4, .2]]))


def learner(geometry, method="balanced"):
    from drosolearn.balanced_learning import BalancedHead
    return BalancedHead(geometry, method, .1)


@pytest.mark.parametrize("method", ["shared", "isolated", "feedback_selective", "balanced"])
def test_no_method_accepts_a_privileged_target_table(geometry, method):
    from drosolearn.balanced_learning import BalancedHead
    with pytest.raises((ValueError, TypeError)):
        BalancedHead(geometry, method, .1, np.array([1, 1, -1, -1]))


@pytest.mark.parametrize("method", ["shared", "isolated", "feedback_selective", "balanced"])
def test_initial_policy_and_observed_memory_are_separate_and_empty(geometry, method):
    actor = learner(geometry, method)
    assert np.array_equal(actor.known, np.zeros(4, np.int8))
    assert np.array_equal(actor.head, np.zeros(6))
    assert np.array_equal(actor.probabilities(), np.full(4, .5))
    assert actor.calls == actor.discoveries == actor.memory_observation_calls == 0


def test_nogo_reveals_nothing_and_does_not_move_balanced_output(geometry):
    actor = learner(geometry)
    result = actor.observe(2, 0, 0)
    assert not actor.known.any() and not actor.head.any()
    assert not result["direction"].any()
    assert result["head_step"] == result["update_factor"] == 0.
    assert actor.discoveries == actor.memory_observation_calls == 0
    assert actor.calls == 1


def test_current_obtained_go_reward_is_known_before_direction(geometry):
    actor = learner(geometry)
    actor.observe(0, 1, 1)
    result = actor.observe(2, 1, -1)
    assert result["known_before"].tolist() == [1, 0, 0, 0]
    assert result["known_after"].tolist() == [1, 0, -1, 0]
    assert result["routing_signs"].tolist() == [1, 0, -1, 0]
    assert result["memory_changed"] == 1
    actual = geometry["unit_rows"] @ (result["update_factor"] * result["direction"])
    assert actual[0] >= -1e-8 and -actual[2] >= -1e-8


@pytest.mark.parametrize("source,reward", [(0, 1), (1, -1), (2, 1), (3, -1)])
def test_first_reward_with_only_source_known_uses_safe_own_direction(geometry, source, reward):
    actor = learner(geometry)
    result = actor.observe(source, 1, reward)
    assert np.allclose(result["direction"], geometry["unit_rows"][source], atol=1e-12)
    assert result["known_after"][source] == reward
    assert np.count_nonzero(result["known_after"]) == 1


def test_all_observed_destinations_are_safe_under_actual_negative_source_update(geometry):
    actor = learner(geometry)
    signs = np.array([1, 1, -1, -1])
    for source, reward in enumerate(signs):
        actor.observe(source, 1, int(reward))
    before = actor.logits().copy()
    result = actor.observe(2, 1, -1)
    assert result["update_factor"] < 0.
    assert np.all(signs * (actor.logits() - before) >= -1e-8)
    assert np.allclose(actor.logits() - before,
        result["update_factor"] * (geometry["unit_rows"] @ result["direction"]), atol=1e-12)


def test_safe_observed_direction_can_still_harm_an_unknown_problem(geometry):
    actor = learner(geometry)
    before = actor.head.copy()
    result = actor.observe(0, 1, 1)
    measured = effects(geometry["unit_rows"], before, actor.head, 0,
        result["known_before"], result["known_after"])
    assert measured["aligned_margin_delta"][0] > 0.
    assert measured["aligned_margin_delta"][2] < 0.
    assert measured["unknown_other_negative_correct_probability_sum"] > 0.
    assert measured["unknown_other_negative_margin_sum"] > 0.
    assert result["known_after"].tolist() == [1, 0, 0, 0]


def test_nogo_skips_balanced_solver_after_prior_learning(geometry, monkeypatch):
    import drosolearn.balanced_learning as model
    actor = learner(geometry)
    actor.observe(0, 1, 1)
    head, known = actor.head.copy(), actor.known.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("NoGo must not run a balanced cone solver")
    monkeypatch.setattr(model, "cone_direction", forbidden)
    result = actor.observe(2, 0, 0)
    assert np.array_equal(actor.head, head) and np.array_equal(actor.known, known)
    assert not result["direction"].any()


def test_repeated_identical_observed_state_reuses_only_its_own_certificate(geometry):
    actor = learner(geometry)
    first = actor.observe(0, 1, 1)
    second = actor.observe(0, 1, 1)
    assert first["case_index"] == second["case_index"] == 0
    assert len(actor.cases) == 1 and actor.solver_requests == 2
    actor.observe(2, 1, -1)
    third = actor.observe(0, 1, 1)
    assert third["case_index"] != first["case_index"]
    assert np.array_equal(actor.cases[third["case_index"]]["key"], [0, 1, 0, -1, 0])


def test_balanced_has_separate_certificate_and_does_not_claim_selective_raw_ratios(geometry):
    actor = learner(geometry)
    result = actor.observe(0, 1, 1)
    assert result["direction_residuals_applicable"] is False
    assert not result["direction_residuals"].any()
    proof = actor.cases[result["case_index"]]
    assert proof["certificate"].shape == (8,)
    assert np.max(proof["certificate"][4:]) <= 1e-8
    assert np.linalg.norm(proof["direction"]) <= 1. + 1e-8
    assert np.all(proof["multipliers"] >= 0.)
    assert np.all(proof["multipliers"][actor.known == 0] == 0.)


def test_repeated_rewards_are_observations_but_not_new_discoveries(geometry):
    actor = learner(geometry)
    first = actor.observe(1, 1, 1)
    second = actor.observe(1, 1, 1)
    assert first["memory_changed"] == 1 and second["memory_changed"] == 0
    assert actor.discoveries == 1 and actor.memory_observation_calls == 2


def test_contradictory_observation_fails_without_head_or_memory_mutation(geometry):
    actor = learner(geometry)
    actor.observe(1, 1, 1)
    before = (actor.head.copy(), actor.known.copy(), actor.calls,
              actor.discoveries, actor.memory_observation_calls)
    with pytest.raises(ValueError):
        actor.observe(1, 1, -1)
    assert np.array_equal(actor.head, before[0]) and np.array_equal(actor.known, before[1])
    assert (actor.calls, actor.discoveries, actor.memory_observation_calls) == before[2:]


@pytest.mark.parametrize("action,reward", [(0, 1), (0, -1), (1, 0), (1, 2), (2, 1)])
def test_counterfactual_or_invalid_reward_is_rejected(geometry, action, reward):
    actor = learner(geometry)
    with pytest.raises(ValueError):
        actor.observe(0, action, reward)
    assert not actor.known.any() and not actor.head.any()


@pytest.mark.parametrize("index", [-1, 4, .5, "1"])
def test_observation_source_index_must_be_valid(geometry, index):
    actor = learner(geometry)
    with pytest.raises(ValueError):
        actor.observe(index, 1, 1)


@pytest.mark.parametrize("method", ["shared", "isolated"])
def test_controls_record_rewards_but_ignore_them_when_choosing_direction(geometry, method):
    actor = learner(geometry, method)
    actor.observe(0, 1, 1)
    result = actor.observe(2, 1, -1)
    assert actor.known.tolist() == [1, 0, -1, 0]
    assert not result["routing_signs"].any()
    expected = geometry["shared_directions"][2] if method == "shared" else geometry["isolated_directions"][2]
    assert np.array_equal(result["direction"], expected)


@pytest.mark.parametrize("method", ["shared", "isolated", "feedback_selective", "balanced"])
def test_obtained_reward_record_is_not_directly_used_as_policy(geometry, method):
    actor = learner(geometry, method)
    for source, reward in enumerate([1, 1, -1, -1]):
        actor.observe(source, 1, reward)
    probability = actor.probabilities()
    assert np.all((probability > 0.) & (probability < 1.))
    assert np.allclose(probability, 1. / (1. + np.exp(-(geometry["unit_rows"] @ actor.head))))
    assert not np.array_equal(probability, (actor.known.astype(float) + 1.) / 2.)


@pytest.mark.parametrize("method", ["shared", "isolated", "feedback_selective", "balanced"])
def test_actual_updates_obey_common_movement_ceiling_and_leave_geometry_fixed(geometry, method):
    frozen = deepcopy(geometry)
    actor = learner(geometry, method)
    path = energy = 0.
    for trial in range(16):
        source = trial % 4
        reward = 1 if source < 2 else -1
        before = actor.head.copy()
        result = actor.observe(source, 1, reward)
        step = np.linalg.norm(actor.head - before)
        path += step
        energy += step ** 2
        assert step <= .1 + 1e-8
        assert step == pytest.approx(abs(result["update_factor"]) * np.linalg.norm(result["direction"]), abs=1e-12)
        assert np.linalg.norm(result["direction"]) <= 1. + 1e-8
    assert path <= 1.6 + 1e-8 and energy <= .16 + 1e-8
    assert np.linalg.norm(actor.head) <= path + 1e-8
    for key in ("unit_rows", "pseudoinverse", "base_influences", "shared_directions", "isolated_directions"):
        assert np.array_equal(geometry[key], frozen[key])


@pytest.mark.parametrize("key,value", [("head_lr", .2), ("methods", ["balanced"]),
    ("effect_threshold", .001), ("model_seeds", [17501]), ("training_trials", 800),
    ("population_pass_count", 1), ("positive_seed_count", 1)])
def test_preregistered_science_design_cannot_be_relaxed(key, value):
    from drosolearn.balanced_learning import SCIENCE, validate_config
    cfg = deepcopy(SCIENCE)
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def effects(rows, before, after, source, known_before, known_after):
    from drosolearn.balanced_learning import trial_effects
    return trial_effects(rows, np.asarray(before, float), np.asarray(after, float),
        np.array([1, 1, -1, -1]), source,
        np.asarray(known_before, np.int8), np.asarray(known_after, np.int8))


def test_probability_harm_is_actual_same_state_sigmoid_difference():
    rows = np.eye(4)
    before, after = np.array([.1, -.2, .3, -.4]), np.array([.3, -.5, .1, -.7])
    result = effects(rows, before, after, 0, [0, 0, 0, 0], [1, 0, 0, 0])
    signs = np.array([1, 1, -1, -1])
    expected = 1. / (1. + np.exp(-signs * after)) - 1. / (1. + np.exp(-signs * before))
    assert np.allclose(result["correct_probability_delta"], expected, atol=1e-14)
    assert np.allclose(result["aligned_margin_delta"], signs * (after - before), atol=1e-14)
    assert result["unknown_other_negative_correct_probability_sum"] == pytest.approx(-expected[1])
    assert result["unknown_other_positive_correct_probability_sum"] == pytest.approx(expected[2] + expected[3])
    assert result["unknown_other_negative_margin_sum"] == pytest.approx(.3)
    assert result["unknown_other_exposure_count"] == 3


@pytest.mark.parametrize("source", range(4))
def test_current_reward_cannot_change_unknown_other_mask(source):
    known_before = np.zeros(4, np.int8)
    known_before[(source + 1) % 4] = 1
    known_after = known_before.copy()
    known_after[source] = 1 if source < 2 else -1
    result = effects(np.eye(4), np.zeros(4), np.zeros(4), source, known_before, known_after)
    expected = known_before == 0
    expected[source] = False
    assert np.array_equal(result["unknown_other_mask_before"], expected)
    assert np.array_equal(result["unknown_other_mask_after"], expected)
    assert result["unknown_other_exposure_count"] == 2


def test_fully_observed_case_has_no_unknown_exposures_even_if_known_damage_exists():
    result = effects(np.eye(4), np.zeros(4), [0., -.2, .2, 0.], 0,
        [1, 1, -1, -1], [1, 1, -1, -1])
    assert not result["unknown_other_mask_before"].any()
    assert result["unknown_other_exposure_count"] == 0
    assert result["unknown_other_negative_correct_probability_sum"] == 0.
    assert result["unknown_other_negative_margin_sum"] == 0.
    assert result["correct_probability_delta"][1] < 0.


def test_zero_update_has_zero_effect_without_erasing_existing_unknown_exposure():
    result = effects(np.eye(4), [.1, .2, .3, .4], [.1, .2, .3, .4], 1,
        [0, 0, -1, 0], [0, 0, -1, 0])
    assert not result["correct_probability_delta"].any()
    assert not result["aligned_margin_delta"].any()
    assert result["unknown_other_exposure_count"] == 2
    assert result["unknown_other_negative_correct_probability_sum"] == 0.


def test_harm_is_not_cancelled_by_larger_help_at_another_unknown_destination():
    result = effects(np.eye(4), np.zeros(4), [0., -.1, -.5, 0.], 0,
        [1, 0, 0, 0], [1, 0, 0, 0])
    delta = result["correct_probability_delta"]
    assert delta[2] > -delta[1] > 0.
    assert result["unknown_other_negative_correct_probability_sum"] == pytest.approx(-delta[1])
    assert result["unknown_other_positive_correct_probability_sum"] == pytest.approx(delta[2])


def test_balanced_objective_uses_mean_known_help_and_not_unknown_rewards(geometry):
    from drosolearn.balanced_learning import observed_problem
    rows = geometry["unit_rows"]
    known = np.array([1, 1, 0, 1], np.int8)
    constraints, objective, indices, helpful = observed_problem(rows, 0, known)
    assert np.array_equal(indices, [0, 1, 3]) and np.array_equal(helpful, [1, 3])
    assert np.allclose(objective, .5 * rows[0] + .5 * rows[[1, 3]].mean(axis=0))
    assert np.array_equal(constraints, rows[[0, 1, 3]])
    assert not np.allclose(objective, .5 * rows[0] + .5 * rows[[1, 3]].sum(axis=0))


def test_negative_gram_and_opposite_sign_can_still_be_known_help(geometry):
    from drosolearn.balanced_learning import observed_problem
    rows = geometry["unit_rows"].copy()
    rows[1] *= -1.
    known = np.array([1, -1, -1, 0], np.int8)
    constraints, objective, indices, helpful = observed_problem(rows, 0, known)
    assert np.array_equal(helpful, [1])
    assert np.allclose(objective, .5 * rows[0] - .5 * rows[1])
    assert np.array_equal(constraints, known[indices, None] * rows[indices])


@pytest.mark.parametrize("known", [np.array([0, 1, -1, 0]), np.array([1, 2, 0, 0]),
    np.array([1., 0., 0., 0.]), np.array([1, 0, 0])])
def test_unobserved_source_or_invalid_observed_table_is_not_filled_from_hidden_truth(geometry, known):
    from drosolearn.balanced_learning import observed_problem
    with pytest.raises(ValueError):
        observed_problem(geometry["unit_rows"], 0, known)
