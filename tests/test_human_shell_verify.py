"""Synthetic, independently calculated tests; no human data or scientific seeds."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "human_shell_independent_verifier", Path(__file__).resolve().parents[1] / "scripts/verify_human_shell.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def graph():
    return {"pre": np.array([0, 1, 2, 3, 0, 2]), "post": np.array([0, 0, 1, 1, 2, 2]),
            "counts": np.array([1., 2., 3., 1., 2., 1.]), "n_kc": 4, "n_mbon": 3}


def short_record(actions, contents):
    return {"id": "A9001", "shell": np.full(len(actions), 3), "action": np.array(actions),
            "content": np.array(contents), "state": np.ones(len(actions), int)}


def behavior_fixture():
    shells, states, actions = [], [], []
    for phase in range(4):
        state = 1 + phase % 2
        for exposure in range(6):
            for shell in (3, 4):
                correct = int(audit.SEASONS[shell][state - 1] == "R")
                shells.append(shell); states.append(state)
                actions.append(correct if exposure >= 3 else 1 - correct)
    return {"shell": np.array(shells), "state": np.array(states)}, np.array(actions)


def test_first_prediction_precedes_any_feedback():
    policy = audit.AuditPolicy("linear", 19, .2)
    tape = audit.replay(policy, short_record([1, 1], [1, 1]), "conditional", 19)
    assert tape["p_go"][0] == .5
    assert tape["p_go"][1] == pytest.approx(1 / (1 + np.exp(-.1)))


def test_signed_punishment_reduces_go():
    policy = audit.AuditPolicy("linear", 19, .2)
    policy.update(3, 1, -1)
    assert policy.probability(3) < .5
    assert policy.probability(2) == .5


@pytest.mark.parametrize("condition", audit.CONDITIONS)
def test_no_go_never_gets_hidden_content(condition):
    policy1 = audit.AuditPolicy(condition, 19, .1, graph())
    policy2 = audit.AuditPolicy(condition, 19, .1, graph())
    a = audit.replay(policy1, short_record([0, 0, 1], [1, 0, 1]), "conditional", 19)
    b = audit.replay(policy2, short_record([0, 0, 1], [0, 1, 1]), "conditional", 19)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(policy1.head, policy2.head)
    np.testing.assert_array_equal(policy1.weights, policy2.weights)


def test_network_first_update_uses_old_zero_head():
    policy = audit.AuditPolicy("plastic", 19, .2, graph())
    initial = policy.weights.copy()
    head_step, internal_step, value_step = policy.update(3, 1, 1)
    assert head_step > 0 and internal_step == 0
    np.testing.assert_array_equal(policy.weights, initial)
    policy.update(3, 1, 1)
    assert np.linalg.norm(policy.weights - initial) > 0
    assert policy.internal_updates == 2


@pytest.mark.parametrize("reward", [-1, 0, 1])
def test_no_go_signed_outcome_contract(reward):
    policy = audit.AuditPolicy("linear", 19, .2)
    if reward:
        with pytest.raises(ValueError, match="NoGo"):
            policy.update(3, 0, reward)
    else:
        assert policy.update(3, 0, reward) == (0., 0., 0.)


def test_fixed_internal_weights_remain_unchanged():
    policy = audit.AuditPolicy("fixed", 19, .2, graph())
    old = policy.weights.copy()
    for reward in [1, -1, 1, 0, 1]:
        policy.update(3, 1, reward)
    np.testing.assert_array_equal(policy.weights, old)
    assert policy.head_updates == 5 and policy.internal_updates == 0


def test_q_signed_reward_and_unobserved_action():
    policy = audit.AuditPolicy("q", 19, .2)
    policy.update(3, 1, -1)
    assert policy.values[2, 1] == -.2
    assert policy.values[2, 0] == 0
    assert policy.value_updates == 1 and policy.head_updates == 0
    assert policy.probability(3) == pytest.approx(1 / (1 + np.exp(.8)))


def test_persistence_tracks_same_shell_and_has_no_reward_learning():
    policy = audit.AuditPolicy("persistence", 19, .2)
    assert policy.probability(3) == .5
    policy.update(3, 1, -1)
    assert policy.probability(3) == .9 and policy.probability(4) == .5
    policy.update(3, 0, 0)
    assert policy.probability(3) == .1
    assert policy.head_updates == policy.internal_updates == 0


def test_free_actions_use_own_feedback_not_human_actions():
    one = short_record([0] * 8, [1] * 8)
    two = short_record([1] * 8, [1] * 8)
    a = audit.replay(audit.AuditPolicy("linear", 19, .2), one, "autonomous", 19)
    b = audit.replay(audit.AuditPolicy("linear", 19, .2), two, "autonomous", 19)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(a["reward"], a["action"])


def test_rp_behavior_hand_values():
    record, actions = behavior_fixture()
    np.testing.assert_allclose(audit.behavioral_metrics(record, actions), [1, 0, 1], atol=0)


def test_behavior_uses_shell_exposure_not_global_three_trials():
    record, actions = behavior_fixture()
    order = np.argsort(record["shell"], kind="stable")
    regrouped = {key: value[order] for key, value in record.items()}
    np.testing.assert_array_equal(audit.behavioral_metrics(record, actions),
                                  audit.behavioral_metrics(regrouped, actions[order]))


def test_behavior_rejects_overlapping_windows():
    record, actions = behavior_fixture()
    keep = np.arange(len(actions)) != 0
    shortened = {key: value[keep] for key, value in record.items()}
    with pytest.raises(ValueError, match="Overlapping"):
        audit.behavioral_metrics(shortened, actions[keep])


def test_brier_is_not_computed_from_log_score_clipping():
    nll, brier = audit.score_choices([0., 1.], [1, 0])
    assert brier == 1.
    assert np.isfinite(nll) and nll > 20


def test_scoring_constant_half_hand_values():
    nll, brier = audit.score_choices([.5, .5, .5], [1, 0, 1])
    assert nll == pytest.approx(np.log(2))
    assert brier == .25


def test_cross_bootstrap_keeps_people_paired_and_seed_shared():
    human = np.array([.1, .5, .9])
    model = human[:, None] + np.array([-.1, .1])[None, :]
    people = np.array([[0, 0, 1], [2, 2, 0]])
    seeds = np.array([[0, 0], [1, 0]])
    delta, _, _ = audit.crossed_draws(human, model, people, seeds)
    np.testing.assert_allclose(delta, [-.1, 0], atol=1e-15)


def test_resampling_preserves_exact_generator_order():
    people, seeds = audit.resampling_indices(3, 2, samples=4, seed=87)
    generator = np.random.default_rng(87)
    np.testing.assert_array_equal(people, generator.integers(0, 3, (4, 3)))
    np.testing.assert_array_equal(seeds, generator.integers(0, 2, (4, 2)))


@pytest.mark.parametrize("bounds,label", [
    ([[-.09, -.04, .01], [.09, .02, .08]], "supported"),
    ([[-.11, -.04, .01], [.09, .02, .08]], "inconclusive"),
    ([[.11, -.04, .01], [.12, .02, .08]], "different"),
    ([[-.10, -.04, .01], [.09, .02, .08]], "supported"),
])
def test_all_three_equivalence_intersection(bounds, label):
    assert audit.equivalence_label(bounds) == label


def test_graph_controls_are_mask_only_and_preserve_required_counts():
    original = graph()
    rewired = audit.transform_graph(original, "rewired", 37)
    assert len(rewired["pre"]) == len(original["pre"])
    np.testing.assert_array_equal(np.bincount(rewired["pre"], minlength=4),
                                  np.bincount(original["pre"], minlength=4))
    np.testing.assert_array_equal(rewired["post"], original["post"])
    np.testing.assert_array_equal(rewired["counts"], original["counts"])
    np.testing.assert_array_equal(original["pre"], graph()["pre"])
    ann = audit.transform_graph(original, "ann", 38)
    assert len(set(zip(ann["pre"], ann["post"]))) == len(ann["pre"])


def test_synthetic_producer_and_independent_arithmetic_contract():
    """A short artificial experiment checks file-axis and temporal contracts."""
    from drosolearn.graph import Projection
    from drosolearn.human_shell import fit, produce, summarize

    g = graph()
    projection = Projection(g["pre"], g["post"], g["counts"], 4, 3,
                            [str(i) for i in range(4)], [str(i) for i in range(3)], {"scope": "synthetic"})
    record, actions = behavior_fixture()
    rows = []
    for index, (shell, state, action) in enumerate(zip(record["shell"], record["state"], actions)):
        season = audit.SEASONS[int(shell)][int(state) - 1]
        rows.append({"trial": index + 1, "shell": int(shell), "state": int(state),
                     "content": 1 if season == "R" else 2, "action": int(action)})
    calibration = {"A9001": rows, "A9002": rows}
    cfg = {"conditions": list(audit.CONDITIONS), "adaptive_conditions": list(audit.CONDITIONS[:8]),
           "calibration_ids": ["A9001", "A9002"], "test_ids": ["A9003", "A9004"],
           "calibration_seeds": [43], "test_seeds": [47, 48], "rates": [.02, .07],
           "trials_per_person": len(rows), "primary_condition": "plastic", "bootstrap_repeats": 17,
           "bootstrap_seed": 79, "equivalence_margin": .1, "prediction_margin": .01, "learning_gain_margin": .05}
    selection, calibration_arrays, _ = fit(calibration, cfg, projection)
    scores = np.empty((8, 2, 2))
    for ci, condition in enumerate(audit.CONDITIONS[:8]):
        kind = "rewired" if condition.startswith("rewired") else "ann" if condition.startswith("ann") else "real"
        transformed = audit.transform_graph(g, kind, 43 + (110000 if kind == "rewired" else 120000 if kind == "ann" else 0))
        for ri, rate in enumerate(cfg["rates"]):
            for pi, pid in enumerate(cfg["calibration_ids"]):
                rec = {"id": pid, "shell": record["shell"], "state": record["state"],
                       "action": actions, "content": np.array([row["content"] for row in rows])}
                policy = audit.AuditPolicy(condition, 43, rate, transformed, selection["calibration_bias"])
                tape = audit.replay(policy, rec, "conditional", 43)
                np.testing.assert_allclose(tape["p_go"], calibration_arrays["probabilities"][ci, ri, 0, pi], rtol=0, atol=1e-13)
                scores[ci, ri, pi] = audit.score_choices(tape["p_go"], actions)[0]
        chosen = min(zip(scores[ci].mean(axis=1), cfg["rates"]))[1]
        assert selection["rates"][condition] == chosen
    test_records = {"A9003": rows, "A9004": rows}
    arrays, measurements, witnesses = produce(test_records, cfg, projection, selection)
    independent_records = {}
    behavior = np.empty((11, 2, 2, 3)); nll = np.empty((11, 2, 2)); brier = np.empty((11, 2, 2))
    for pi, pid in enumerate(cfg["test_ids"]):
        rec = {"id": pid, "shell": record["shell"], "state": record["state"],
               "action": actions, "content": np.array([row["content"] for row in rows])}
        independent_records[pid] = rec
        for si, seed in enumerate(cfg["test_seeds"]):
            for ci, condition in enumerate(audit.CONDITIONS):
                kind = "rewired" if condition.startswith("rewired") else "ann" if condition.startswith("ann") else "real"
                transformed = audit.transform_graph(g, kind, seed + (110000 if kind == "rewired" else 120000 if kind == "ann" else 0))
                for mi, mode in enumerate(("conditional", "autonomous")):
                    policy = audit.AuditPolicy(condition, seed, selection["rates"].get(condition, cfg["rates"][0]),
                                               transformed, selection["calibration_bias"])
                    initial = policy.witness()
                    tape = audit.replay(policy, rec, mode, seed)
                    for key, field in (("probabilities", "p_go"), ("actions", "action"), ("rewards", "reward"),
                                       ("head_steps", "head_step"), ("internal_steps", "internal_step"), ("value_steps", "value_step")):
                        np.testing.assert_allclose(arrays[key][mi, ci, si, pi], tape[field], rtol=0, atol=1e-13)
                    if mode == "conditional":
                        nll[ci, pi, si], brier[ci, pi, si] = audit.score_choices(tape["p_go"], actions)
                    else:
                        behavior[ci, pi, si] = audit.behavioral_metrics(rec, tape["action"])
                    if pi == 0:
                        for stage, witness in (("initial", initial), ("final", policy.witness())):
                            for field, value in witness.items():
                                np.testing.assert_allclose(witnesses[f"{mode}_{condition}_{stage}_{field}"][si], value,
                                                           rtol=0, atol=1e-13)
    expected = audit.expected_summary(cfg, independent_records, behavior, nll, brier)
    summary = summarize(test_records, cfg, arrays, measurements)
    audit.compare_tree(summary, expected, "Synthetic entire summary", subset=True)
