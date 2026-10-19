"""Synthetic design checks; never reads actual human observations."""
import copy
import numpy as np
import pytest

from drosolearn.human_shell import (BEHAVIOR_METRICS, _trajectory, acquired_reward,
                                   behavior_metrics, fit, loss, produce, summarize)
from drosolearn.human_shell_models import CONDITIONS, ADAPTIVE, SimplePolicy
from test_human_shell_models import small_graph


def synthetic_rows():
    rows = []
    for exposure in range(40):
        for shell in range(1, 7):
            if shell not in (3, 4) and exposure >= 34:
                continue
            state = 1 + (exposure // 10) % 2
            action = int((shell + exposure) % 3 != 0)
            content = 1 if shell in (1, 3) else 2
            rows.append({"trial": len(rows) + 1, "shell": shell, "state": state,
                         "content": content, "action": action})
    assert len(rows) == 216
    return rows


def synthetic_config():
    return {"conditions": list(CONDITIONS), "adaptive_conditions": list(ADAPTIVE),
            "calibration_ids": ["A9001"], "test_ids": ["A9002", "A9003"],
            "calibration_seeds": [31], "test_seeds": [32], "rates": [.1],
            "trials_per_person": 216, "bootstrap_repeats": 20, "bootstrap_seed": 33,
            "equivalence_margin": .1, "prediction_margin": .01, "learning_gain_margin": .05}


@pytest.mark.parametrize("content,reward", [(0, 0), (1, 1), (2, -1)])
def test_counterfactual_trace_mask(content, reward):
    assert acquired_reward(0, content) == 0
    assert acquired_reward(1, content) == reward


def test_prediction_precedes_same_trial_update():
    rows = [{"shell": 1, "content": 1, "action": 1}, {"shell": 1, "content": 2, "action": 1}]
    model = SimplePolicy("linear", .2, .5)
    p, _, _, _ = _trajectory(model, rows)
    assert p[0] == .5 and p[1] == pytest.approx(1 / (1 + np.exp(-.1)))


def test_hidden_nogo_content_cannot_change_conditional_predictions():
    rows = [{"shell": i % 6 + 1, "content": 1, "action": 0} for i in range(12)]
    changed = [{**r, "content": 2} for r in rows]
    a, b = SimplePolicy("q", .2, .5), SimplePolicy("q", .2, .5)
    p1 = _trajectory(a, rows)[0]
    p2 = _trajectory(b, changed)[0]
    assert np.array_equal(p1, p2) and np.array_equal(a.values, b.values)


def test_autonomous_uses_own_action_consequence():
    rows = [{"shell": 1, "content": 2, "action": 1}]
    model = SimplePolicy("uniform", .1, .5)
    _, action, reward, _ = _trajectory(model, rows, True, np.array([.8]))
    assert action.tolist() == [0] and reward.tolist() == [0]


def test_nll_clipping_does_not_change_raw_brier():
    nll, brier = loss(np.array([0, 1]), np.array([0., 1.]))
    assert np.isfinite(nll).all() and np.array_equal(brier, [0., 0.])


@pytest.mark.parametrize("action,p", [([0], [float("nan")]), ([0], [1.1]), ([2], [.5]), ([0, 1], [.5])])
def test_invalid_loss_observation_rejected(action, p):
    with pytest.raises(ValueError):
        loss(action, p)


def test_behavior_hand_constructed_perfect_learning_and_reversal():
    rows = synthetic_rows()
    actions = np.zeros(216, dtype=int)
    for shell in (3, 4):
        idx = [i for i, r in enumerate(rows) if r["shell"] == shell]
        for phase in range(4):
            episode = idx[phase * 10:(phase + 1) * 10]
            target = int((shell == 3 and phase % 2 == 0) or (shell == 4 and phase % 2 == 1))
            actions[episode] = target
            actions[episode[:3]] = 1 - target
    metrics = behavior_metrics(rows, actions)
    assert metrics == {"acquisition_gain": 1., "reversal_early": 0., "reversal_recovery": 1.}


def test_behavior_unequal_phases_and_overlapping_windows():
    rows = synthetic_rows()
    idx = [i for i, r in enumerate(rows) if r["shell"] == 3]
    for t, i in enumerate(idx):
        rows[i]["state"] = 1 if t < 7 or 20 <= t < 33 else 2
    assert set(behavior_metrics(rows)) == set(BEHAVIOR_METRICS)
    for t, i in enumerate(idx):
        rows[i]["state"] = 1 if t < 5 or 20 <= t < 33 else 2
    with pytest.raises(ValueError):
        behavior_metrics(rows)


def test_small_pipeline_shapes_separation_and_reproducibility():
    cfg = synthetic_config()
    calibration = {"A9001": synthetic_rows()}
    test = {pid: synthetic_rows() for pid in cfg["test_ids"]}
    before = copy.deepcopy(calibration)
    selection, cal_arrays, cal_rows = fit(calibration, cfg, small_graph())
    assert calibration == before
    assert cal_arrays["probabilities"].shape == (8, 1, 1, 1, 216)
    assert len(cal_rows) == 8
    arrays, rows, witnesses = produce(test, cfg, small_graph(), selection)
    assert arrays["probabilities"].shape == (2, 11, 1, 2, 216)
    assert len(rows) == 44
    assert witnesses["conditional_q_final_value_table"].shape == (1, 6, 2)
    assert witnesses["conditional_q_final_head"].shape == (1, 0)
    human_actions = np.asarray([[r["action"] for r in test[pid]] for pid in cfg["test_ids"]])
    assert np.array_equal(arrays["actions"][0], np.broadcast_to(human_actions, (11, 1, 2, 216)))
    for row in rows:
        assert (row["mean_nll"] is None) == (row["mode"] == "autonomous")
    q = CONDITIONS.index("q")
    assert np.count_nonzero(arrays["head_steps"][:, q]) == 0
    assert np.count_nonzero(arrays["value_steps"][:, q]) > 0
    summary = summarize(test, cfg, arrays, rows)
    assert summary == summarize(test, cfg, arrays, rows)
    assert summary["human_test_people"] == 2 and summary["model_seeds"] == 1
    assert summary["H_similarity"]["claim"] == "behavioral_equivalence_only"
    assert summary["autonomous_nll_interpreted"] is False
    assert summary["mechanism_claim"] is False
