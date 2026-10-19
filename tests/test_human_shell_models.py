"""Hand-scale fixtures only: no human files or registered scientific seeds."""
import numpy as np
import pytest

from drosolearn.graph import Projection
from drosolearn.human_shell_models import (CATALOG, ShellActor, SimplePolicy,
                                         build_graphs, make_model)


def small_graph():
    return Projection(np.arange(6), np.array([0, 1, 2, 0, 1, 2]), np.ones(6), 6, 3,
                      list(map(str, range(6))), list(map(str, range(3))), {"synthetic": True})


def test_catalog_has_only_current_shell_identity():
    assert CATALOG.shape == (6, 2, 7)
    assert np.array_equal(CATALOG[:, 0], np.zeros((6, 7)))
    assert np.array_equal(CATALOG[:, 1, :6], np.eye(6))
    assert np.all(CATALOG[..., 6] == 0)


@pytest.mark.parametrize("reward,sign", [(1, 1), (-1, -1), (0, 0)])
def test_linear_signed_update_has_hand_computed_direction(reward, sign):
    model = SimplePolicy("linear", .2, .5)
    step, internal, value = model.observe(2, 1, reward)
    assert model.head[1] == pytest.approx(sign * .1)
    assert step == pytest.approx(abs(sign * .1))
    assert internal == value == 0


def test_nogo_does_not_reveal_go_outcome():
    model = SimplePolicy("q", .2, .5)
    model.observe(1, 0, 0)
    assert np.array_equal(model.values, np.zeros((6, 2)))
    assert model.value_updates == 1 and model.head_updates == 0
    with pytest.raises(ValueError):
        model.observe(1, 0, -1)


def test_q_updates_only_chosen_value_separate_from_head():
    model = SimplePolicy("q", .2, .5)
    head, internal, value = model.observe(4, 1, -1)
    assert head == internal == 0 and value == pytest.approx(.2)
    assert model.head.size == 0 and model.values[3, 1] == pytest.approx(-.2)
    assert np.count_nonzero(model.values) == 1
    assert model.probability(4) == pytest.approx(1 / (1 + np.exp(.8)))


def test_persistence_tracks_same_shell_only_without_reward_learning():
    model = SimplePolicy("persistence", .2, .4)
    assert model.probability(2) == .5
    model.observe(2, 1, 1)
    assert model.probability(2) == .9 and model.probability(3) == .5
    model.observe(3, 0, 0)
    assert model.probability(3) == .1 and model.probability(2) == .9
    assert model.head_updates == model.internal_updates == model.value_updates == 0


def test_old_head_eligibility_and_zero_reward_leave_internal_state():
    model = ShellActor(small_graph(), 17, .2, True)
    initial = model.weights.copy()
    h1, i1, _ = model.observe(3, 1, 1)
    assert h1 > 0 and i1 == 0
    assert np.array_equal(initial, model.weights)
    old_head = model.head.copy()
    model.observe(3, 1, 0)
    assert np.array_equal(initial, model.weights) and np.array_equal(old_head, model.head)
    _, i2, _ = model.observe(3, 1, 1)
    assert i2 > 0
    assert np.all(model.weights >= 0) and np.all(model.weights <= model.upper)
    assert model.internal_updates == model.head_updates == 3


def test_normalized_catalog_is_initial_rms_one():
    model = ShellActor(small_graph(), 19, .1, False)
    z = model.witness()["probe_decision"]
    assert np.sqrt(np.mean(np.sum((z[:, 1] - z[:, 0]) ** 2, axis=-1))) == pytest.approx(1.)


def test_shared_topology_does_not_share_mutable_learning_state():
    graph = small_graph()
    original = graph.weights.copy()
    first, second = (ShellActor(graph, 21, .3, True) for _ in range(2))
    for _ in range(3):
        first.observe(1, 1, 1)
    assert np.array_equal(graph.weights, original)
    assert np.all(second.head == 0) and np.array_equal(second.weights, second.initial_weights)
    assert not np.shares_memory(first.weights, second.weights)


def test_sub_ulp_proposals_are_recorded_as_zero_actual_change():
    linear = SimplePolicy("linear", .1, .5)
    linear.head[0] = -1e16
    assert .1 * (1 - linear.probability(1)) > 0
    assert linear.observe(1, 1, 1)[0] == 0
    q = SimplePolicy("q", 1e-20, .5)
    q.values[0, 1] = 1.
    assert q.observe(1, 1, 0)[2] == 0
    actor = ShellActor(small_graph(), 29, .1, False)
    _, _, z = actor._parts(1)
    actor.head[:] = -1e16 * np.sign(z[1] - z[0])
    assert np.linalg.norm(.1 * (1 - actor.probability(1)) * (z[1] - z[0])) > 0
    assert actor.observe(1, 1, 1)[0] == 0


def test_topology_factory_and_model_reset_are_deterministic():
    graphs = build_graphs(small_graph(), 23)
    first = make_model("rewired_plastic", 23, .1, .5, graphs)
    second = make_model("rewired_plastic", 23, .1, .5, graphs)
    assert np.array_equal(first.encoder, second.encoder)
    first.observe(2, 1, 1)
    assert np.all(second.head == 0)
    assert second.probability(2) == .5
