"""Independent numeric guards using invented seeds and synthetic graphs only."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("shell_credit_independent", Path(__file__).resolve().parents[1] / "scripts/verify_shell_credit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def config():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/shell_credit_v1.json").read_text(encoding="utf8"))
    cfg.update(scope="integration_test", seeds=[71, 72], exposures_per_shell=12, checkpoint_every_exposures=3)
    return cfg


def synthetic_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


def coupled():
    return np.eye(6) + .5 * np.ones((6, 6))


def test_registered_configuration_without_running_scientific_seeds():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/shell_credit_v1.json").read_text(encoding="utf8"))
    audit.validate_science(cfg)


@pytest.mark.parametrize("key,value", [("head_lr", .2), ("seeds", [15001]), ("exposures_per_shell", 240),
                                       ("scope", "integration_test"), ("reward_probability", 1.),
                                       ("criteria", {}), ("stream_namespace", 202610192)])
def test_actual_validation_rejects_unregistered_choices(key, value):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/shell_credit_v1.json").read_text(encoding="utf8"))
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg)


@pytest.mark.parametrize("seed,component", [(True, 0), (-1, 0), (71., 0), (71, True), (71, -1), (71, 4)])
def test_canonical_stream_keys(seed, component):
    with pytest.raises(ValueError): audit.stream(seed, component)


def test_stream_order_is_explicit_and_components_differ():
    for component in range(4):
        np.testing.assert_array_equal(audit.stream(71, component).random(12),
                                      np.random.default_rng(np.random.SeedSequence([202610191, 71, component])).random(12))
    assert not np.array_equal(audit.stream(71, 2).random(8), audit.stream(71, 3).random(8))
    assert not np.array_equal(audit.stream(71, 3).random(8), audit.stream(72, 3).random(8))


def test_task_has_three_labels_balanced_blocks_and_no_extra_outcome_clues():
    plan = audit.trial_plan(71, 12)
    assert plan["target_go"].sum() == 3
    for block in plan["shells"].reshape(-1, 6):
        assert set(block) == set(range(1, 7))
    expected = np.where(plan["outcome_uniforms"] < .8, 2 * plan["target_go"][plan["shells"] - 1] - 1, 0)
    np.testing.assert_array_equal(plan["potential_rewards"], expected)


def test_svd_dual_satisfies_identity_and_own_match_for_both_signs():
    d = coupled(); geom = audit.geometry(d)
    np.testing.assert_allclose(d @ geom["dual_rows"].T, np.eye(6), atol=1e-13)
    shared, isolated, q = audit.influence_matrices(d, geom["dual_rows"], .1)
    np.testing.assert_allclose(np.diagonal(shared, axis1=1, axis2=2), np.diagonal(isolated, axis1=1, axis2=2))
    assert np.max(np.abs(isolated[:, ~np.eye(6, dtype=bool)])) < 1e-12
    assert np.any(shared[:, ~np.eye(6, dtype=bool)] != 0)
    np.testing.assert_array_equal(q[0], -.4 * np.eye(6)); np.testing.assert_array_equal(q[1], .4 * np.eye(6))


@pytest.mark.parametrize("d", [np.ones((6, 8)), np.diag([1., 1, 1, 1, 1, 1e-5]),
                               np.full((6, 8), np.nan), np.eye(5)])
def test_geometry_fails_closed(d):
    with pytest.raises(ValueError): audit.geometry(d)


def test_svd_dual_does_not_call_producer_gram_solver(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("producer linear-solve path")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    assert audit.geometry(coupled())["rank"] == 6


def tiny_plan(actions, potentials):
    n = len(actions)
    return {"shells": np.ones(n, dtype=int), "target_go": np.ones(6, int),
            "potential_rewards": np.asarray(potentials), "action_uniforms": np.asarray(actions)}


@pytest.mark.parametrize("condition", ["shared", "isolated", "uniform"])
def test_nogo_does_not_use_hidden_go_reward(condition):
    d = coupled(); dual = audit.geometry(d)["dual_rows"]
    left, _ = audit.replay(d, dual, tiny_plan([1., 1.], [1, -1]), condition, .1, [0, 2])
    right, _ = audit.replay(d, dual, tiny_plan([1., 1.], [-1, 1]), condition, .1, [0, 2])
    np.testing.assert_array_equal(left["heads"], right["heads"])
    np.testing.assert_array_equal(left["rewards"], [0, 0])
    np.testing.assert_array_equal(left["head_steps"], [0, 0])


def test_go_is_predicted_before_update_and_empty_go_does_not_change_head():
    d = np.eye(6); dual = audit.geometry(d)["dual_rows"]
    a, rows = audit.replay(d, dual, tiny_plan([0., 0., 0.], [1, 0, -1]), "shared", .1, [0, 1, 2, 3])
    assert a["probabilities"][0] == .5
    assert a["probabilities"][1] == pytest.approx(audit.core.logistic(.05))
    assert a["probabilities"][2] == a["probabilities"][1]
    assert a["head_steps"][1] == 0
    assert rows[0]["p_correct"] == .5
    assert np.all(a["head_update_calls"] == 1) and np.all(a["internal_update_calls"] == 0)


@pytest.fixture
def produced():
    # Importing the producer is confined to this synthetic cross-implementation
    # test; the verifier itself has no scientific producer import.
    from drosolearn.graph import Projection
    from drosolearn import shell_credit
    cfg, g = config(), synthetic_graph()
    projection = Projection(g["pre"], g["post"], g["counts"], g["n_kc"], g["n_mbon"],
                            [f"k{i}" for i in range(16)], [f"m{i}" for i in range(16)], {"synthetic": True})
    a, rows = shell_credit._produce(cfg, projection)
    return cfg, g, a, rows, shell_credit.summarize(cfg, a, rows)


def test_full_tiny_producer_and_independent_replay_agree(produced):
    cfg, graph, arrays, rows, summary = produced
    checks = audit.verify_against_graph(cfg, arrays, rows, summary, graph, allow_fixture=True)
    assert checks["independently_replayed_actions"] == 432
    assert checks["head_update_calls"] == 288 and checks["internal_update_calls"] == 0


def test_csv_string_values_are_also_independently_checked(produced):
    cfg, graph, arrays, rows, summary = produced
    csv_rows = [{key: str(value) for key, value in row.items()} for row in rows]
    audit.verify_against_graph(cfg, arrays, csv_rows, summary, graph, allow_fixture=True)


@pytest.mark.parametrize("field", ["actions", "rewards", "head_update_calls", "internal_update_calls",
                                   "probabilities", "head_steps", "weights_final", "kc_final",
                                   "heads", "dual_rows", "action_uniforms", "influence_isolated"])
def test_corrupted_numeric_artifact_is_rejected(produced, field):
    cfg, graph, original, rows, summary = produced
    arrays = copy.deepcopy(original)
    arrays[field].flat[0] += 1
    with pytest.raises(ValueError):
        audit.verify_against_graph(cfg, arrays, rows, summary, graph, allow_fixture=True)


def test_corrupted_row_or_summary_or_missing_artifact_is_rejected(produced):
    cfg, graph, arrays, original, summary = produced
    rows = copy.deepcopy(original); rows[0]["exposure"] += 1
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, arrays, rows, summary, graph, allow_fixture=True)
    wrong = copy.deepcopy(summary); wrong["H_credit"]["supported"] = not wrong["H_credit"]["supported"]
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, arrays, original, wrong, graph, allow_fixture=True)
    missing = dict(arrays); missing.pop("internal_update_calls")
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, missing, original, summary, graph, allow_fixture=True)


def test_criterion_requires_own_acquisition_as_well_as_effect(produced):
    cfg, _, original, _, _ = produced
    arrays = copy.deepcopy(original)
    for si in range(2):
        arrays["probe_probabilities"][si, :, -1] = .5
        arrays["probe_probabilities"][si, 1, -1] = np.where(arrays["target_go"][si], .7, .3)
    summary = audit.expected_summary(cfg, arrays)
    assert summary["H_credit"]["effect_criterion_met"] is True
    assert summary["H_credit"]["isolated_acquisition_gate"] is False
    assert summary["H_credit"]["supported"] is False
