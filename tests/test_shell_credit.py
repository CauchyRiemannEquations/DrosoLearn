"""Synthetic source/geometry/learning guards; no actual scientific graph or seeds."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from drosolearn.graph import Projection
from drosolearn import shell_credit as study


def config():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/shell_credit_v1.json").read_text(encoding="utf8"))
    cfg.update(scope="integration_test", seeds=[17, 18], exposures_per_shell=12, checkpoint_every_exposures=3)
    return cfg


def graph():
    n = 16
    return Projection(np.arange(n), np.arange(n), np.ones(n), n, n,
                      [f"kc{i}" for i in range(n)], [f"mb{i}" for i in range(n)], {"synthetic": True})


def coupled_gaps():
    return np.eye(6) + .5 * np.ones((6, 6))


def test_science_config_validated_without_execution():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/shell_credit_v1.json").read_text(encoding="utf8"))
    study.validate_config(cfg)


@pytest.mark.parametrize("key,value", [("head_lr", .3), ("reward_probability", 1.), ("seeds", [15001]),
                                       ("exposures_per_shell", 240), ("gram_condition_max", 1e12),
                                       ("geometry_tolerance", 1e-3), ("conditions", ["shared", "isolated"])])
def test_registered_science_cannot_expand_or_change(key, value):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/shell_credit_v1.json").read_text(encoding="utf8"))
    cfg[key] = value
    with pytest.raises(ValueError):
        study.validate_config(cfg)


def test_integration_cannot_use_scientific_seed():
    cfg = config(); cfg["seeds"] = [15001]
    with pytest.raises(ValueError): study.validate_config(cfg)


def test_dual_matches_own_step_and_blocks_other_shells():
    d = coupled_gaps(); geom = study.geometry(d)
    for reward in (-1, 1):
        shared = study.HeadPolicy("shared", d, geom["dual_rows"], .1)
        isolated = study.HeadPolicy("isolated", d, geom["dual_rows"], .1)
        shared.observe(3, 1, reward); isolated.observe(3, 1, reward)
        shared_logits, isolated_logits = d @ shared.head, d @ isolated.head
        assert isolated_logits[2] == pytest.approx(shared_logits[2])
        assert np.max(np.abs(np.delete(isolated_logits, 2))) < 1e-12
        assert np.all(np.abs(np.delete(shared_logits, 2)) > 0)


@pytest.mark.parametrize("condition", ["shared", "isolated", "uniform"])
def test_neutral_and_nogo_never_leak_hidden_content(condition):
    d = coupled_gaps(); dual = study.geometry(d)["dual_rows"]
    model = study.HeadPolicy(condition, d, dual, .1)
    model.observe(2, 0, 0); model.observe(2, 1, 0)
    assert np.array_equal(model.head, np.zeros(6))
    with pytest.raises(ValueError): model.observe(2, 0, -1)


def test_fixed_uniform_is_not_a_reward_learner():
    d = coupled_gaps(); dual = study.geometry(d)["dual_rows"]
    model = study.HeadPolicy("uniform", d, dual, .1)
    for shell in range(1, 7): model.observe(shell, 1, 1)
    assert model.calls == 0 and np.array_equal(model.head, np.zeros(6))
    assert np.array_equal(model.probabilities(), np.full(6, .5))


def test_rank_deficient_or_ill_conditioned_geometry_stops():
    with pytest.raises(ValueError): study.geometry(np.ones((6, 8)))
    d = np.eye(6); d[-1, -1] = 1e-5
    with pytest.raises(ValueError): study.geometry(d)


def test_gate_failure_precedes_any_task_random_generation(monkeypatch):
    def stopped(*args, **kwargs): raise ValueError("fixture rank failure")
    def no_task_draw(*args, **kwargs): raise AssertionError("task draw before geometry gate")
    monkeypatch.setattr(study, "geometry", stopped)
    monkeypatch.setattr(study, "_rng", no_task_draw)
    with pytest.raises(ValueError, match="seed 17: fixture rank failure"):
        study._produce(config(), graph())


def test_actual_step_records_rounding_not_nominal_proposal():
    d = coupled_gaps(); dual = study.geometry(d)["dual_rows"]
    model = study.HeadPolicy("shared", d, dual, .1)
    model.head[:] = -1e16
    assert model.observe(1, 1, 1) == 0


def test_static_matching_gate_also_precedes_task_draws(monkeypatch):
    original_geometry = study.geometry
    def bad_dual(*args, **kwargs):
        result = original_geometry(*args, **kwargs)
        result["dual_rows"] = result["dual_rows"] + .2
        return result
    def no_task_draw(*args, **kwargs): raise AssertionError("task draw before static matching gate")
    monkeypatch.setattr(study, "geometry", bad_dual)
    monkeypatch.setattr(study, "_rng", no_task_draw)
    with pytest.raises(ValueError, match="Static own-logit match/isolation gate failed"):
        study._produce(config(), graph())


@pytest.fixture
def produced():
    cfg, projection = config(), graph()
    before = projection.weights.copy()
    arrays, rows = study._produce(cfg, projection)
    assert np.array_equal(projection.weights, before)
    return cfg, arrays, rows


def test_small_pipeline_budget_schedule_and_mask(produced):
    cfg, a, rows = produced
    assert a["actions"].shape == (2, 3, 72) and len(rows) == 432
    for si in range(2):
        assert np.sum(a["target_go"][si]) == 3
        for block in a["shells"][si].reshape(-1, 6): assert sorted(block.tolist()) == list(range(1, 7))
        for ci in range(3):
            assert np.array_equal(a["rewards"][si, ci], np.where(a["actions"][si, ci], a["potential_rewards"][si], 0))
    assert a["head_update_calls"].sum() == 288 and a["internal_update_calls"].sum() == 0
    assert not np.any(a["internal_steps"])


def test_model_state_and_probe_separation(produced):
    _, a, _ = produced
    for si in range(2):
        for ci in range(3):
            assert np.array_equal(a["weights_final"][si, ci], a["weights_initial"][si])
            for key in ("kc", "mbon", "decision"):
                assert np.array_equal(a[f"{key}_final"][si, ci], a[f"{key}_initial"][si])
    assert np.array_equal(a["heads"][:, 2], np.zeros_like(a["heads"][:, 2]))
    assert np.array_equal(a["probe_probabilities"][:, 2], np.full_like(a["probe_probabilities"][:, 2], .5))


def test_static_impulse_arrays_match_both_reward_signs(produced):
    _, a, _ = produced
    for si in range(2):
        for ri, sign in enumerate((-1, 1)):
            assert np.allclose(np.diag(a["influence_shared"][si, ri]), np.diag(a["influence_isolated"][si, ri]))
            assert np.max(np.abs(a["influence_isolated"][si, ri][~np.eye(6, dtype=bool)])) < 1e-12
            assert np.allclose(a["influence_q_reference"][si, ri], np.eye(6) * sign * .4)
        assert a["influence_shared"][si, 1].diagonal().mean() == pytest.approx(.05)


def test_row_order_resources_and_summary(produced):
    cfg, arrays, rows = produced
    assert [(r["seed"], r["condition"], r["trial"]) for r in rows] == [
        (seed, c, t) for seed in cfg["seeds"] for c in study.CONDITIONS for t in range(1, 73)]
    summary = study.summarize(cfg, arrays, rows)
    assert summary["resources"]["actual_choice_feedback_trials"] == 432
    assert summary["counts"]["actual_choice_feedback_trials"] == 432
    assert summary["resources"]["isolated_extra_fixed_coefficients"] == 96
    assert summary["historical_results_replaced"] is summary["human_similarity_tested"] is False
    assert summary["resources"]["diverged_autonomous_paths_have_equal_steps_not_claimed"] is True


def test_summary_requires_effect_and_acquisition_gate(produced):
    cfg, original, rows = produced
    arrays = copy.deepcopy(original)
    for si in range(2):
        target = arrays["target_go"][si]
        arrays["probe_probabilities"][si, :, -1] = .5
        arrays["probe_probabilities"][si, 1, -1] = np.where(target, .9, .1)
    supported = study.summarize(cfg, arrays, rows)
    assert supported["H_credit"]["supported"] is True
    for si in range(2):
        target = arrays["target_go"][si]
        arrays["probe_probabilities"][si, 1, -1] = np.where(target, .7, .3)
    failed = study.summarize(cfg, arrays, rows)
    assert failed["H_credit"]["effect_criterion_met"] is True
    assert failed["H_credit"]["isolated_acquisition_gate"] is False
    assert failed["H_credit"]["supported"] is False


def test_synthetic_run_reproducible_without_shared_mutable_state(produced):
    cfg, arrays, rows = produced
    again, again_rows = study._produce(cfg, graph())
    assert rows == again_rows
    assert all(a.tobytes() == again[k].tobytes() for k, a in arrays.items())
