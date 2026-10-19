"""Independent invented-graph algebra and tamper tests, no scientific seed draws."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "representation_independent_test", Path(__file__).resolve().parents[1] / "scripts/verify_representation_update.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def tiny_graph():
    n = 16
    return {"pre": np.arange(n), "post": np.arange(n), "counts": np.ones(n), "n_kc": n, "n_mbon": n}


def fixture_config():
    return {"study_id": "representation_update_v1", "scope": "integration_test",
            "module": "drosolearn.representation_update", "protocol": "docs/representation-update-protocol.md",
            "topologies": list(audit.TOPOLOGIES), "representations": list(audit.INPUTS), "updates": list(audit.UPDATES),
            "model_seeds": [90081, 90082], "training_trials": 8, "checkpoints": [0, 4, 8],
            "head_lr": .1, "rng_namespace": 202610193, "svd_rcond": 1e-12, "numeric_tolerance": 1e-8,
            "exact_condition_max": 1e6, "acquisition_threshold": .8, "acquisition_gain": .25,
            "class_threshold": .7, "population_pass_count": 2, "effect_threshold": .1, "positive_seed_count": 2,
            "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": audit.INPUT_CATALOG,
            "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}


def write_input_catalog(root):
    path = root / audit.INPUT_CATALOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"study_id": "representation_update_v1", "representations": list(audit.INPUTS),
                               "catalog": {key: value.tolist() for key, value in audit.independent_catalog_arrays().items()}}), encoding="utf8")
    return path


def test_label_free_identity_catalog_equal_norm_but_different_geometry():
    raw, identity = audit.finite_catalog()
    assert raw.shape == identity.shape == (4, 7)
    assert np.allclose(np.linalg.norm(raw, axis=1), np.linalg.norm(identity, axis=1))
    assert np.allclose(np.linalg.norm(raw, axis=1), np.sqrt(6.) / 3.)
    assert np.array_equal(identity[:, 6], raw[:, 6])
    assert np.count_nonzero(identity[:, :4], axis=1).tolist() == [1] * 4
    expected_positions = [0, 3, 1, 2]
    assert np.argmax(identity[:, :4], axis=1).tolist() == expected_positions


def test_raw_nonseparability_exact_left_null_certificate():
    raw, identity = audit.finite_catalog()
    signed = np.array([1., 1., -1., -1.])
    assert np.max(np.abs(signed @ raw)) == 0.
    assert np.linalg.norm(signed @ identity) > 0
    _, witness = audit.minimum_norm_margin(raw, signed)
    assert witness is None


def test_identity_analytical_witness_is_not_a_trained_policy():
    _, identity = audit.finite_catalog()
    candidates, best = audit.minimum_norm_margin(identity, np.array([1., 1., -1., -1.]))
    assert len(candidates) == 15 and best is not None
    assert np.min(best["margins"]) >= 1 - 1e-9
    assert len(best["head"]) == 7
    assert np.allclose(best["head"][-3:], 0., atol=1e-9)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_rank_deficient_geometry_is_retained_and_own_effect_matched(rank):
    if rank == 1:
        features = np.ones((4, 1))
    else:
        features = np.eye(4)[:, :rank]
        for i in range(rank, 4): features[i] = np.arange(1, rank + 1)
    g = audit.independent_geometry(features)
    assert g["rank"] == rank
    assert np.allclose(np.diag(g["shared_influence"]), 1.)
    assert np.allclose(np.diag(g["projected_influence"]), 1.)
    assert np.allclose(g["projector"] @ g["projector"], g["projector"])
    assert np.allclose(g["projector"], g["projector"].T)
    if rank == 4:
        assert np.allclose(g["projected_influence"], np.eye(4))
    else:
        assert np.max(np.abs(g["projected_influence"] - np.eye(4))) > .1


def test_svd_cutoff_is_relative_and_not_an_adaptive_rank_fallback():
    x = np.eye(4)
    x[:, 3] = x[:, 0] + 1e-14 * np.array([0., 0., 0., 1.])
    x[3, 1] = 1.
    a = audit.independent_geometry(x)
    b = audit.independent_geometry(17. * x)
    assert a["rank"] == b["rank"] == 3
    assert np.allclose(a["D"], b["D"])
    assert np.allclose(a["projected_influence"], b["projected_influence"])


def test_projected_update_is_not_l2_budget_matched():
    features = np.array([[1., 0., 0., 0.], [1., .001, 0., 0.], [1., 0., .001, 0.], [1., 0., 0., .001]])
    g = audit.independent_geometry(features)
    assert g["rank"] == 4
    assert np.min(g["projected_direction_norms"]) > 500.
    assert np.allclose(g["shared_direction_norms"], 1.)
    assert np.allclose(g["projected_influence"], np.eye(4), atol=1e-9)


@pytest.mark.parametrize("bad", [np.zeros((4, 2)), np.ones((3, 2)), np.full((4, 2), np.nan)])
def test_zero_or_nonfinite_feature_rows_rejected(bad):
    with pytest.raises(ValueError): audit.independent_geometry(bad)


def test_fixed_graph_evidence_has_paired_nogo_and_go_without_weight_changes():
    raw, _ = audit.finite_catalog()
    evidence = audit.feature_evidence(tiny_graph(), 90081, raw)
    assert evidence["probe_kc"].shape == (4, 2, 16)
    assert evidence["probe_mbon"].shape == (4, 2, 16)
    assert np.all(evidence["weights"] == 1.)
    assert np.all(evidence["probe_kc"][:, 0] == evidence["probe_kc"][0, 0])
    assert np.array_equal(evidence["differences"], evidence["probe_mbon"][:, 1] - evidence["probe_mbon"][:, 0])


def compare_reports(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict): compare_reports(actual[key], value)
        elif isinstance(value, np.ndarray): assert np.allclose(actual[key], value, atol=1e-9, rtol=1e-9), key
        elif isinstance(value, (float, np.floating)): assert actual[key] == pytest.approx(value, abs=1e-9), key
        else: assert actual[key] == value, key


@pytest.mark.parametrize("kind", ["raw", "identity", "asymmetric_rank3", "rank4"])
def test_current_geometry_helpers_match_independent_invented_input_arithmetic(kind):
    # Production helper is exercised only here, never imported by the verifier.
    from drosolearn.representation_geometry import geometry, hard_margin_witness
    raw, identity = audit.finite_catalog()
    matrix = {"raw": raw, "identity": identity, "rank4": np.eye(4),
              "asymmetric_rank3": np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., 3.]])}[kind]
    compare_reports(geometry(matrix), audit.geometry_report(matrix))
    for index in range(16):
        labels = np.array([1 if index & (1 << j) else -1 for j in range(4)])
        compare_reports(hard_margin_witness(matrix, labels), audit.margin_report(matrix, labels))


@pytest.mark.parametrize("rule", audit.UPDATES)
def test_reward_only_replay_keeps_internal_and_activity_fixed(rule):
    inputs, _ = audit.finite_catalog()
    actor = audit.ReplayCell(tiny_graph(), 90081, inputs, rule)
    evidence = copy.deepcopy(actor.evidence)
    assert np.array_equal(actor.probabilities(), np.full(4, .5))
    first = actor.observe(0, 1, 1)
    assert first > 0 and actor.logits()[0] == pytest.approx(.05)
    assert actor.calls == 1
    assert actor.observe(1, 0, 0) == 0
    assert actor.calls == 2
    for key, value in evidence.items(): assert np.array_equal(actor.evidence[key], value), key
    with pytest.raises(ValueError): actor.observe(1, 0, -1)
    with pytest.raises(ValueError): actor.observe(4, 1, 1)


@pytest.fixture
def independently_replayed():
    cfg = fixture_config()
    arrays, rows = audit.expected_artifacts(cfg, tiny_graph())
    return cfg, arrays, rows, audit.expected_summary(cfg, arrays)


def test_independent_full_synthetic_replay_and_budget(independently_replayed):
    cfg, a, rows, summary = independently_replayed
    assert a["trial_actions"].shape == (3, 2, 2, 2, 8)
    assert a["checkpoint_probabilities"].shape == (3, 2, 2, 2, 3, 4)
    assert summary["counts"]["training_feedback_trials"] == 192
    assert summary["counts"]["static_probe_cases"] == 288
    assert summary["counts"]["head_update_calls"] == 192
    assert summary["counts"]["internal_update_calls"] == 0
    result = audit.verify_against_graph(cfg, tiny_graph(), a, rows, summary, allow_fixture=True)
    assert result["independently_replayed_training_actions"] == 192
    assert result["all_checkpoints_and_trial_head_states_replayed"]


def test_full_producer_agrees_with_independent_graph_and_reward_arithmetic(monkeypatch, tmp_path):
    from drosolearn import representation_update as producer
    from drosolearn.graph import Projection
    raw = tiny_graph(); n = raw["n_kc"]
    graph = Projection(raw["pre"], raw["post"], raw["counts"], n, n,
                       [str(i) for i in range(n)], [str(i) for i in range(n)], {})
    monkeypatch.setattr(producer, "load_projection", lambda path: graph)
    cfg = fixture_config(); write_input_catalog(tmp_path)
    audit.verify_input_catalog(cfg, tmp_path)
    arrays, rows = producer.produce(cfg, tmp_path)
    summary = producer.summarize(cfg, arrays, rows)
    checks = audit.verify_against_graph(cfg, raw, arrays, rows, summary, allow_fixture=True)
    assert checks["independently_replayed_training_actions"] == 192


@pytest.mark.parametrize("field", ["inputs", "targets", "lexicographic_ids", "schedules", "uniforms", "trial_actions",
                                    "trial_probabilities", "trial_rewards", "trial_head_steps", "trial_own_score_steps",
                                    "trial_other_score_steps", "trial_internal_steps", "trial_value_update_calls",
                                    "checkpoint_logits", "checkpoint_heads", "checkpoint_internal_calls",
                                    "w_real_raw_weights", "finalfixed_ann_identity_probe_mbon",
                                    "g_real_raw_influence_shared", "g_rewired_identity_directions",
                                    "g_real_identity_unit_diagnostics_full_row_condition_defined",
                                    "m_ann_raw_candidate_heads", "a_raw_candidate_feasible"])
def test_catalog_geometry_training_or_fixed_state_tampering_rejected(independently_replayed, field):
    cfg, original, rows, summary = independently_replayed
    arrays = {key: value.copy() for key, value in original.items()}
    flat = arrays[field].reshape(-1)
    if flat.dtype.kind == "b": flat[-1] = not flat[-1]
    else: flat[-1] += 1 if flat.dtype.kind in "iu" else .25
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def test_trial_csv_reordering_or_hidden_reward_change_rejected(independently_replayed):
    cfg, arrays, original, summary = independently_replayed
    rows = copy.deepcopy(original); rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    rows = copy.deepcopy(original); rows[0]["reward"] += 1
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)


def test_missing_or_extra_array_or_summary_claim_rejected(independently_replayed):
    cfg, original, rows, summary = independently_replayed
    arrays = dict(original); del arrays["trial_actions"]
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    arrays = dict(original); arrays["unregistered"] = np.zeros(1)
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), arrays, rows, summary, allow_fixture=True)
    summary = copy.deepcopy(summary); summary["H_generalization"] = True
    with pytest.raises(ValueError): audit.verify_against_graph(cfg, tiny_graph(), original, rows, summary, allow_fixture=True)


@pytest.mark.parametrize("key,value", [("head_lr", .2), ("rng_namespace", 99), ("svd_rcond", 1e-6),
                                       ("normalization", "old_rms"), ("effect_threshold", .01), ("class_threshold", .2)])
def test_new_threshold_or_update_change_rejected(key, value):
    cfg = fixture_config(); cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)


def test_integration_fixture_cannot_authorize_science_seed_or_scientific_budget():
    cfg = fixture_config(); cfg["model_seeds"] = [17101]
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)
    cfg = fixture_config(); cfg["training_trials"] = 400
    with pytest.raises(ValueError): audit.validate_science(cfg, allow_fixture=True)
    with pytest.raises(ValueError): audit.validate_science(fixture_config())


@pytest.mark.parametrize("kind", ["numeric", "class", "ids", "extra", "float_labels"])
def test_input_catalog_semantic_changes_rejected(tmp_path, kind):
    path = write_input_catalog(tmp_path)
    doc = json.loads(path.read_text())
    if kind == "numeric": doc["catalog"]["inputs"][0][0][0] += 1e-13
    elif kind == "class": doc["catalog"]["targets"][0] = 0
    elif kind == "ids": doc["catalog"]["lexicographic_ids"].reverse()
    elif kind == "extra": doc["label_hint"] = True
    else: doc["catalog"]["targets"] = [float(v) for v in doc["catalog"]["targets"]]
    path.write_text(json.dumps(doc), encoding="utf8")
    with pytest.raises(ValueError): audit.verify_input_catalog(fixture_config(), tmp_path)


def test_exact_raw_certificate_with_integer_minors_and_nonlinear_limit():
    proof = audit.raw_certificate()
    assert proof["rank_minor_determinant"] == -1
    assert proof["affine_minor_determinant"] == 1
    assert proof["exact_row_rank"] == 3
    assert proof["exact_affine_dimension"] == 2
    assert proof["three_correct_example_scaled_logits"] == [1, 1, -1, 3]
    assert proof["nonlinear_feature_separation_not_ruled_out"]


def test_summary_gates_do_not_accept_high_mean_with_one_unlearned_class(independently_replayed):
    cfg, original, _, _ = independently_replayed
    arrays = copy.deepcopy(original)
    p = arrays["checkpoint_probabilities"]; p[:] = .5
    for ri, ui in ((0, 1), (1, 0)):
        p[0, ri, ui, :, -1, :2] = .99
        p[0, ri, ui, :, -1, 2:] = .35
    summary = audit.expected_summary(cfg, arrays)
    # Mean .82 and gain .32 pass, but class-0 accuracy .65 blocks acquisition.
    assert summary["families"][1]["mean_final_expected_accuracy"] > .8
    assert summary["H_update"]["treated_acquisition_pass_seeds"] == 0
    assert not summary["H_update"]["supported"] and not summary["H_representation"]["supported"]


def test_summary_requires_effect_size_even_when_every_seed_acquires(independently_replayed):
    cfg, original, _, _ = independently_replayed
    arrays = copy.deepcopy(original)
    p = arrays["checkpoint_probabilities"]; p[:] = .5
    p[0, :, :, :, -1, :2] = .95
    p[0, :, :, :, -1, 2:] = .05
    p[0, 0, 0, :, -1, :2] = .90
    p[0, 0, 0, :, -1, 2:] = .10
    summary = audit.expected_summary(cfg, arrays)
    for key in ("H_update", "H_representation"):
        assert summary[key]["positive_seeds"] == 2
        assert summary[key]["treated_acquisition_pass_seeds"] == 2
        assert summary[key]["mean_effect"] == pytest.approx(.05)
        assert not summary[key]["supported"]


def test_nonprimary_numeric_failure_closes_both_primary_hypotheses(independently_replayed):
    cfg, original, _, _ = independently_replayed
    arrays = copy.deepcopy(original)
    p = arrays["checkpoint_probabilities"]; p[:] = .5
    for ri, ui in ((0, 1), (1, 0)):
        p[0, ri, ui, :, -1, :2] = .95
        p[0, ri, ui, :, -1, 2:] = .05
    before = audit.expected_summary(cfg, arrays)
    assert before["H_update"]["supported"] and before["H_representation"]["supported"]
    arrays["g_ann_identity_numeric_update_qualified"][1] = False
    after = audit.expected_summary(cfg, arrays)
    for key in ("H_update", "H_representation"):
        assert after[key]["treated_acquisition_pass_seeds"] == 2
        assert not after[key]["numeric_update_qualified_all"]
        assert not after[key]["supported"]
