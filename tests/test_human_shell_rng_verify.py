"""Synthetic checks for tuple-key independent verification; no human outcomes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "human_shell_rng_independent_verifier", Path(__file__).resolve().parents[1] / "scripts/verify_human_shell_rng.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_tuple_keys_split_the_old_diagonal_collision():
    report = audit.key_audit(["A9001", "A9002"], [17, 18])
    assert report["cells"] == 4
    assert report["old_unique_integer_keys"] == 3
    assert report["old_collision_groups"] == 1
    assert report["old_duplicate_uses_beyond_first"] == 1
    assert report["old_equal_key_pairs"] == 1
    assert report["new_unique_entropy_tuples"] == 4
    assert report["new_unique_seedsequence_states"] == 4
    assert report["new_unique_pcg64_states"] == 4
    assert report["rng_draws_in_key_audit"] == 0


def test_generator_state_proves_the_registered_entropy_order():
    actual = audit.generator(17, "A9001").bit_generator.state
    expected = np.random.default_rng(np.random.SeedSequence([20261019, 17, 9001])).bit_generator.state
    assert actual == expected


def test_same_cell_deliberately_shares_uniforms_across_conditions():
    np.testing.assert_array_equal(audit.generator(17, "A9001").random(8),
                                  audit.generator(17, "A9001").random(8))


def test_old_equal_sum_pairs_now_have_different_streams():
    assert 17 + 9002 == 18 + 9001
    assert not np.array_equal(audit.generator(17, "A9002").random(8),
                             audit.generator(18, "A9001").random(8))


@pytest.mark.parametrize("seed,pid", [(True, "A9001"), (-1, "A9001"), (17, "9001"),
                                      (17, "A90001"), (17, "A9001 "), (17., "A9001"), (17, "A\u0669\u0660\u0660\u0661")])
def test_noncanonical_rng_identity_is_rejected(seed, pid):
    with pytest.raises(ValueError):
        audit.entropy_tuple(seed, pid)


@pytest.mark.parametrize("ids,seeds", [(["A9001", "A9001"], [17]), (["A9001"], [17, 17]),
                                       ([], [17]), (["A9001"], [])])
def test_audit_rejects_duplicate_or_empty_axes(ids, seeds):
    with pytest.raises(ValueError):
        audit.key_audit(ids, seeds)


class ConstantPolicy:
    def __init__(self, probability):
        self.p = probability
        self.observed = []

    def probability(self, shell):
        return self.p

    def update(self, shell, action, reward):
        self.observed.append((shell, action, reward))
        return 0., 0., 0.


def test_nogo_never_reads_the_hidden_potential_content():
    policy = ConstantPolicy(0.)
    record = {"id": "A9001", "shell": [1, 3, 6], "content": [1, 2, 999]}
    tape = audit.replay_tuple(policy, record, 17)
    np.testing.assert_array_equal(tape["action"], [0, 0, 0])
    np.testing.assert_array_equal(tape["reward"], [0, 0, 0])
    assert policy.observed == [(1, 0, 0), (3, 0, 0), (6, 0, 0)]


def test_go_receives_only_the_chosen_current_content():
    policy = ConstantPolicy(1.)
    record = {"id": "A9001", "shell": [1, 3, 6], "content": [0, 1, 2]}
    tape = audit.replay_tuple(policy, record, 17)
    np.testing.assert_array_equal(tape["action"], [1, 1, 1])
    np.testing.assert_array_equal(tape["reward"], [0, 1, -1])


def test_tuple_trajectory_ignores_the_human_action_column():
    a = {"id": "A9001", "shell": np.full(8, 3), "content": np.ones(8, int), "action": np.zeros(8, int)}
    b = {**a, "action": np.ones(8, int)}
    left = audit.replay_tuple(audit.core.AuditPolicy("linear", 17, .2), a, 17)
    right = audit.replay_tuple(audit.core.AuditPolicy("linear", 17, .2), b, 17)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_tuple_trajectory_still_predicts_before_feedback():
    record = {"id": "A9001", "shell": np.full(8, 3), "content": np.ones(8, int)}
    tape = audit.replay_tuple(audit.core.AuditPolicy("linear", 17, .2), record, 17)
    assert tape["p_go"][0] == .5
    expected = .5
    parameter = 0.
    for index in range(8):
        assert tape["p_go"][index] == pytest.approx(expected)
        action = int(tape["action"][index])
        parameter += .2 * (action - expected) * action
        expected = 1 / (1 + np.exp(-parameter))


def test_corrected_producer_matches_independent_tuple_numeric_replay():
    """All families on two short artificial people; no real records or study seeds."""
    from drosolearn.graph import Projection
    from drosolearn.human_shell import produce
    from drosolearn.human_shell_rng import produce_corrected

    graph = {"pre": np.array([0, 1, 2, 3, 0, 2]), "post": np.array([0, 0, 1, 1, 2, 2]),
             "counts": np.array([1., 2., 3., 1., 2., 1.]), "n_kc": 4, "n_mbon": 3}
    projection = Projection(graph["pre"], graph["post"], graph["counts"], 4, 3,
                            [str(i) for i in range(4)], [str(i) for i in range(3)], {"scope": "synthetic"})
    rows = []
    for phase in range(4):
        state = 1 + phase % 2
        for exposure in range(6):
            for shell in (3, 4):
                is_reward = audit.core.SEASONS[shell][state - 1] == "R"
                action = int(is_reward) if exposure >= 3 else int(not is_reward)
                rows.append({"trial": len(rows) + 1, "shell": shell, "state": state,
                             "content": 1 if is_reward else 2, "action": action})
    cfg = {"test_ids": ["A9001", "A9002"], "test_seeds": [17, 18], "trials_per_person": len(rows),
           "conditions": list(audit.core.CONDITIONS), "rates": [.02, .07]}
    selection = {"rates": dict.fromkeys(audit.core.CONDITIONS[:8], .02), "calibration_bias": .5}
    records = {pid: rows for pid in cfg["test_ids"]}
    original, original_rows, original_witnesses = produce(records, cfg, projection, selection)
    original_bytes = {name: value.tobytes() for name, value in original.items()}
    arrays, _, witnesses = produce_corrected(original, original_rows, original_witnesses,
                                             records, cfg, projection, selection)
    for name in arrays:
        assert original[name].tobytes() == original_bytes[name]
        audit.identical_array_bytes(arrays[name][0], original[name][0], name)
    for si, seed in enumerate(cfg["test_seeds"]):
        for ci, condition in enumerate(audit.core.CONDITIONS):
            kind = "rewired" if condition.startswith("rewired") else "ann" if condition.startswith("ann") else "real"
            transformed = audit.core.transform_graph(graph, kind,
                                                     seed + (110000 if kind == "rewired" else 120000 if kind == "ann" else 0))
            for pi, pid in enumerate(cfg["test_ids"]):
                policy = audit.core.AuditPolicy(condition, seed, .02, transformed, .5)
                rec = {"id": pid, **{name: np.array([row[name] for row in rows])
                                      for name in ("shell", "state", "action", "content")}}
                initial = policy.witness()
                tape = audit.replay_tuple(policy, rec, seed)
                for field, key in (("p_go", "probabilities"), ("action", "actions"), ("reward", "rewards"),
                                   ("head_step", "head_steps"), ("internal_step", "internal_steps"), ("value_step", "value_steps")):
                    np.testing.assert_allclose(tape[field], arrays[key][1, ci, si, pi], rtol=0, atol=1e-13)
                if pi == 0:
                    for stage, witness in (("initial", initial), ("final", policy.witness())):
                        for name, value in witness.items():
                            key = f"autonomous_{condition}_{stage}_{name}"
                            np.testing.assert_allclose(value, witnesses[key][si], rtol=0, atol=1e-13)
                    for name in original_witnesses:
                        if name.startswith("conditional_") or "_initial_" in name:
                            audit.identical_array_bytes(witnesses[name], original_witnesses[name], name)


def test_preserved_array_comparison_checks_dtype_as_well_as_values():
    with pytest.raises(ValueError, match="Preserved array"):
        audit.identical_array_bytes(np.array([1], np.int64), np.array([1], np.int8), "conditional")


def test_conditional_csv_comparison_retains_literal_line_endings(tmp_path):
    path = tmp_path / "measurements.csv"
    path.write_bytes(b"mode,condition,seed,participant_id\r\nconditional,linear,17,A9001\r\nautonomous,linear,17,A9001\r\n")
    header, conditional, count = audit.conditional_csv_bytes(path)
    assert header.endswith(b"\r\n")
    assert conditional == b"conditional,linear,17,A9001\r\n"
    assert count == 1


def reuse_fixture(tmp_path):
    parent, result = tmp_path / "original", tmp_path / "corrected"
    parent.mkdir(); result.mkdir()
    summary = {"H_prediction": {"supported": False}, "human_metrics": [],
               "conditional_family_mean_nll": {"linear": .5}, "conditional_family_mean_brier": {"linear": .25}}
    for directory in (parent, result):
        for name, value in (("summary.json", summary), ("selection.json", {"rates": {"linear": .02}}),
                            ("cohort_audit.json", {"scope": "synthetic"})):
            (directory / name).write_text(json.dumps(value), encoding="utf-8")
        (directory / "trajectory_metrics.csv").write_bytes(
            b"mode,condition,seed,participant_id,p\r\nconditional,linear,17,A9001,0.5\r\nautonomous,linear,17,A9001,0.7\r\n")
        np.savez(directory / "trajectories.npz", probabilities=np.full((2, 1, 1, 1, 2), .5))
        np.savez(directory / "witnesses.npz", conditional_linear_final_head=np.array([[.1]]),
                 autonomous_linear_initial_head=np.array([[0.]]), autonomous_linear_final_head=np.array([[.2]]))
    cfg = {"conditions": ["linear"], "test_ids": ["A9001"], "test_seeds": [17]}
    return parent, result, cfg


def test_parent_reuse_checks_all_immutable_components(tmp_path):
    parent, result, cfg = reuse_fixture(tmp_path)
    report = audit.assert_parent_reuse(parent, result, cfg)
    assert report["conditional_csv_records"] == 1
    assert report["conditional_witness_fields"] == 1
    assert report["unchanged_autonomous_initial_witness_fields"] == 1


def test_parent_reuse_rejects_semantically_equal_csv_reformatting(tmp_path):
    parent, result, cfg = reuse_fixture(tmp_path)
    path = result / "trajectory_metrics.csv"
    path.write_bytes(path.read_bytes().replace(b"0.5", b".5"))
    with pytest.raises(ValueError, match="conditional CSV"):
        audit.assert_parent_reuse(parent, result, cfg)


def test_parent_reuse_rejects_a_changed_initial_model(tmp_path):
    parent, result, cfg = reuse_fixture(tmp_path)
    path = result / "witnesses.npz"
    with np.load(path) as source:
        values = {key: source[key] for key in source.files}
    values["autonomous_linear_initial_head"] = np.array([[.01]])
    np.savez(path, **values)
    with pytest.raises(ValueError, match="Preserved array"):
        audit.assert_parent_reuse(parent, result, cfg)


def test_registered_repair_preserves_criteria_and_split():
    root = Path(__file__).resolve().parents[1]
    original = audit.core.read_json(root / "configs/human_shell_v1.json")
    corrected = audit.core.read_json(root / "configs/human_shell_rng_v2.json")
    audit.validate_repair_config(corrected, original)
    modified = dict(corrected, equivalence_margin=.2)
    with pytest.raises(ValueError, match="criterion"):
        audit.validate_repair_config(modified, original)


def test_independent_state_inventory_matches_wrapper_on_synthetic_addresses():
    path = Path(__file__).resolve().parents[1] / "scripts/run_human_shell_rng.py"
    spec = importlib.util.spec_from_file_location("shell_rng_wrapper_contract_test", path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    ids, seeds = ["A9001", "A9002"], [17, 18]
    expected = audit.key_audit(ids, seeds)
    assert wrapper.key_inventory({"test_ids": ids, "test_seeds": seeds}) == expected


@pytest.mark.parametrize("inventory,missing,replacement", [
    ("source_files", "scripts/verify_human_shell_rng.py", "docs/literature-review.md"),
    ("parent_files", "results/human_shell_v1/verification.json", "results/unrelated_parent.json"),
])
def test_exact_inventory_rejects_same_count_replacement(inventory, missing, replacement):
    root = Path(__file__).resolve().parents[1]
    original = audit.core.read_json(root / "configs/human_shell_v1.json")
    corrected = audit.core.read_json(root / "configs/human_shell_rng_v2.json")
    corrected[inventory] = [replacement if name == missing else name for name in corrected[inventory]]
    with pytest.raises(ValueError, match="Exact corrected"):
        audit.validate_repair_config(corrected, original)


@pytest.mark.parametrize("inventory", ["source_files", "parent_files"])
def test_exact_inventory_rejects_duplicates(inventory):
    root = Path(__file__).resolve().parents[1]
    original = audit.core.read_json(root / "configs/human_shell_v1.json")
    corrected = audit.core.read_json(root / "configs/human_shell_rng_v2.json")
    corrected[inventory].append(corrected[inventory][0])
    with pytest.raises(ValueError, match="Exact corrected"):
        audit.validate_repair_config(corrected, original)


@pytest.mark.parametrize("changed", ["source_files", "graph_files", "external_files"])
def test_repair_must_retain_original_source_graph_and_human_hashes(changed):
    original = {"source_files": {"old.py": "aaa"}, "graph_files": {"edges.csv": "bbb"},
                "external_files": {"Dataset.zip": {"bytes": 5, "sha256": "ccc"}}}
    corrected = {"source_files": {"old.py": "aaa", "new.py": "ddd"}, "graph_files": {"edges.csv": "bbb"},
                 "external_files": {"Dataset.zip": {"bytes": 5, "sha256": "ccc"}}}
    audit.validate_unchanged_model_source(corrected, original)
    corrected[changed] = dict(corrected[changed])
    key = next(iter(corrected[changed]))
    corrected[changed][key] = "changed"
    with pytest.raises(ValueError, match="Repair changed"):
        audit.validate_unchanged_model_source(corrected, original)
