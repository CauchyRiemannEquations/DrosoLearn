"""Independent reductions and policy checks on invented source records only."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("behavior_independent", Path(__file__).resolve().parents[1] / "scripts/verify_shell_behavior.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def record():
    shells = np.tile(np.arange(1, 7), 36)
    exposures = np.repeat(np.arange(36), 6)
    states = (exposures // 9) % 2 + 1
    content = np.zeros(216, dtype=int)
    for i, (cue, state) in enumerate(zip(shells, states)):
        season = audit.core.SEASONS[int(cue)][int(state) - 1]
        content[i] = (1 if season == "R" else 2 if season == "P" else 0) if exposures[i] % 3 else 0
    return {"shell": shells, "state": states, "content": content,
            "action": (np.arange(216) % 3 == 0).astype(int)}


def fixture_config():
    return {"test_ids": ["A9001", "A9002"], "test_seeds": [41, 42], "synthetic_seeds": [71, 72],
            "conditions": list(audit.CONDITIONS), "trials_per_person": 216}


def as_rows(r):
    return [{key: int(r[key][i]) for key in ("shell", "state", "content", "action")} for i in range(len(r["shell"]))]


def test_real_config_validates_without_drawing_scientific_streams():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/shell_behavior_v1.json").read_text(encoding="utf8"))
    original = json.loads((root / "configs/human_shell_rng_v2.json").read_text(encoding="utf8"))
    audit.validate_science(cfg, original)


@pytest.mark.parametrize("key,value", [("synthetic_seeds", [16001]), ("q_alpha", .3), ("q_beta", 40.),
                                       ("rng_namespace", 0), ("trials_per_person", 432),
                                       ("scope", "confirmatory_human_similarity"), ("test_ids", ["A9001"] * 29)])
def test_prospective_scope_and_budget_cannot_change(key, value):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/shell_behavior_v1.json").read_text(encoding="utf8"))
    original = json.loads((root / "configs/human_shell_rng_v2.json").read_text(encoding="utf8"))
    cfg[key] = value
    with pytest.raises(ValueError): audit.validate_science(cfg, original)


def test_old_person_order_cannot_be_reshuffled():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/shell_behavior_v1.json").read_text(encoding="utf8"))
    original = json.loads((root / "configs/human_shell_rng_v2.json").read_text(encoding="utf8"))
    cfg["test_ids"] = list(reversed(cfg["test_ids"]))
    with pytest.raises(ValueError): audit.validate_science(cfg, original)


@pytest.mark.parametrize("seed,pid", [(True, "A9001"), (-1, "A9001"), (71., "A9001"),
                                      (71, "9001"), (71, "A90001"), (71, "A9001 "), (71, "A\u0669\u0660\u0660\u0661")])
def test_canonical_rng_address(seed, pid):
    with pytest.raises(ValueError): audit.generator(seed, pid)


def test_tuple_stream_matches_contract_and_separates_equal_sum_cells():
    np.testing.assert_array_equal(audit.generator(71, "A9001").random(12),
                                  np.random.default_rng(np.random.SeedSequence([2026101902, 71, 9001])).random(12))
    assert not np.array_equal(audit.generator(71, "A9002").random(12), audit.generator(72, "A9001").random(12))


def test_phase_partition_and_counts_exclude_each_boundary():
    r = record(); groups = audit.phase_groups(r)
    assert len(groups) == 24 and all(len(loc) == 9 for _, _, loc in groups)
    actions = np.ones((1, 1, 216), dtype=int)
    counts = audit.transition_counts({"A9001": r}, ["A9001"], actions)
    assert counts[..., 0].sum() == 6 * 4 * 8 == 192
    assert counts[..., 1].sum() == counts[..., 2].sum() == 192
    assert counts[..., 0, :].sum() == 0  # NoGo category empty when all actions Go.


def test_hidden_nogo_event_does_not_reveal_environment_content():
    left = record(); right = copy.deepcopy(left); right["content"][:] = 2
    actions = np.zeros((1, 1, 216), dtype=int)
    a = audit.transition_counts({"A9001": left}, ["A9001"], actions)
    b = audit.transition_counts({"A9001": right}, ["A9001"], actions)
    np.testing.assert_array_equal(a, b)
    assert a[..., 0, 0].sum() == a[..., 0, 2].sum() == 192
    assert a[..., 0, 1].sum() == 0


def test_next_go_and_stay_are_distinct_statistics():
    r = record(); actions = np.zeros((1, 1, 216), int)
    cue1 = np.flatnonzero(r["shell"] == 1)
    actions[0, 0, cue1] = np.arange(36) % 2
    counts = audit.transition_counts({"A9001": r}, ["A9001"], actions)
    assert counts[0, 0, 0, :, 0, 1].sum() > 0
    assert counts[0, 0, 0, :, 0, 2].sum() == 0


def test_source_phase_mismatch_or_nonbinary_actions_are_rejected():
    r = record(); r["state"][:] = 1
    with pytest.raises(ValueError): audit.phase_groups(r)
    with pytest.raises(ValueError): audit.transition_counts({"A9001": record()}, ["A9001"], np.full((1, 1, 216), 2))


def test_q_probability_precedes_signed_observed_reward_update(monkeypatch):
    class AlwaysGo:
        def random(self, n): return np.zeros(n)
    monkeypatch.setattr(audit, "generator", lambda *args: AlwaysGo())
    r = {"shell": np.ones(3, int), "content": np.array([1, 0, 2])}
    p, a, rewards, steps, table = audit.replay_policy(r, 71, "A9001", "q")
    np.testing.assert_array_equal(a, [1, 1, 1]); np.testing.assert_array_equal(rewards, [1, 0, -1])
    assert p[0] == .5 and p[1] == pytest.approx(audit.core.logistic(.4))
    assert p[2] == pytest.approx(audit.core.logistic(.36))
    assert table[0, 1] == pytest.approx(-.019)
    np.testing.assert_allclose(steps, [.1, .01, .109])


def test_nogo_never_reads_hidden_reward_in_q(monkeypatch):
    class AlwaysNoGo:
        def random(self, n): return np.ones(n)
    monkeypatch.setattr(audit, "generator", lambda *args: AlwaysNoGo())
    a = {"shell": np.ones(3, int), "content": np.array([999, 999, 999])}
    p, chosen, rewards, steps, table = audit.replay_policy(a, 71, "A9001", "q")
    np.testing.assert_array_equal(chosen, [0, 0, 0]); np.testing.assert_array_equal(rewards, [0, 0, 0])
    assert not np.any(steps) and not np.any(table)


@pytest.mark.parametrize("policy", ["uniform", "persistence"])
def test_reward_unaware_policies_ignore_sign_intervention(policy):
    left = audit.replay_policy(record(), 71, "A9001", policy)
    right = audit.replay_policy(record(), 71, "A9001", policy, flip=True)
    np.testing.assert_array_equal(left[0], right[0]); np.testing.assert_array_equal(left[1], right[1])
    np.testing.assert_array_equal(left[2], -right[2]); assert not np.any(left[3])


@pytest.fixture(scope="module")
def produced():
    # Producer imports are restricted to test-side fixture construction.
    from drosolearn import shell_behavior as producer
    cfg = fixture_config(); ids = cfg["test_ids"]
    records = {pid: record() for pid in ids}; rowsource = {pid: as_rows(records[pid]) for pid in ids}
    old = np.random.default_rng(4011).integers(0, 2, (11, 2, 2, 216), dtype=np.int8)
    human = np.stack([records[pid]["action"] for pid in ids])
    all_actions = np.concatenate([human[None], old.reshape(22, 2, 216)])
    a = {"transition_counts": producer.transition_counts(rowsource, ids, all_actions)}
    shape = (3, 2, 2, 216)
    for name, dtype in (("probabilities", float), ("actions", np.int8), ("rewards", np.int8), ("value_steps", float)):
        a["synthetic_" + name] = np.empty(shape, dtype=dtype)
        a["counterfactual_" + name] = np.empty(shape[1:], dtype=dtype)
    a["final_value_tables"] = np.zeros((2, 2, 2, 6, 2)); a["synthetic_metrics"] = np.empty((3, 2, 2, 3))
    for si, seed in enumerate(cfg["synthetic_seeds"]):
        for pi, pid in enumerate(ids):
            for fi, policy in enumerate(audit.POLICIES):
                tape = producer.simulate(rowsource[pid], seed, pid, policy)
                for field, value in zip(("probabilities", "actions", "rewards", "value_steps"), tape[:4]):
                    a["synthetic_" + field][fi, si, pi] = value
                a["synthetic_metrics"][fi, si, pi] = producer.metric_vector(rowsource[pid], tape[1])
                if policy == "q": a["final_value_tables"][0, si, pi] = tape[4]
            tape = producer.simulate(rowsource[pid], seed, pid, "q", flip=True)
            for field, value in zip(("probabilities", "actions", "rewards", "value_steps"), tape[:4]):
                a["counterfactual_" + field][si, pi] = value
            a["final_value_tables"][1, si, pi] = tape[4]
    a["scripted_actions"] = a["synthetic_actions"][2].copy()
    a["scripted_metrics"] = np.stack([np.stack([producer.metric_vector(rowsource[pid], a["scripted_actions"][si, pi])
                                              for pi, pid in enumerate(ids)]) for si in range(2)])
    rows = producer.measurement_rows(cfg, a)
    return cfg, records, old, a, rows, producer.summarize(cfg, a, rows)


def test_full_synthetic_source_producer_independent_agreement(produced):
    cfg, records, old, a, rows, summary = produced
    result = audit.verify_against_records(cfg, records, old, a, rows, summary)
    assert result["independently_replayed_synthetic_choices"] == 3456
    assert result["saved_action_streams"] == 23
    assert result["within_phase_pairs"] == 23 * 2 * 192
    assert result["complete_strata_including_empty_cells"] == 23 * 6 * 4 * 4


def test_empty_denominator_rows_preserved_as_null(produced):
    cfg, records, old, a, rows, summary = produced
    empty = [row for row in rows if row["kind"] == "saved_transition" and row["denominator"] == 0]
    assert empty and all(row["numerator"] == 0 and row["value"] is None for row in empty)
    csv_rows = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    audit.verify_against_records(cfg, records, old, a, csv_rows, summary)


@pytest.mark.parametrize("field", ["transition_counts", "synthetic_probabilities", "synthetic_actions", "synthetic_rewards",
                                   "synthetic_value_steps", "counterfactual_actions", "counterfactual_rewards",
                                   "final_value_tables", "synthetic_metrics", "scripted_metrics", "scripted_actions"])
def test_corrupted_arrays_rejected(produced, field):
    cfg, records, old, original, rows, summary = produced
    a = copy.deepcopy(original); a[field].flat[0] += 1
    with pytest.raises(ValueError): audit.verify_against_records(cfg, records, old, a, rows, summary)


def test_corrupted_measurement_and_overclaim_rejected(produced):
    cfg, records, old, a, rows, summary = produced
    wrong_rows = copy.deepcopy(rows); wrong_rows[0]["denominator"] += 1
    with pytest.raises(ValueError): audit.verify_against_records(cfg, records, old, a, wrong_rows, summary)
    wrong_summary = copy.deepcopy(summary); wrong_summary["metric_identification"]["scripted_policy_is_eligible_fair_learning_control"] = True
    with pytest.raises(ValueError): audit.verify_against_records(cfg, records, old, a, rows, wrong_summary)


def test_tape_equality_does_not_turn_copy_into_learner(produced):
    _, _, _, a, _, summary = produced
    np.testing.assert_array_equal(a["scripted_actions"], a["synthetic_actions"][2])
    np.testing.assert_array_equal(a["scripted_metrics"], a["synthetic_metrics"][2])
    assert summary["metric_identification"]["scripted_policy_online_updates"] == 0
    assert summary["metric_identification"]["scripted_policy_is_eligible_fair_learning_control"] is False
    assert summary["metric_identification"]["three_summaries_uniquely_identify_reward_learning"] is False
