"""Small invented fixtures for independent no-learning calibration verification."""
from copy import deepcopy
import importlib.util
from itertools import product
import json
from pathlib import Path

import pytest

from test_metric_calibration_run import fixture  # Small seed-3/4, eight-trial invented streams only.


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_metric_calibration.py"
SPEC = importlib.util.spec_from_file_location("independent_metric_calibration_verifier", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def records(actions=(0, 0, 1), rewards=(1, 0, 1), states=(0, 0, 0), phases=None):
    return [{"stream_id": "synthetic/only", "trial": index + 1, "phase": (phases or ["phase1"] * len(actions))[index],
             "state": states[index], "action": action, "reward": rewards[index]}
            for index, action in enumerate(actions)]


@pytest.mark.parametrize("environment", VERIFY.ENVIRONMENTS)
@pytest.mark.parametrize("policy,rate", [("iid_fair", .5), ("sticky", .9)])
def test_reward_independent_fair_and_sticky_joint_count_ratios(environment, policy, rate):
    rows = VERIFY.replay_stream(11, "fixed", policy, environment, 12, 6)
    counts = VERIFY.expected_counts(rows, policy, environment)
    for cell in counts["by_feedback"].values():
        assert cell["ratio_of_expected_counts"] == pytest.approx(rate)
        assert cell["expected_stay"] + cell["expected_switch"] == pytest.approx(cell["expected_denominator"])
    assert sum(cell["expected_denominator"] for cell in counts["by_feedback"].values()) == pytest.approx(10.)


@pytest.mark.parametrize("environment,stay0,stay1", [
    ("deterministic", .8, .2),
    ("probabilistic", 13 / 17, .5),
    ("uninformative", .68, .68),
    ("swapped", .5, 13 / 17),
])
def test_fixed_state_biased_policy_can_have_feedback_differences_without_updates(environment, stay0, stay1):
    rows = VERIFY.replay_stream(11, "fixed", "iid_biased", environment, 12, 6)
    counts = VERIFY.expected_counts(rows, "iid_biased", environment)["by_feedback"]
    assert counts["0"]["ratio_of_expected_counts"] == pytest.approx(stay0)
    assert counts["1"]["ratio_of_expected_counts"] == pytest.approx(stay1)


@pytest.mark.parametrize("environment", VERIFY.ENVIRONMENTS)
def test_two_equal_reversal_phases_mask_biased_policy_feedback_differences(environment):
    rows = VERIFY.replay_stream(11, "reversal", "iid_biased", environment, 12, 6)
    counts = VERIFY.expected_counts(rows, "iid_biased", environment)["by_feedback"]
    for cell in counts.values():
        assert cell["ratio_of_expected_counts"] == pytest.approx(.68)
    phase1 = VERIFY.expected_counts(rows, "iid_biased", environment, stratum="phase1")["by_feedback"]
    phase2 = VERIFY.expected_counts(rows, "iid_biased", environment, stratum="phase2")["by_feedback"]
    assert phase1["0"]["expected_denominator"] == pytest.approx(phase2["1"]["expected_denominator"])
    assert phase1["1"]["ratio_of_expected_counts"] == pytest.approx(phase2["0"]["ratio_of_expected_counts"])


def test_conditional_actual_feedback_counts_match_hand_written_adjacencies():
    summary, ledger = VERIFY.independent_counts(records())
    assert summary["by_feedback"]["1"] == {"denominator": 1, "stay": 1, "switch": 0,
                                              "stay_rate": 1., "switch_rate": 0.}
    assert summary["by_feedback"]["0"] == {"denominator": 1, "stay": 0, "switch": 1,
                                              "stay_rate": 0., "switch_rate": 1.}
    assert [row["feedback"] for row in ledger] == ["1", "0"]
    assert [row["stay"] for row in ledger] == ["1", "0"]


@pytest.mark.parametrize("schedule,reasons", [
    ("fixed", "phase_boundary"), ("reversal", "phase_boundary|state_boundary"),
])
def test_original_global_trial_ids_and_every_boundary_reason_are_preserved(schedule, reasons):
    rows = VERIFY.replay_stream(11, schedule, "iid_fair", "deterministic", 6, 3)
    summary, ledger = VERIFY.independent_counts(rows)
    assert summary["candidates"] == 5
    assert summary["eligible"] == 4
    assert summary["excluded"] == 1
    assert ledger[2]["trial"] == "3"
    assert ledger[2]["next_trial"] == "4"
    assert ledger[2]["reasons"] == reasons
    assert ledger[2]["stay"] == ""
    phase2, phase2_ledger = VERIFY.independent_counts(rows, stratum="phase2")
    assert phase2["trials"] == 3 and phase2["eligible"] == 2
    assert [row["trial"] for row in phase2_ledger] == ["4", "5"]


def test_zero_actual_denominator_is_missing_and_never_zero_rate():
    summary, _ = VERIFY.independent_counts(records((1, 1, 1), (0, 0, 0)))
    assert summary["by_feedback"]["1"] == {"denominator": 0, "stay": 0, "switch": 0,
                                             "stay_rate": None, "switch_rate": None}


def test_expected_count_ratio_is_not_the_expected_random_denominator_ratio():
    """Three trials with iid p(1)=.8 and deterministic reward for action 0."""
    expected = VERIFY.expected_counts(records(), "iid_biased", "deterministic")["by_feedback"]["1"]
    weighted_ratio, positive_denominator_probability = 0., 0.
    for actions in product((0, 1), repeat=3):
        probability = .8 ** sum(actions) * .2 ** (3 - sum(actions))
        denominator = sum(action == 0 for action in actions[:-1])
        numerator = sum(actions[index] == actions[index + 1] == 0 for index in range(2))
        if denominator:
            positive_denominator_probability += probability
            weighted_ratio += probability * numerator / denominator
    assert expected["expected_denominator"] == pytest.approx(.4)
    assert expected["expected_stay"] == pytest.approx(.08)
    assert expected["ratio_of_expected_counts"] == pytest.approx(.2)
    conditional_expected_empirical_ratio = weighted_ratio / positive_denominator_probability
    assert conditional_expected_empirical_ratio == pytest.approx(7 / 45)
    assert conditional_expected_empirical_ratio != pytest.approx(expected["ratio_of_expected_counts"])


def test_action_and_reward_uniforms_are_shared_across_registered_environments_and_policies():
    reference = VERIFY.replay_stream(11, "fixed", "iid_fair", "deterministic", 12, 6)
    for schedule, policy, environment in product(VERIFY.SCHEDULES, VERIFY.POLICIES, VERIFY.ENVIRONMENTS):
        rows = VERIFY.replay_stream(11, schedule, policy, environment, 12, 6)
        assert [row["action_uniform"] for row in rows] == [row["action_uniform"] for row in reference]
        assert [row["reward_uniform"] for row in rows] == [row["reward_uniform"] for row in reference]
    assert [row["action_uniform"] for row in reference] != [row["reward_uniform"] for row in reference]


def test_changing_reward_environment_cannot_change_nonlearning_action_path():
    for policy in VERIFY.POLICIES:
        paths = [VERIFY.replay_stream(11, "fixed", policy, environment, 12, 6)
                 for environment in VERIFY.ENVIRONMENTS]
        assert all([row["action"] for row in path] == [row["action"] for row in paths[0]] for path in paths)


def test_expected_counts_do_not_read_realized_reward_or_action_values():
    original = VERIFY.replay_stream(11, "fixed", "iid_biased", "probabilistic", 12, 6)
    changed = deepcopy(original)
    for record in changed:
        record["action"], record["reward"] = 1 - record["action"], 1 - record["reward"]
    assert VERIFY.expected_counts(original, "iid_biased", "probabilistic") == VERIFY.expected_counts(
        changed, "iid_biased", "probabilistic")


@pytest.mark.parametrize("arguments", [
    (True, "fixed", "iid_fair", "deterministic", 6, 3),
    (-1, "fixed", "iid_fair", "deterministic", 6, 3),
    (11, "unknown", "iid_fair", "deterministic", 6, 3),
    (11, "fixed", "learner", "deterministic", 6, 3),
    (11, "fixed", "iid_fair", "unknown", 6, 3),
    (11, "fixed", "iid_fair", "deterministic", 6, 0),
    (11, "fixed", "iid_fair", "deterministic", 6, 6),
    (11, "fixed", "iid_fair", "deterministic", 6., 3),
])
def test_bad_calibration_coordinates_are_not_silently_coerced(arguments):
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.replay_stream(*arguments)


def test_exact_counts_and_roundoff_only_analytic_comparison_are_distinct():
    assert not VERIFY.exact({"count": True}, {"count": 1})
    assert not VERIFY.close({"count": 1.}, {"count": 1})
    assert VERIFY.close({"probability": .3}, {"probability": .1 + .2})
    assert not VERIFY.close({"probability": .31}, {"probability": .3})


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"rate":NaN}', '{"rate":Infinity}'])
def test_duplicate_or_nonfinite_json_is_rejected(tmp_path, payload):
    path = tmp_path / "synthetic.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.read_json(path)


def test_finite_json_roundtrip_has_no_type_coercion(tmp_path):
    path = tmp_path / "synthetic.json"
    value = {"count": 2, "rate": None, "false_claim": False}
    path.write_text(json.dumps(value), encoding="utf-8")
    assert VERIFY.exact(VERIFY.read_json(path), value)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@pytest.fixture
def completed(fixture):
    from test_metric_calibration_run import register, run
    register(fixture)
    result = run(fixture)
    assert result["status"] == "completed"
    return fixture


def checked(completed):
    return VERIFY.verify(completed["registration"], completed["out"], root=completed["root"],
                         config_path=completed["config_path"])


def refresh_hash(completed, filename):
    path = completed["out"] / "artifact_hashes.json"
    hashes = VERIFY.read_json(path)
    hashes[filename] = VERIFY.digest(completed["out"] / filename)
    save_json(path, hashes)


def test_independent_end_to_end_replays_only_invented_short_streams(completed):
    result = checked(completed)
    assert result["status"] == "passed"
    assert result["actual_trials_verified"] == 384
    assert result["streams_verified"] == 48
    assert result["candidate_pairs_verified"] == 48 * 7
    assert result["eligible_pairs_verified"] == 48 * 6
    assert result["strata_verified"] == 48 * 3
    assert result["summary_cases_verified"] == 72
    assert result["primary_artifacts_verified"] == 6
    assert result["parameter_updates"] == 0
    assert result["human_or_species_comparison"] is False


@pytest.mark.parametrize("target", ["source", "parent", "config"])
def test_independent_verifier_rejects_frozen_input_byte_changes(completed, target):
    path = {"source": completed["root"] / "src/drosolearn/metric_calibration.py",
            "parent": completed["root"] / completed["cfg"]["parent_files"][0],
            "config": completed["config_path"]}[target]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


@pytest.mark.parametrize("filename", ["trials.csv", "ledger.csv", "metrics.json", "summary.json", "metadata.json"])
def test_artifact_digest_change_is_rejected(completed, filename):
    path = completed["out"] / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


@pytest.mark.parametrize("target", ["count", "expected_count", "ratio", "removed_stream", "overclaim"])
def test_rehashing_wrong_metric_cannot_replace_independent_recalculation(completed, target):
    path = completed["out"] / "metrics.json"
    value = VERIFY.read_json(path)
    stream = value["streams"][0]
    if target == "count":
        stream["strata"]["all"]["empirical"]["by_feedback"]["1"]["stay"] += 1
    elif target == "expected_count":
        stream["strata"]["all"]["analytic"]["by_feedback"]["1"]["expected_denominator"] += .01
    elif target == "ratio":
        stream["strata"]["all"]["analytic"]["by_feedback"]["1"]["ratio_of_expected_counts"] += .01
    elif target == "removed_stream":
        value["streams"].pop()
    else:
        value["learning_proven"] = True
    save_json(path, value)
    refresh_hash(completed, "metrics.json")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


@pytest.mark.parametrize("target", ["mean", "defined_seeds", "claim", "parameter_updates"])
def test_rehashed_summary_mean_denominator_and_claim_tampering_is_rejected(completed, target):
    path = completed["out"] / "summary.json"
    value = VERIFY.read_json(path)
    if target == "mean":
        value["cases"][0]["by_feedback"]["0"]["mean_defined_seed_ratio"] += .01
    elif target == "defined_seeds":
        value["cases"][0]["by_feedback"]["0"]["defined_seeds"] = 1
    elif target == "claim":
        value["analytic_claims"]["H1_no_learning_feedback_difference"]["reference"] = -.6
    else:
        value["parameter_updates"] = 1
    save_json(path, value)
    refresh_hash(completed, "summary.json")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


def test_rehashed_actual_action_tampering_is_rejected_by_rng_replay(completed):
    path = completed["out"] / "trials.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    values = lines[1].split(",")
    values[8] = str(1 - int(values[8]))
    lines[1] = ",".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    refresh_hash(completed, "trials.csv")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


def test_rehashed_ledger_cannot_include_the_excluded_boundary(completed):
    path = completed["out"] / "ledger.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    values = lines[4].split(",")
    assert values[6] == "phase_boundary"
    values[5:7] = ["1", ""]
    values[8] = "1"
    lines[4] = ",".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    refresh_hash(completed, "ledger.csv")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


def test_independent_verifier_rejects_added_trial_even_with_valid_hash(completed):
    path = completed["out"] / "trials.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
    refresh_hash(completed, "trials.csv")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


@pytest.mark.parametrize("change", ["future", "naive"])
def test_independent_chronology_gate_survives_rehashed_registration(completed, change):
    reg = VERIFY.read_json(completed["registration"])
    reg["registered_at_utc"] = "2099-01-01T00:00:00+00:00" if change == "future" else "2000-01-01T00:00:00"
    save_json(completed["registration"], reg)
    (completed["out"] / "registration.json").write_bytes(completed["registration"].read_bytes())
    metadata = VERIFY.read_json(completed["out"] / "metadata.json")
    metadata["registration_sha256"] = VERIFY.digest(completed["registration"])
    save_json(completed["out"] / "metadata.json", metadata)
    refresh_hash(completed, "registration.json")
    refresh_hash(completed, "metadata.json")
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)


def test_verifier_cli_records_failure_without_overwriting_previous_output(completed, tmp_path):
    path = completed["out"] / "metrics.json"
    path.write_bytes(path.read_bytes() + b"\n")
    output = tmp_path / "verification.json"
    argv = ["--root", str(completed["root"]), "--config", str(completed["config_path"]),
            "--registration", str(completed["registration"]), "--result-dir", str(completed["out"]), "--out", str(output)]
    assert VERIFY.main(argv) == 1
    result = VERIFY.read_json(output)
    assert result["status"] == "failed" and result["errors"]
    before = output.read_bytes()
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.main(argv)
    assert output.read_bytes() == before


def test_synthetic_scope_cannot_smuggle_registered_scientific_seeds(completed):
    config = VERIFY.read_json(completed["config_path"])
    config["seeds"] = [9001]
    save_json(completed["config_path"], config)
    with pytest.raises(VERIFY.VerificationError):
        checked(completed)
