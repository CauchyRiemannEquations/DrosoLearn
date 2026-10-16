"""Independent ablation checks use only invented seed17/18 engineering tasks."""
import ast
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_learning_ablation.py"
SPEC = importlib.util.spec_from_file_location("independent_learning_ablation_verifier", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_runtime_imports_only_standard_libraries_numpy_and_explicit_independent_verifier_file():
    tree = ast.parse(SCRIPT.read_text(encoding="utf8"))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imports == {"__future__", "datetime", "pathlib"}
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert names == {"argparse", "importlib.util", "json", "re", "numpy"}
    assert 'with_name("verify_stochastic_feedback.py")' in SCRIPT.read_text(encoding="utf8")


def arguments(*, linear=True, head_enabled=True, internal_enabled=True):
    stimulus = np.zeros((2, 7))
    stimulus[:, 0] = [.25, .5]
    encoder = np.empty((0, 7)) if linear else np.asarray([[1., 0., 0., 0., 0., 0., 0.]])
    return {"stimulus": stimulus, "action_uniform": .1, "reward_uniform": .1, "larger": True,
            "encoder": encoder, "bias": np.empty(0) if linear else np.zeros(1),
            "pre": np.empty(0, dtype=int) if linear else np.asarray([0]),
            "post": np.empty(0, dtype=int) if linear else np.asarray([0]),
            "weights": np.empty(0) if linear else np.asarray([.7]),
            "head": np.asarray([.6, 0., 0., 0., 0., 0., 0.]) if linear else np.asarray([.6]),
            "scale": .25, "linear": linear, "head_enabled": head_enabled,
            "internal_enabled": internal_enabled, "head_lr": .1, "internal_lr": .01,
            "upper": np.empty(0) if linear else np.asarray([1.4])}


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("branch", VERIFY.BRANCHES)
def test_factorial_masks_change_only_requested_available_parameter_set(linear, branch):
    head_enabled, internal_enabled = VERIFY.MASKS[branch]
    args = arguments(linear=linear, head_enabled=head_enabled, internal_enabled=internal_enabled)
    row, weights, head = VERIFY.masked_step(**args)
    assert row["head_update_called"] == int(head_enabled)
    assert row["internal_update_called"] == int(internal_enabled and not linear)
    assert (row["readout_step_l2"] > 0) == head_enabled
    assert (row["internal_step_l2"] > 0) == (internal_enabled and not linear)
    if not head_enabled:
        np.testing.assert_array_equal(head, args["head"])
    if not internal_enabled or linear:
        np.testing.assert_array_equal(weights, args["weights"])


def test_internal_only_update_uses_old_head_and_original_fixed_scale():
    args = arguments(linear=False, head_enabled=False, internal_enabled=True)
    row, weights, head = VERIFY.masked_step(**args)
    p = 1 / (1 + np.exp(-.42))
    factor = (1 - p) * .5
    assert weights[0] == pytest.approx(.7 + .01 * factor * .6)
    np.testing.assert_array_equal(head, [.6])
    assert row["optimizer_factor"] == pytest.approx(factor)


def test_simultaneous_updates_do_not_use_new_readout_in_internal_eligibility():
    args = arguments(linear=False)
    row, weights, head = VERIFY.masked_step(**args)
    assert head[0] > .6
    assert weights[0] == pytest.approx(.7 + .01 * row["optimizer_factor"] * .6)
    assert weights[0] != pytest.approx(.7 + .01 * row["optimizer_factor"] * head[0], abs=1e-9)


def test_frozen_branch_retains_learned_nonzero_state_instead_of_resetting_to_chance():
    args = arguments(linear=False, head_enabled=False, internal_enabled=False)
    row, weights, head = VERIFY.masked_step(**args)
    np.testing.assert_array_equal(weights, [.7])
    np.testing.assert_array_equal(head, [.6])
    assert row["p_right"] > .5
    assert row["readout_step_l2"] == row["internal_step_l2"] == 0


@pytest.mark.parametrize("reward_uniform", [0., .2, .8, .999999])
def test_linear_internal_only_is_exactly_frozen_even_with_feedback(reward_uniform):
    left = arguments(head_enabled=False, internal_enabled=True)
    right = arguments(head_enabled=False, internal_enabled=False)
    left["reward_uniform"] = right["reward_uniform"] = reward_uniform
    a, aw, ah = VERIFY.masked_step(**left)
    b, bw, bh = VERIFY.masked_step(**right)
    assert a == b
    np.testing.assert_array_equal(aw, bw)
    np.testing.assert_array_equal(ah, bh)


@pytest.mark.parametrize("larger", [False, True])
@pytest.mark.parametrize("action_uniform", [.1, .9])
def test_actual_reward_threshold_follows_current_phase_and_actual_choice(larger, action_uniform):
    args = arguments()
    args.update(larger=larger, action_uniform=action_uniform)
    first, _, _ = VERIFY.masked_step(**args)
    q = first["q_reward"]
    args["reward_uniform"] = q
    exact, _, _ = VERIFY.masked_step(**args)
    assert exact["reward"] == 0
    args["reward_uniform"] = np.nextafter(q, 0.)
    below, _, _ = VERIFY.masked_step(**args)
    assert below["reward"] == 1


def test_scientific_config_is_validated_without_fitting_or_generating_inputs():
    cfg = VERIFY.read_json(ROOT / "configs/learning_ablation_v1.json")
    VERIFY.config_check(cfg)
    assert cfg["seeds"] == list(range(12001, 12011))


@pytest.mark.parametrize("seed", [12001, 12010])
def test_scientific_seeds_cannot_be_engineering_training_fixtures(seed):
    cfg = VERIFY.read_json(ROOT / "configs/learning_ablation_v1.json")
    cfg.update(scope="integration_test", seeds=[seed])
    with pytest.raises(ValueError):
        VERIFY.config_check(cfg)


@pytest.mark.parametrize("name,comparison", [("H_I", "full_minus_head_only"), ("H_H", "full_minus_internal_only")])
@pytest.mark.parametrize("values,expected", [([.0625] * 8 + [0., 0.], True),
                                            ([.125] * 7 + [0., 0., 0.], False),
                                            ([.01] * 10, False), ([0.] * 10, False)])
def test_both_component_criteria_keep_zero_negative_and_all_seed_denominators(name, comparison, values, expected):
    # Ten invented scalar differences: no stimulus generation or model fitting.
    pairs = [{"seed": 17 + i, "difference": value} for i, value in enumerate(values)]
    result = VERIFY.component_hypothesis(name, comparison, pairs, VERIFY.CRITERIA)
    assert result["criterion_met"] is expected
    assert result["n_seeds"] == 10
    assert result["positive_seed_count"] == sum(value > 0 for value in values)
    assert result["all_seed_differences"] == pairs


@pytest.fixture(scope="module")
def audited_fixture(tmp_path_factory):
    from test_learning_ablation_run import make_fixture
    from drosolearn import learning_ablation as producer
    monkeypatch = pytest.MonkeyPatch()
    try:
        fixture = make_fixture(tmp_path_factory.mktemp("independent-ablation"), monkeypatch)
        producer.register(fixture["root"], fixture["config"], fixture["registration"])
        producer.run(fixture["root"], fixture["config"], fixture["registration"], fixture["out"])
        report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"])
        assert report["status"] == "passed", report["errors"]
        yield fixture
    finally:
        monkeypatch.undo()


def clone(original, tmp_path):
    root = tmp_path / "rehashed_mutant"
    shutil.copytree(original["root"], root)
    return {**original, "root": root, "registration": root / original["registration"].relative_to(original["root"]),
            "out": root / original["out"].relative_to(original["root"])}


def rehash(fixture, relative):
    path = fixture["out"] / "artifact_hashes.json"
    manifest = VERIFY.read_json(path)
    manifest[relative] = VERIFY.digest(fixture["out"] / relative)
    path.write_text(json.dumps(manifest), encoding="utf8")


def replace_json(fixture, relative, value):
    (fixture["out"] / relative).write_text(json.dumps(value, allow_nan=False), encoding="utf8")
    rehash(fixture, relative)


def read_csv(fixture, relative):
    with (fixture["out"] / relative).open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def replace_csv(fixture, relative, rows):
    with (fixture["out"] / relative).open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rehash(fixture, relative)


def rejected(fixture):
    report = VERIFY.verify(fixture["root"], fixture["registration"], fixture["out"])
    assert report["status"] == "failed" and report["passed"] is False and report["errors"]


def test_complete_independent_replay_preserves_primary_bytes_and_exact_shared_prefix_budgets(audited_fixture):
    manifest = VERIFY.read_json(audited_fixture["out"] / "artifact_hashes.json")
    report = VERIFY.verify(audited_fixture["root"], audited_fixture["registration"], audited_fixture["out"])
    assert report["passed"], report["errors"]
    counts = report["counts"]
    assert counts["actual_choice_feedback_trials"] == 320
    assert counts["prefix_training_trials"] == 64 and counts["branch_training_trials"] == 256
    assert counts["head_update_calls"] == 192 and counts["internal_update_calls"] == 144
    assert counts["prefix_clones"] == 32 and counts["checkpoints"] == 120
    assert counts["linear_equivalent_checkpoint_pairs"] == 12 and counts["linear_equivalent_trial_pairs"] == 32
    assert counts["fixed_initialization_field_checks"] == 1080
    assert report["learning_curves_sha256"] == VERIFY.digest(audited_fixture["out"] / "learning_curves.csv")
    assert report["artifact_hashes_sha256"] == VERIFY.digest(audited_fixture["out"] / "artifact_hashes.json")
    assert all(VERIFY.digest(audited_fixture["out"] / p) == sha for p, sha in manifest.items())
    assert not (audited_fixture["out"] / "verification.json").exists()


@pytest.mark.parametrize("field", ["action", "reward", "q_reward", "p_right", "chosen_count", "reward_uniform",
                                  "head_enabled", "internal_enabled", "head_update_called", "internal_update_called",
                                  "head_updates", "internal_updates", "cumulative_head_updates", "cumulative_internal_updates"])
def test_rehashed_actual_reward_mask_and_local_vs_inherited_counter_corruption_is_rejected(audited_fixture, tmp_path, field):
    fixture = clone(audited_fixture, tmp_path)
    name = "seed_17/trials.csv"
    rows = read_csv(fixture, name)
    row = next(r for r in rows if r["model"] == "real" and r["branch"] == "internal_only" and r["trial"] == "2")
    if field in {"action", "reward", "head_enabled", "internal_enabled", "head_update_called", "internal_update_called"}:
        row[field] = 1 - int(row[field])
    elif field in {"chosen_count", "head_updates", "internal_updates", "cumulative_head_updates", "cumulative_internal_updates"}:
        row[field] = int(row[field]) + 1
    else:
        row[field] = float(row[field]) + .013
    replace_csv(fixture, name, rows)
    rejected(fixture)


@pytest.mark.parametrize("field", sorted(VERIFY.SNAPSHOT_FIELDS))
def test_every_rehashed_clone_initialization_bound_scale_activity_counter_and_evaluation_field_is_replayed(audited_fixture, tmp_path, field):
    fixture = clone(audited_fixture, tmp_path)
    name = "seed_17/real/internal_only/phase2_0000.npz"
    with np.load(fixture["out"] / name, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays[field].flat[0] += 1 if np.issubdtype(arrays[field].dtype, np.integer) else .013
    np.savez_compressed(fixture["out"] / name, **arrays)
    rehash(fixture, name)
    rejected(fixture)


@pytest.mark.parametrize("field", ["action_phase2_uniforms", "reward_phase1_uniforms", "train_phase1_nuisance", "train_phase2_nuisance"])
def test_rehashed_shared_inputs_and_uniforms_are_independently_regenerated(audited_fixture, tmp_path, field):
    fixture = clone(audited_fixture, tmp_path)
    name = "seed_17/stimuli.npz"
    with np.load(fixture["out"] / name, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays[field].flat[0] += .013
    np.savez_compressed(fixture["out"] / name, **arrays)
    rehash(fixture, name)
    rejected(fixture)


@pytest.mark.parametrize("section,field", [("outcomes", "prefix_acquired"), ("outcomes", "phase2_gain"),
                                          ("outcomes", "phase2_raw_auc"), ("outcomes", "phase2_baseline_adjusted_auc"),
                                          ("outcomes", "qualified_adaptation"), ("paired_contrasts", "difference"),
                                          ("contrast_statistics", "mean"), ("branch_verdicts", "prefix_acquisition_count")])
def test_rehashed_acquisition_qualification_gain_auc_and_factorial_contrasts_are_recomputed(audited_fixture, tmp_path, section, field):
    fixture = clone(audited_fixture, tmp_path)
    summary = VERIFY.read_json(fixture["out"] / "summary.json")
    row = summary[section][0]
    old = row[field]
    row[field] = not old if type(old) is bool else old + .013
    replace_json(fixture, "summary.json", summary)
    rejected(fixture)


@pytest.mark.parametrize("name", ["H_I", "H_H"])
@pytest.mark.parametrize("field", ["criterion_met", "positive_seed_count", "mean_effect", "n_seeds"])
def test_each_primary_component_hypothesis_is_reconstructed_without_filtering_failed_prefix_seeds(audited_fixture, tmp_path, name, field):
    fixture = clone(audited_fixture, tmp_path)
    summary = VERIFY.read_json(fixture["out"] / "summary.json")
    old = summary["primary_hypotheses"][name][field]
    summary["primary_hypotheses"][name][field] = not old if type(old) is bool else old + 1
    replace_json(fixture, "summary.json", summary)
    rejected(fixture)


@pytest.mark.parametrize("filename,field", [("stages.json", "actual_reward_count"),
                                           ("stages.json", "cumulative_head_updates"),
                                           ("model_resources.json", "head_update_calls"),
                                           ("clone_evidence.json", "inherited_internal_updates")])
def test_resource_and_clone_metadata_cannot_count_copied_prefixes_as_new_training(audited_fixture, tmp_path, filename, field):
    fixture = clone(audited_fixture, tmp_path)
    rows = VERIFY.read_json(fixture["out"] / filename)
    rows[0][field] += 1
    replace_json(fixture, filename, rows)
    rejected(fixture)


@pytest.mark.parametrize("mode", ["duplicate_prefix", "drop_failure"])
def test_duplicate_prefix_trials_and_removed_failed_cases_are_not_accepted_after_rehash(audited_fixture, tmp_path, mode):
    fixture = clone(audited_fixture, tmp_path)
    if mode == "duplicate_prefix":
        rows = read_csv(fixture, "seed_17/trials.csv")
        rows.append(dict(rows[0]))
        replace_csv(fixture, "seed_17/trials.csv", rows)
    else:
        summary = VERIFY.read_json(fixture["out"] / "summary.json")
        assert summary["failures"]
        summary["failures"].pop()
        replace_json(fixture, "summary.json", summary)
    rejected(fixture)


def test_rehashed_factorial_interaction_is_reconstructed_separately_from_simple_component_differences(audited_fixture, tmp_path):
    fixture = clone(audited_fixture, tmp_path)
    summary = VERIFY.read_json(fixture["out"] / "summary.json")
    row = next(r for r in summary["paired_contrasts"] if r["comparison"] == "interaction" and r["metric"] == "phase2_end")
    row["difference"] += .013
    replace_json(fixture, "summary.json", summary)
    rejected(fixture)


def test_frozen_branch_cannot_be_silently_reset_to_unlearned_chance_state(audited_fixture, tmp_path):
    fixture = clone(audited_fixture, tmp_path)
    name = "seed_17/linear/frozen/phase2_0008.npz"
    with np.load(fixture["out"] / name, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    assert np.any(arrays["readout_weights"])
    arrays["readout_weights"][:] = 0.
    np.savez_compressed(fixture["out"] / name, **arrays)
    rehash(fixture, name)
    rejected(fixture)


def test_independent_verifier_never_calls_producer_learning_forward_masks_or_summary(audited_fixture, monkeypatch):
    from drosolearn import learning_ablation as producer
    def forbidden(*args, **kwargs):
        raise AssertionError("Producer called by independent verification")
    for name in ("make_splits", "selected_update", "summarize", "branch_masks", "evaluate", "original_state_pack", "gate", "component_hypothesis"):
        monkeypatch.setattr(producer, name, forbidden)
    for learner in (producer.LinearLearner, producer.NormalizedLearner):
        for name in ("features", "raw_activities", "probabilities", "update", "snapshot"):
            monkeypatch.setattr(learner, name, forbidden)
    report = VERIFY.verify(audited_fixture["root"], audited_fixture["registration"], audited_fixture["out"])
    assert report["passed"], report["errors"]
