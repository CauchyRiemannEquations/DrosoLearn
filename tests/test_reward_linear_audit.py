"""Independent arithmetic acceptance and rehashed semantic corruption checks."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil

import numpy as np
import pytest

from drosolearn import reward_linear as producer
from drosolearn.reward_linear_audit import verify
from test_reward_linear import prepare_reward_fixture


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audited_reward_fixture(tmp_path_factory):
    root, config, registration = prepare_reward_fixture(tmp_path_factory.mktemp("reward-audit"))
    output = root / "fixture-output"
    producer.run(root, config, registration, output)
    report = verify(output)
    assert report["passed"], report["errors"]
    return output


def clone(fixture, path):
    root = path / "tampered-study"
    shutil.copytree(fixture.parent, root)
    return root / fixture.name


def rehash(output, relative):
    path = output / "artifact_hashes.json"
    hashes = json.loads(path.read_text(encoding="utf-8"))
    hashes[relative] = hashlib.sha256((output / relative).read_bytes()).hexdigest()
    path.write_text(json.dumps(hashes), encoding="utf-8")


def save_json(output, relative, data):
    (output / relative).write_text(json.dumps(data), encoding="utf-8")
    rehash(output, relative)


def arrays(output, relative):
    with np.load(output / relative, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def save_arrays(output, relative, data):
    np.savez_compressed(output / relative, **data)
    rehash(output, relative)


def rows(output, relative):
    with (output / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_rows(output, relative, data):
    with (output / relative).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)
    rehash(output, relative)


def change(value):
    if type(value) is bool:
        return not value
    if isinstance(value, list):
        value[0] = change(value[0])
        return value
    if isinstance(value, str):
        return value + " tampered"
    return value + .013


def corrupt_config(output, field, value):
    metadata = json.loads((output / "metadata.json").read_text())
    registration = json.loads((output / "registration.json").read_text())
    config = dict(metadata["config"])
    config[field] = value
    path = output.parent / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    metadata["config"] = config
    registration["config"] = config
    registration["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(output, "registration.json", registration)
    metadata["registration_sha256"] = hashlib.sha256((output / "registration.json").read_bytes()).hexdigest()
    save_json(output, "metadata.json", metadata)


def test_all_nine_reward_learners_and_every_case_replay(audited_reward_fixture):
    report = verify(audited_reward_fixture)
    assert report["passed"], report["errors"]
    counts = report["counts"]
    assert counts["conditions"] == counts["actual_fits"] == 9
    assert counts["trials"] == 9 * 8
    assert counts["checkpoints"] == 9 * 3
    assert counts["curve_rows"] == 9 * 3 * 2
    assert counts["transfer_files"] == 9 * 2
    assert counts["case_probability_entries"] == 9 * 2 * 4 * 27 * 8
    assert counts["candidate_score_entries"] == 9 * 2 * 27 * 8 * 2 * 14
    assert counts["primary_unseen_pairs"] == 15


def test_audit_does_not_import_any_producer_or_model():
    tree = ast.parse((ROOT / "src/drosolearn/reward_linear_audit.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert {node.module for node in imports} == {"datetime", "itertools", "pathlib", "encoding_audit", "linearity_audit"}
    for node in imports:
        if node.module == "linearity_audit":
            assert [alias.name for alias in node.names] == ["_source_projection"]
    assert {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)} == {"json", "numpy"}


def test_audit_never_calls_producer_learning_or_metrics(audited_reward_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Producer called during independent audit")
    for name in ("gap_rms", "snapshot", "make_splits", "policy_metrics", "transfer_evaluate", "summarize"):
        monkeypatch.setattr(producer, name, forbidden)
    for cls in (producer.LinearLearner, producer.NormalizedLearner):
        monkeypatch.setattr(cls, "features", forbidden)
        monkeypatch.setattr(cls, "update", forbidden)
    report = verify(audited_reward_fixture)
    assert report["passed"], report["errors"]


def test_audit_preserves_primary_and_frozen_source_bytes(audited_reward_fixture):
    manifest = json.loads((audited_reward_fixture / "artifact_hashes.json").read_text())
    registration = json.loads((audited_reward_fixture / "registration.json").read_text())
    assert verify(audited_reward_fixture)["passed"]
    assert manifest == {name: hashlib.sha256((audited_reward_fixture / name).read_bytes()).hexdigest() for name in manifest}
    assert registration["source_hashes"] == {name: hashlib.sha256((audited_reward_fixture.parent / name).read_bytes()).hexdigest()
                                             for name in registration["source_hashes"]}


@pytest.mark.parametrize("field,value", [
    ("seeds", [6001]), ("conditions", ["linear_normalized", "linear_random_reward"]),
    ("experiment_id", "different"), ("graph", "data/other_graph"),
    ("head_initialization", "random"), ("feature_normalization", "evaluation_features"),
    ("random_reward", "copied_correct_reward"),
])
def test_rehashed_fixture_alias_and_model_scope_rejected(audited_reward_fixture, tmp_path, field, value):
    output = clone(audited_reward_fixture, tmp_path)
    corrupt_config(output, field, value)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("condition", ["linear_raw", "linear_normalized", "linear_random_reward", "plastic"])
@pytest.mark.parametrize("field", ["feature_scale", "initial_feature_gap_rms", "decision_activity", "choice_probability", "readout_weights"])
def test_rehashed_normalization_and_state_recomputed(audited_reward_fixture, tmp_path, condition, field):
    output = clone(audited_reward_fixture, tmp_path)
    relative = f"seed_81/{condition}/checkpoint_0008.npz"
    data = arrays(output, relative)
    data[field].flat[0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["feature_scale", "initial_feature_gap_rms", "readout_parameters", "internal_parameters", "reward_rule"])
def test_rehashed_resource_metadata_recomputed(audited_reward_fixture, tmp_path, field):
    output = clone(audited_reward_fixture, tmp_path)
    data = json.loads((output / "model_resources.json").read_text())
    data[0][field] = change(data[0][field])
    save_json(output, "model_resources.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("field", ["action_uniforms", "reward_uniforms"])
def test_rehashed_action_and_random_reward_streams_recomputed(audited_reward_fixture, tmp_path, field):
    output = clone(audited_reward_fixture, tmp_path)
    relative = "seed_81/stimuli.npz"
    data = arrays(output, relative)
    data[field].flat[0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("condition", ["linear_normalized", "linear_random_reward", "plastic"])
@pytest.mark.parametrize("field", ["action", "reward", "p_right", "optimizer_factor", "readout_step_l2", "internal_step_l2", "decision_left_l2"])
def test_rehashed_reward_and_gradient_trials_replayed(audited_reward_fixture, tmp_path, condition, field):
    output = clone(audited_reward_fixture, tmp_path)
    relative = "seed_81/trials.csv"
    data = rows(output, relative)
    row = next(row for row in data if row["condition"] == condition and row["trial"] == "2")
    row[field] = 1 - int(row[field]) if field in ("action", "reward") else float(row[field]) + .013
    save_rows(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("condition", ["linear_raw", "linear_normalized", "linear_random_reward", "plastic"])
@pytest.mark.parametrize("field", ["code_values", "score_cache", "map_p_target", "map_greedy_accuracy", "map_p_right"])
def test_rehashed_every_transfer_type_recomputed(audited_reward_fixture, tmp_path, condition, field):
    output = clone(audited_reward_fixture, tmp_path)
    relative = f"seed_81/{condition}/transfer_final.npz"
    data = arrays(output, relative)
    data[field].flat[0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("section,field", [
    ("outcomes", "before_accuracy"), ("outcomes", "after_accuracy"), ("outcomes", "gain"), ("outcomes", "acquired"),
    ("outcomes", "ood_accuracy"), ("outcomes", "head_updates"), ("outcomes", "internal_updates"),
    ("outcomes", "focal_before"), ("outcomes", "focal_after"), ("outcomes", "focal_greedy_after"),
    ("outcomes", "unseen_before"), ("outcomes", "unseen_after"), ("outcomes", "anchored_after"),
    ("outcomes", "all_after"), ("outcomes", "qualified_unseen_transfer"), ("outcomes", "qualified_focal_transfer"),
    ("family_verdicts", "n_seeds"), ("family_verdicts", "acquisition_count"),
    ("family_verdicts", "acquisition_criterion_met"), ("family_verdicts", "joint_identity_transfer_count"),
    ("family_verdicts", "joint_transfer_criterion_met"), ("family_verdicts", "mean_after_accuracy"),
    ("family_verdicts", "sd_after_accuracy"), ("family_verdicts", "mean_unseen_after"),
    ("paired_reward_contrasts", "iid_reward_gap"),
])
def test_rehashed_all_summary_claims_recomputed(audited_reward_fixture, tmp_path, section, field):
    output = clone(audited_reward_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    data[section][0][field] = change(data[section][0][field])
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


@pytest.mark.parametrize("hypothesis,field", [
    ("H_R", "criterion_met"), ("H_R", "mean_iid_reward_gap"), ("H_R", "positive_seed_count"),
    ("H_R", "acquisition_count"), ("H_G", "criterion_met"), ("H_G", "joint_identity_transfer_count"),
])
def test_rehashed_primary_hypotheses_recomputed(audited_reward_fixture, tmp_path, hypothesis, field):
    output = clone(audited_reward_fixture, tmp_path)
    data = json.loads((output / "summary.json").read_text())
    data["hypotheses"][hypothesis][field] = change(data["hypotheses"][hypothesis][field])
    save_json(output, "summary.json", data)
    report = verify(output)
    assert not report["passed"] and report["errors"]


def test_rehashed_nuisance_changes_cannot_hide_new_normalization_data(audited_reward_fixture, tmp_path):
    output = clone(audited_reward_fixture, tmp_path)
    relative = "seed_81/stimuli.npz"
    data = arrays(output, relative)
    data["train_nuisance"][0, 0, 0] += .013
    save_arrays(output, relative, data)
    report = verify(output)
    assert not report["passed"] and report["errors"]
